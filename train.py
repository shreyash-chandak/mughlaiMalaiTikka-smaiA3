import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from app import MODEL_NAME, PROMPT_ENSEMBLES, SPECIALIST_CLUSTER, extract_image_features, extract_text_features, normalize


SEED = 42


@dataclass
class ImageRecord:
    """Store a single image path together with its monument label."""

    image_path: str
    monument_name: str


class MughalMonumentDataset(Dataset):
    """Load monument images and pair them with CLIP-compatible text prompts.

    Args:
        records: Image records to expose through the dataset.
        transform: Optional torchvision transform applied to each PIL image.
    """

    def __init__(self, records: list[ImageRecord], transform: transforms.Compose | None = None) -> None:
        """Store dataset records and an optional image transform.

        Args:
            records: Image records to expose through the dataset.
            transform: Optional torchvision transform applied to each PIL image.
        """

        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of image records in the dataset."""

        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, str]:
        """Load one image sample and the exact training text template.

        Args:
            index: Dataset index.

        Returns:
            tuple[Image.Image, str, str]: Image, paired text prompt, and class label.
        """

        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        text = f"a photograph of {record.monument_name}, a Mughal monument"
        return image, text, record.monument_name


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Fine-tune CLIP for Mughal monument identification.")
    parser.add_argument("--data_dir", default="./data", help="Directory containing class folders of images.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument(
        "--output_dir",
        default="./clip_mughal_finetuned",
        help="Directory where the best checkpoint and training log will be saved.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training splits and shuffling.

    Args:
        seed: Random seed value.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation image transforms.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: Training and validation transforms.
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    return train_transform, val_transform


def collect_image_records(data_dir: str) -> tuple[list[ImageRecord], Counter]:
    """Collect image paths from the dataset directory using supported class-folder names.

    Args:
        data_dir: Root data directory.

    Returns:
        tuple[list[ImageRecord], Counter]: Image records and class-count statistics.
    """

    image_records: list[ImageRecord] = []
    class_counts: Counter = Counter()
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    supported_classes = set(SPECIALIST_CLUSTER)

    if not os.path.isdir(data_dir):
        print(f"Warning: data directory '{data_dir}' does not exist. No images were found.")
        return image_records, class_counts

    for root, _, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        if class_name not in supported_classes:
            continue

        for file_name in sorted(files):
            image_path = os.path.join(root, file_name)
            if os.path.isfile(image_path) and os.path.splitext(file_name)[1].lower() in valid_suffixes:
                image_records.append(ImageRecord(image_path=image_path, monument_name=class_name))
                class_counts[class_name] += 1

    return image_records, class_counts


def warn_underpopulated_classes(class_names: list[str], class_counts: Counter) -> None:
    """Print a warning for classes with fewer than 10 images.

    Args:
        class_names: Supported class names.
        class_counts: Image counts per class.
    """

    underpopulated = [name for name in class_names if class_counts.get(name, 0) < 10]
    if underpopulated:
        print("Warning: the following classes have fewer than 10 images:")
        for monument_name in underpopulated:
            print(f"  - {monument_name}: {class_counts.get(monument_name, 0)} images")


def split_records(records: list[ImageRecord]) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split records into train and validation sets with stratification when possible.

    Args:
        records: Full set of discovered image records.

    Returns:
        tuple[list[ImageRecord], list[ImageRecord]]: Train and validation splits.
    """

    if len(records) < 2:
        return records, []

    labels = [record.monument_name for record in records]
    label_counts = Counter(labels)
    can_stratify = all(count >= 2 for count in label_counts.values()) and len(label_counts) >= 2

    if can_stratify:
        train_records, val_records = train_test_split(
            records,
            test_size=0.2,
            random_state=SEED,
            shuffle=True,
            stratify=labels,
        )
        return list(train_records), list(val_records)

    print("Warning: stratified split was not possible for every class; using a class-aware fallback split.")
    grouped: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.monument_name].append(record)

    rng = random.Random(SEED)
    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []

    for class_records in grouped.values():
        rng.shuffle(class_records)
        if len(class_records) == 1:
            train_records.extend(class_records)
            continue

        val_count = max(1, int(round(0.2 * len(class_records))))
        val_count = min(val_count, len(class_records) - 1)
        val_records.extend(class_records[:val_count])
        train_records.extend(class_records[val_count:])

    return train_records, val_records


def collate_batch(
    batch: list[tuple[Image.Image, str, str]],
    processor: CLIPProcessor,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert a list of dataset items into CLIP processor tensors.

    Args:
        batch: Batch of dataset examples.
        processor: CLIP processor used to tokenize text and preprocess images.

    Returns:
        tuple[dict[str, torch.Tensor], list[str]]: Batched processor outputs and class labels.
    """

    images, texts, labels = zip(*batch)
    encoded = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
    return encoded, list(labels)


def contrastive_loss(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute the symmetric CLIP-style InfoNCE loss.

    Args:
        image_features: Normalized image embeddings.
        text_features: Normalized text embeddings.
        temperature: Temperature used to scale logits.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """

    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def freeze_for_finetuning(model: CLIPModel) -> None:
    """Freeze CLIP parameters except the last three vision blocks and visual projection.

    Args:
        model: CLIP model to prepare for fine-tuning.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_index in (9, 10, 11):
        for parameter in model.vision_model.encoder.layers[layer_index].parameters():
            parameter.requires_grad = True

    for parameter in model.visual_projection.parameters():
        parameter.requires_grad = True


def print_parameter_summary(model: CLIPModel) -> None:
    """Print total, trainable, and frozen parameter counts.

    Args:
        model: CLIP model whose parameters should be summarized.
    """

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")


def move_batch_to_device(batch_inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move processor outputs onto the active training device.

    Args:
        batch_inputs: Processor output tensors.
        device: Target device.

    Returns:
        dict[str, torch.Tensor]: Device-mapped batch tensors.
    """

    return {key: value.to(device) for key, value in batch_inputs.items()}


@torch.no_grad()
def encode_prompt_ensembles(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    class_names: list[str],
) -> tuple[list[str], torch.Tensor]:
    """Encode prompt ensembles into one normalized embedding per class.

    Args:
        model: CLIP model used for text encoding.
        processor: CLIP processor used for tokenization.
        device: Torch device for feature extraction.
        class_names: Available class names that should be encoded.

    Returns:
        tuple[list[str], torch.Tensor]: Class names and normalized class embeddings.
    """

    class_embeddings: list[torch.Tensor] = []

    for monument_name in class_names:
        encoded = processor(
            text=PROMPT_ENSEMBLES[monument_name],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = move_batch_to_device(encoded, device)
        text_features = extract_text_features(model, encoded)
        text_features = normalize(text_features)
        class_embeddings.append(normalize(text_features.mean(dim=0, keepdim=True)).squeeze(0))

    return class_names, torch.stack(class_embeddings, dim=0)


@torch.no_grad()
def evaluate_zero_shot_top1(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> float:
    """Evaluate validation accuracy with the prompt ensembles from ``app.py``.

    Args:
        model: CLIP model being fine-tuned.
        processor: CLIP processor for validation image preprocessing.
        dataloader: Validation dataloader.
        device: Torch device for evaluation.
        class_names: Available class names that should be evaluated.

    Returns:
        float: Top-1 validation accuracy in ``[0, 1]``.
    """

    if len(dataloader.dataset) == 0:
        return 0.0

    class_names, class_embeddings = encode_prompt_ensembles(model, processor, device, class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}

    correct = 0
    total = 0
    for batch_inputs, labels in dataloader:
        image_only_inputs = {"pixel_values": batch_inputs["pixel_values"].to(device)}
        image_features = extract_image_features(model, image_only_inputs["pixel_values"])
        image_features = normalize(image_features)
        logits = image_features @ class_embeddings.T
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()

        for predicted_index, label in zip(predictions, labels):
            correct += int(predicted_index == class_to_index[label])
            total += 1

    return correct / total if total else 0.0


def save_checkpoint(
    model: CLIPModel,
    processor: CLIPProcessor,
    output_dir: str,
    log_payload: dict[str, Any],
) -> None:
    """Save the best fine-tuned checkpoint and training log to disk.

    Args:
        model: Model to save.
        processor: Matching processor to save.
        output_dir: Target checkpoint directory.
        log_payload: JSON-serializable training metadata.
    """

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    write_training_log(output_dir, log_payload)


def write_training_log(output_dir: str, log_payload: dict[str, Any]) -> None:
    """Write training metadata without touching saved model weights.

    Args:
        output_dir: Directory containing the checkpoint.
        log_payload: JSON-serializable training metadata.
    """

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log_payload, handle, indent=2)


def train_one_epoch(
    model: CLIPModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Run one training epoch and return the mean loss.

    Args:
        model: CLIP model being fine-tuned.
        dataloader: Training dataloader.
        optimizer: Optimizer for trainable parameters.
        scaler: Mixed-precision gradient scaler.
        device: Torch device for training.
        use_amp: Whether AMP should be enabled.

    Returns:
        float: Mean training loss for the epoch.
    """

    model.train()
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_inputs, _ in progress_bar:
        batch_inputs = move_batch_to_device(batch_inputs, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(**batch_inputs)
            image_features = normalize(outputs.image_embeds)
            text_features = normalize(outputs.text_embeds)
            loss = contrastive_loss(image_features, text_features)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_batches if total_batches else 0.0


def build_dataloaders(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    processor: CLIPProcessor,
    batch_size: int,
    use_cuda: bool,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        train_records: Training split records.
        val_records: Validation split records.
        processor: CLIP processor for collate-time preprocessing.
        batch_size: Dataloader batch size.
        use_cuda: Whether CUDA is available for faster host-to-device transfer.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """

    train_transform, val_transform = build_transforms()
    train_dataset = MughalMonumentDataset(train_records, transform=train_transform)
    val_dataset = MughalMonumentDataset(val_records, transform=val_transform)
    collate_fn = partial(collate_batch, processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def main() -> None:
    """Run the full CLIP fine-tuning workflow from the command line."""

    args = parse_args()
    set_seed(SEED)

    expected_training_classes = SPECIALIST_CLUSTER
    records, class_counts = collect_image_records(args.data_dir)
    class_names = sorted(set(record.monument_name for record in records))
    warn_underpopulated_classes(expected_training_classes, class_counts)

    if len(records) < 2:
        print("Not enough images were found to create a train/validation split. Exiting without training.")
        return

    train_records, val_records = split_records(records)
    print(f"Discovered {len(records)} images across {sum(count > 0 for count in class_counts.values())} populated classes.")
    print(f"Train images: {len(train_records)} | Val images: {len(val_records)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    freeze_for_finetuning(model)
    print_parameter_summary(model)

    train_loader, val_loader = build_dataloaders(train_records, val_records, processor, args.batch_size, use_amp)

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-6, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_val_top1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch_index in range(args.epochs):
        epochs_ran = epoch_index + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        model.eval()
        val_top1 = evaluate_zero_shot_top1(model, processor, val_loader, device, class_names)
        scheduler.step()

        print(f"Epoch {epoch_index + 1} | train_loss: {train_loss:.4f} | val_top1: {val_top1 * 100:.1f}%")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
            training_log = {
                "epochs_run": epochs_ran,
                "best_val_top1": best_val_top1,
                "best_epoch": best_epoch,
                "class_names": class_names,
                "base_model": MODEL_NAME,
                "frozen_layers": "text_model + text_projection + vision layers 0-8",
                "timestamp": datetime.utcnow().isoformat(),
            }
            save_checkpoint(model, processor, args.output_dir, training_log)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 3:
            print("Early stopping: validation top-1 did not improve for 3 consecutive epochs.")
            break

    final_log = {
        "epochs_run": epochs_ran,
        "best_val_top1": max(best_val_top1, 0.0),
        "best_epoch": best_epoch,
        "class_names": class_names,
        "base_model": MODEL_NAME,
        "frozen_layers": "text_model + text_projection + vision layers 0-8",
        "timestamp": datetime.utcnow().isoformat(),
    }

    if best_epoch > 0:
        write_training_log(args.output_dir, final_log)
    else:
        print("Training finished without a validation improvement checkpoint. No model was saved.")


if __name__ == "__main__":
    main()
