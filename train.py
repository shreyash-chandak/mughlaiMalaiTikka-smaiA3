import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from app import MODEL_NAME, PROMPT_ENSEMBLES, SPECIALIST_CLUSTER, extract_image_features, extract_text_features, normalize


APPROX_K_FOLDS = 7
TRAINABLE_VISION_BLOCKS = (7, 8, 9, 10, 11)


@dataclass
class ImageRecord:
    """Store a single image path together with its monument label."""

    image_path: str
    monument_name: str


class MughalMonumentDataset(Dataset):
    """Expose original and augmented copies of every image record.

    Args:
        records: Image records to expose through the dataset.
        original_transform: Transform used for the original copy of each image.
        augmented_transform: Transform used for the augmented copy of each image.
        include_augmented_copy: Whether to append an additional augmented sample per record.
        deterministic_seed_base: Optional base seed used to make augmented evaluation samples deterministic.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        original_transform: transforms.Compose,
        augmented_transform: transforms.Compose,
        include_augmented_copy: bool,
        deterministic_seed_base: int | None = None,
    ) -> None:
        """Store dataset records, transforms, and duplication behavior.

        Args:
            records: Image records to expose through the dataset.
            original_transform: Transform used for the original copy of each image.
            augmented_transform: Transform used for the augmented copy of each image.
            include_augmented_copy: Whether to append an additional augmented sample per record.
            deterministic_seed_base: Optional base seed used to make augmented evaluation samples deterministic.
        """

        self.records = records
        self.original_transform = original_transform
        self.augmented_transform = augmented_transform
        self.include_augmented_copy = include_augmented_copy
        self.deterministic_seed_base = deterministic_seed_base

    def __len__(self) -> int:
        """Return the number of samples, including augmented copies when enabled.

        Returns:
            int: Dataset length.
        """

        multiplier = 2 if self.include_augmented_copy else 1
        return len(self.records) * multiplier

    def __getitem__(self, index: int) -> tuple[Image.Image, str, str]:
        """Load one original or augmented sample and its CLIP training text.

        Args:
            index: Dataset index.

        Returns:
            tuple[Image.Image, str, str]: Image, paired text prompt, and class label.
        """

        record_index = index // 2 if self.include_augmented_copy else index
        use_augmented_copy = self.include_augmented_copy and index % 2 == 1
        record = self.records[record_index]
        image = Image.open(record.image_path).convert("RGB")

        if use_augmented_copy:
            image = self.apply_transform(image, self.augmented_transform, record_index)
        else:
            image = self.apply_transform(image, self.original_transform, record_index)

        text = f"a photograph of {record.monument_name}, a Mughal monument"
        return image, text, record.monument_name

    def apply_transform(self, image: Image.Image, transform: transforms.Compose, record_index: int) -> Image.Image:
        """Apply a transform, optionally with deterministic randomness for eval splits.

        Args:
            image: Input PIL image.
            transform: Transform pipeline to apply.
            record_index: Index of the underlying record.

        Returns:
            Image.Image: Transformed image.
        """

        if self.deterministic_seed_base is None:
            return transform(image)

        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        deterministic_seed = self.deterministic_seed_base + record_index
        random.seed(deterministic_seed)
        torch.manual_seed(deterministic_seed)
        transformed = transform(image)
        random.setstate(random_state)
        torch.random.set_rng_state(torch_state)
        return transformed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Fine-tune CLIP for Mughal monument identification.")
    parser.add_argument("--data_dir", default="./data", help="Directory containing class folders of images.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for repeated k-fold runs.")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated k-fold cycles to execute.")
    parser.add_argument(
        "--output_dir",
        default="./clip_mughal_finetuned",
        help="Directory where the best checkpoint and training log will be saved.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training, splits, and augment sampling.

    Args:
        seed: Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create original and augmented transform pipelines.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: Original-view and augmented-view transforms.
    """

    original_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    augmented_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.35),
        ]
    )
    return original_transform, augmented_transform


def collect_image_records(data_dir: str) -> tuple[list[ImageRecord], Counter]:
    """Collect image paths from the dataset directory using supported class-folder names.

    Args:
        data_dir: Root data directory.

    Returns:
        tuple[list[ImageRecord], Counter]: Image records and class-count statistics.
    """

    class_counts: Counter = Counter()
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    supported_classes = set(SPECIALIST_CLUSTER)
    grouped_paths: defaultdict[str, list[str]] = defaultdict(list)

    if not os.path.isdir(data_dir):
        print(f"Warning: data directory '{data_dir}' does not exist. No images were found.")
        return [], class_counts

    for root, _, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        if class_name not in supported_classes:
            continue

        for file_name in sorted(files):
            image_path = os.path.join(root, file_name)
            if os.path.isfile(image_path) and os.path.splitext(file_name)[1].lower() in valid_suffixes:
                grouped_paths[class_name].append(image_path)

    image_records: list[ImageRecord] = []
    for class_name, image_paths in grouped_paths.items():
        if len(image_paths) > 80:
            image_paths = random.sample(image_paths, 80)

        for image_path in sorted(image_paths):
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


def build_fold_splits(records: list[ImageRecord], seed: int) -> list[tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]]:
    """Build approximate 70/15/15 train/val/test folds using 7-way stratified rotation.

    Args:
        records: Full set of discovered image records.
        seed: Random seed used to shuffle folds reproducibly.

    Returns:
        list[tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]]:
            Rotating train/validation/test fold triples.
    """

    if len(records) < 3:
        return []

    labels = [record.monument_name for record in records]
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values()) if label_counts else 0
    if min_class_count < 3:
        return []

    n_splits = min(APPROX_K_FOLDS, min_class_count)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_indices = np.arange(len(records))
    fold_indices = [test_indices.tolist() for _, test_indices in splitter.split(all_indices, labels)]

    fold_splits: list[tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]] = []
    for fold_index in range(n_splits):
        test_indices = set(fold_indices[fold_index])
        val_indices = set(fold_indices[(fold_index + 1) % n_splits])
        train_indices = [
            index
            for index in range(len(records))
            if index not in test_indices and index not in val_indices
        ]

        train_records = [records[index] for index in train_indices]
        val_records = [records[index] for index in sorted(val_indices)]
        test_records = [records[index] for index in sorted(test_indices)]
        fold_splits.append((train_records, val_records, test_records))

    return fold_splits


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
    """Freeze CLIP parameters except the last five vision blocks and visual projection.

    Args:
        model: CLIP model to prepare for fine-tuning.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_index in TRAINABLE_VISION_BLOCKS:
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
    split_name: str,
) -> tuple[float, dict[str, float]]:
    """Evaluate top-1 accuracy and print a per-class breakdown for a split.

    Args:
        model: CLIP model being fine-tuned.
        processor: CLIP processor for evaluation image preprocessing.
        dataloader: Evaluation dataloader.
        device: Torch device for evaluation.
        class_names: Available class names that should be evaluated.
        split_name: Human-readable name of the evaluated split.

    Returns:
        tuple[float, dict[str, float]]: Overall accuracy and per-class accuracies.
    """

    if len(dataloader.dataset) == 0:
        return 0.0, {}

    class_names, class_embeddings = encode_prompt_ensembles(model, processor, device, class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}

    correct = 0
    total = 0
    class_correct: defaultdict[str, int] = defaultdict(int)
    class_total: defaultdict[str, int] = defaultdict(int)

    for batch_inputs, labels in dataloader:
        image_only_inputs = {"pixel_values": batch_inputs["pixel_values"].to(device)}
        image_features = extract_image_features(model, image_only_inputs["pixel_values"])
        image_features = normalize(image_features)
        logits = image_features @ class_embeddings.T
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()

        for predicted_index, label in zip(predictions, labels):
            if predicted_index == class_to_index[label]:
                class_correct[label] += 1
            class_total[label] += 1
            correct += int(predicted_index == class_to_index[label])
            total += 1

    per_class_accuracy = {
        class_name: class_correct[class_name] / class_total[class_name]
        for class_name in class_total
    }
    print(f"\nPer-class accuracy ({split_name}):")
    for class_name in sorted(class_total):
        print(f"{class_name}: {per_class_accuracy[class_name] * 100:.2f}%")

    return (correct / total if total else 0.0), per_class_accuracy


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
    test_records: list[ImageRecord],
    processor: CLIPProcessor,
    batch_size: int,
    use_cuda: bool,
    eval_seed_base: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with original and augmented copies.

    Args:
        train_records: Training split records.
        val_records: Validation split records.
        test_records: Test split records.
        processor: CLIP processor for collate-time preprocessing.
        batch_size: Dataloader batch size.
        use_cuda: Whether CUDA is available for faster host-to-device transfer.
        eval_seed_base: Base seed used to make val/test augmented copies deterministic.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test dataloaders.
    """

    original_transform, augmented_transform = build_transforms()
    train_dataset = MughalMonumentDataset(
        records=train_records,
        original_transform=original_transform,
        augmented_transform=augmented_transform,
        include_augmented_copy=True,
        deterministic_seed_base=None,
    )
    val_dataset = MughalMonumentDataset(
        records=val_records,
        original_transform=original_transform,
        augmented_transform=augmented_transform,
        include_augmented_copy=True,
        deterministic_seed_base=eval_seed_base,
    )
    test_dataset = MughalMonumentDataset(
        records=test_records,
        original_transform=original_transform,
        augmented_transform=augmented_transform,
        include_augmented_copy=True,
        deterministic_seed_base=eval_seed_base + 10000,
    )
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def run_fold_training(
    args: argparse.Namespace,
    run_index: int,
    fold_index: int,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    class_names: list[str],
    device: torch.device,
    use_amp: bool,
    global_best_val_top1: float,
    run_seed: int,
) -> tuple[dict[str, Any], float]:
    """Train and evaluate one fold, using validation for selection and test for reporting.

    Args:
        args: Parsed command-line arguments.
        run_index: One-based run index.
        fold_index: One-based fold index.
        train_records: Training split records.
        val_records: Validation split records.
        test_records: Test split records.
        class_names: Class names available in the dataset.
        device: Torch device for training and evaluation.
        use_amp: Whether mixed precision should be enabled.
        global_best_val_top1: Best validation score seen so far across all runs/folds.
        run_seed: Seed assigned to the enclosing run.

    Returns:
        tuple[dict[str, Any], float]: Fold summary and updated global best validation accuracy.
    """

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    freeze_for_finetuning(model)
    print_parameter_summary(model)

    eval_seed_base = run_seed * 100 + fold_index
    train_loader, val_loader, test_loader = build_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        processor=processor,
        batch_size=args.batch_size,
        use_cuda=use_amp,
        eval_seed_base=eval_seed_base,
    )
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-6, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_val_top1 = -1.0
    best_test_top1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0
    best_per_class_accuracy: dict[str, float] = {}

    for epoch_index in range(args.epochs):
        epochs_ran = epoch_index + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        model.eval()
        val_top1, _ = evaluate_zero_shot_top1(model, processor, val_loader, device, class_names, "validation")
        test_top1, test_per_class_accuracy = evaluate_zero_shot_top1(model, processor, test_loader, device, class_names, "test")
        scheduler.step()

        print(
            f"Fold {fold_index} | Epoch {epoch_index + 1} | "
            f"train_loss: {train_loss:.4f} | val_top1: {val_top1 * 100:.1f}% | test_top1: {test_top1 * 100:.1f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_test_top1 = test_top1
            best_epoch = epoch_index + 1
            best_per_class_accuracy = test_per_class_accuracy
            epochs_without_improvement = 0

            if val_top1 > global_best_val_top1:
                global_best_val_top1 = val_top1
                training_log = {
                    "epochs_run": epochs_ran,
                    "best_val_top1": best_val_top1,
                    "best_test_top1": best_test_top1,
                    "best_epoch": best_epoch,
                    "best_seed": run_seed,
                    "best_run": run_index,
                    "best_fold": fold_index,
                    "class_names": class_names,
                    "base_model": MODEL_NAME,
                    "frozen_layers": "text_model + text_projection + vision layers 0-6",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(model, processor, args.output_dir, training_log)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 3:
            print("Early stopping: validation top-1 did not improve for 3 consecutive epochs.")
            break

    return {
        "fold_index": fold_index,
        "best_val_top1": max(best_val_top1, 0.0),
        "best_test_top1": best_test_top1,
        "best_epoch": best_epoch,
        "epochs_run": epochs_ran,
        "per_class_accuracy": best_per_class_accuracy,
    }, global_best_val_top1


def run_training_once(
    args: argparse.Namespace,
    run_index: int,
    run_seed: int,
    device: torch.device,
    use_amp: bool,
    global_best_val_top1: float,
) -> tuple[dict[str, Any], float]:
    """Run one full k-fold cycle for a given seed.

    Args:
        args: Parsed command-line arguments.
        run_index: One-based run index for logging.
        run_seed: Random seed used for this run.
        device: Torch device for training and evaluation.
        use_amp: Whether mixed precision should be enabled.
        global_best_val_top1: Best validation accuracy seen across earlier runs/folds.

    Returns:
        tuple[dict[str, Any], float]: Run summary and updated global best validation accuracy.
    """

    set_seed(run_seed)
    records, class_counts = collect_image_records(args.data_dir)
    class_names = sorted(set(record.monument_name for record in records))
    warn_underpopulated_classes(class_names, class_counts)

    if len(records) < 3:
        return {
            "run_index": run_index,
            "seed": run_seed,
            "record_count": len(records),
            "class_names": class_names,
            "fold_results": [],
            "mean_val_top1": 0.0,
            "mean_test_top1": 0.0,
            "per_class_accuracy": {},
        }, global_best_val_top1

    fold_splits = build_fold_splits(records, run_seed)
    if not fold_splits:
        return {
            "run_index": run_index,
            "seed": run_seed,
            "record_count": len(records),
            "class_names": class_names,
            "fold_results": [],
            "mean_val_top1": 0.0,
            "mean_test_top1": 0.0,
            "per_class_accuracy": {},
        }, global_best_val_top1

    print(
        f"Discovered {len(records)} images across {sum(count > 0 for count in class_counts.values())} populated classes. "
        f"Using {len(fold_splits)} rotating folds for a full k-fold evaluation with an approximate 70/15/15 train/val/test split."
    )

    fold_results: list[dict[str, Any]] = []
    per_class_results: defaultdict[str, list[float]] = defaultdict(list)
    for fold_offset, (train_records, val_records, test_records) in enumerate(fold_splits, start=1):
        print(
            f"\n--- Run {run_index} | Fold {fold_offset}/{len(fold_splits)} ---\n"
            f"Train images: {len(train_records)} | Val images: {len(val_records)} | Test images: {len(test_records)}"
        )
        fold_summary, global_best_val_top1 = run_fold_training(
            args=args,
            run_index=run_index,
            fold_index=fold_offset,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            class_names=class_names,
            device=device,
            use_amp=use_amp,
            global_best_val_top1=global_best_val_top1,
            run_seed=run_seed,
        )
        fold_results.append(fold_summary)
        for class_name, class_accuracy in fold_summary["per_class_accuracy"].items():
            per_class_results[class_name].append(class_accuracy)

    mean_val_top1 = float(np.mean([summary["best_val_top1"] for summary in fold_results]))
    mean_test_top1 = float(np.mean([summary["best_test_top1"] for summary in fold_results]))
    aggregated_per_class_accuracy = {
        class_name: float(np.mean(class_accuracies))
        for class_name, class_accuracies in per_class_results.items()
    }

    return {
        "run_index": run_index,
        "seed": run_seed,
        "record_count": len(records),
        "class_names": class_names,
        "fold_results": fold_results,
        "mean_val_top1": mean_val_top1,
        "mean_test_top1": mean_test_top1,
        "per_class_accuracy": aggregated_per_class_accuracy,
    }, global_best_val_top1


def main() -> None:
    """Run repeated k-fold CLIP fine-tuning and aggregate validation/test metrics."""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    run_summaries: list[dict[str, Any]] = []
    global_best_val_top1 = -1.0

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        print(f"\n=== Run {run_idx + 1}/{args.runs} | seed={run_seed} ===")
        run_summary, global_best_val_top1 = run_training_once(
            args=args,
            run_index=run_idx + 1,
            run_seed=run_seed,
            device=device,
            use_amp=use_amp,
            global_best_val_top1=global_best_val_top1,
        )

        if run_summary["record_count"] < 3 or not run_summary["fold_results"]:
            print("Not enough images were found to build rotating k-fold train/val/test splits. Exiting without training.")
            return

        run_summaries.append(run_summary)
        print(
            f"Run {run_summary['run_index']}: "
            f"val={run_summary['mean_val_top1'] * 100:.1f}% | "
            f"test={run_summary['mean_test_top1'] * 100:.1f}%"
        )

    val_results = [summary["mean_val_top1"] for summary in run_summaries]
    test_results = [summary["mean_test_top1"] for summary in run_summaries]
    mean_val_acc = float(np.mean(val_results))
    std_val_acc = float(np.std(val_results))
    mean_test_acc = float(np.mean(test_results))
    std_test_acc = float(np.std(test_results))

    print(f"\nFinal Results over {args.runs} runs:")
    print(f"Mean Validation Accuracy: {mean_val_acc * 100:.2f}%")
    print(f"Validation Std Dev: +/-{std_val_acc * 100:.2f}%")
    print(f"Mean Test Accuracy: {mean_test_acc * 100:.2f}%")
    print(f"Test Std Dev: +/-{std_test_acc * 100:.2f}%")

    print("\nPer-class test accuracy across runs:")
    aggregated_per_class_accuracy: dict[str, dict[str, float]] = {}
    all_class_names = sorted({class_name for summary in run_summaries for class_name in summary["per_class_accuracy"]})
    for class_name in all_class_names:
        class_values = [
            summary["per_class_accuracy"][class_name]
            for summary in run_summaries
            if class_name in summary["per_class_accuracy"]
        ]
        class_mean = float(np.mean(class_values))
        class_std = float(np.std(class_values))
        aggregated_per_class_accuracy[class_name] = {
            "mean_accuracy": class_mean,
            "std_accuracy": class_std,
        }
        print(f"{class_name}: {class_mean * 100:.2f}% +/- {class_std * 100:.2f}%")

    final_log = {
        "runs": args.runs,
        "base_seed": args.seed,
        "folds_per_run": len(run_summaries[0]["fold_results"]) if run_summaries else 0,
        "run_results": run_summaries,
        "mean_val_top1": mean_val_acc,
        "std_val_top1": std_val_acc,
        "mean_test_top1": mean_test_acc,
        "std_test_top1": std_test_acc,
        "best_val_top1": max(
            fold_summary["best_val_top1"]
            for summary in run_summaries
            for fold_summary in summary["fold_results"]
        ),
        "class_names": run_summaries[0]["class_names"] if run_summaries else [],
        "per_class_accuracy": aggregated_per_class_accuracy,
        "base_model": MODEL_NAME,
        "frozen_layers": "text_model + text_projection + vision layers 0-6",
        "timestamp": datetime.utcnow().isoformat(),
    }
    write_training_log(args.output_dir, final_log)


if __name__ == "__main__":
    main()
