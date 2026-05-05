# Codex Task Brief — T12.4 Mughal Monument Identifier

You are working on an existing Streamlit application (`app.py`) that identifies
15 Mughal-era monuments using CLIP zero-shot classification. The codebase also
contains `metadata.json` (monument info), `requirements.txt`, and `README.md`.

Read the full `app.py` and `CONTEXT.md` (if it exists) before writing any code.
After completing every task below, update `CONTEXT.md` with what you changed,
why, and any decisions you made. This file is your running log — future sessions
will read it first.

---

## Task 1 — CLIP Fine-Tuning Pipeline

### Goal
Create a standalone training script `train.py` that fine-tunes the CLIP vision
encoder on labelled images of the 15 monuments, then saves the weights so
`app.py` can load them instead of the base model.

### Data sources (in priority order)
1. `./data/` directory — if the user has already placed images here, use them.
   Expect the folder structure: `data/<MonumentName>/image1.jpg`, etc.
   Monument folder names must exactly match the keys in `PROMPT_ENSEMBLES` inside
   `app.py` (e.g. `"Taj Mahal"`, `"Humayun's Tomb"`).
2. If `./data/` is missing or has fewer than 10 images for any class, print a
   clear warning listing which classes are under-populated. Do not crash — train
   on whatever is available.

### What the script must do

**Dataset class**
- Load images from `./data/<MonumentName>/`.
- Apply augmentation using `torchvision.transforms`:
  - `RandomHorizontalFlip()`
  - `RandomResizedCrop(224, scale=(0.7, 1.0))`
  - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`
  - `RandomGrayscale(p=0.05)` — helps with old/faded monument photos
- Pair each image with its class text using this exact template (matches
  inference prompts): `f"a photograph of {monument_name}, a Mughal monument"`
- 80/20 train/val split, stratified by class.

**Model setup**
- Load `openai/clip-vit-base-patch32`.
- **Freeze the text encoder entirely** (`model.text_model`). Do not unfreeze it
  under any circumstance — we have too few images to train it safely.
- **Freeze all vision encoder layers except the last 3 transformer blocks**
  (indices 9, 10, 11 of `model.vision_model.encoder.layers`). Unfreeze the
  vision projection layer (`model.visual_projection`) as well.
- Print a parameter count summary: total params, trainable params, frozen params.

**Loss function — InfoNCE / contrastive loss**
```python
def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```
Use this exactly. Do not substitute cross-entropy classification loss — the
contrastive formulation is what keeps CLIP's embedding space meaningful.

**Training hyperparameters**
- Learning rate: `1e-6` (AdamW). This is intentionally very low — do not raise
  it. CLIP pre-trained features are fragile at higher LRs.
- Weight decay: `0.01`
- Batch size: `32` (or largest that fits in GPU memory — add a `--batch_size`
  CLI arg defaulting to 32)
- Epochs: `10` (add a `--epochs` CLI arg defaulting to 10)
- LR scheduler: `CosineAnnealingLR(optimizer, T_max=epochs)`
- Mixed precision: use `torch.cuda.amp.autocast()` + `GradScaler` if CUDA is
  available, plain float32 if CPU-only (Colab T4 free tier must work).

**Validation loop**
After each epoch, run zero-shot top-1 accuracy on the val split:
- Use the same ensembled prompts from `PROMPT_ENSEMBLES` in `app.py` (import or
  copy the dict into `train.py`).
- Print: `Epoch N | train_loss: X.XXXX | val_top1: XX.X%`
- Save the best checkpoint (highest val_top1) to `./clip_mughal_finetuned/`.
  Use `model.save_pretrained()` and `processor.save_pretrained()`.
- Also save a `training_log.json` alongside it:
  ```json
  {
    "epochs_run": 10,
    "best_val_top1": 0.87,
    "best_epoch": 7,
    "class_names": ["Taj Mahal", ...],
    "base_model": "openai/clip-vit-base-patch32",
    "frozen_layers": "text_model + vision layers 0-8",
    "timestamp": "2025-..."
  }
  ```

**Early stopping**
Stop training if val_top1 does not improve for 3 consecutive epochs. Print a
message explaining why training stopped early.

### CLI interface for `train.py`
```
python train.py --data_dir ./data --epochs 10 --batch_size 32 --output_dir ./clip_mughal_finetuned
```
All arguments optional with the defaults above.

---

## Task 2 — Load Fine-Tuned Weights in `app.py`

Modify the `load_clip()` function in `app.py` so it:

1. Checks whether `./clip_mughal_finetuned/` exists AND contains a valid model
   (check for `config.json` inside the directory).
2. If yes: loads from `./clip_mughal_finetuned/` and shows a small green badge
   in the UI: `✦ Fine-tuned model active` with the best val accuracy read from
   `training_log.json`.
3. If no: falls back silently to `openai/clip-vit-base-patch32` (current
   behaviour). Show a grey badge: `◦ Base CLIP model (zero-shot)`.

Put both badges just below the hero banner so the user always knows which model
is running. Do not alter any other part of the UI.

---

## Task 3 — Out-of-Domain (OOD) Detection

The current app always returns the closest monument even when the image has
nothing to do with Mughal architecture (e.g. a selfie, a dog, a receipt).

Add a confidence threshold check to the `predict()` function and the result
display logic.

### How to implement it

In `predict()`, after computing `probs`, add:

```python
CONFIDENCE_THRESHOLD = 0.15  # tunable — see note below

top_prob = probs[0][1]  # highest probability after sorting
is_ood = top_prob < CONFIDENCE_THRESHOLD
return results, is_ood
```

**Choosing the threshold**: 0.15 is a starting point. With 15 classes, random
chance is 1/15 ≈ 6.7%. A genuine match typically scores above 30–40% after
temperature scaling. 15% sits safely between noise and signal. Add a comment
explaining this reasoning in the code.

**Important**: the threshold should be read from a constant at the top of the
file (name it `OOD_THRESHOLD`) so it is easy to adjust without hunting through
the code.

### What to show in the UI when OOD is detected

Replace the entire prediction result panel with this message block — keep the
same visual style as the rest of the app (use the existing CSS variables, do
not introduce new colours):

```
🕌  [large icon, centred]

Not a Mughal Monument
[in pred-name style, maroon]

This image does not appear to match any of the 15 Mughal monuments
in our database. For best results, upload a clear exterior or
interior photograph of one of the supported monuments.

[smaller text, stone colour]

Supported monuments: [the 15 chips, same style as the left panel]

[Confidence scores — show as a small collapsed expander titled
 "See raw scores (all low)" so the user can inspect if curious]
```

Do not show history, visit info, fun facts, or the Google Maps link when OOD.

---

## Task 4 — Update `requirements.txt`

Add any new dependencies introduced by `train.py`. At minimum expect:
- `torchvision` (for augmentation transforms)
- `scikit-learn` (for stratified split)
- `tqdm` (for training progress bars)

Do not remove existing entries. Pin versions only if a specific version is
required for a known compatibility reason — otherwise leave unpinned.

---

## Task 5 — Update `CONTEXT.md`

After every task above, append a dated entry to `CONTEXT.md` with the following
structure:

```markdown
## [DATE] — [Task name]

**What changed:** ...
**Files modified:** ...
**Key decisions:** ...
**Known issues / TODO:** ...
```

If `CONTEXT.md` does not exist yet, create it with a header section first:

```markdown
# T12.4 Mughal Monument Identifier — Context Log

This file is maintained by Codex. Read it at the start of every session before
touching any code. It records every significant change to the codebase.

Project: SMAI Assignment 3, T12.4, IIIT Hyderabad 2025-26
Stack: Streamlit, CLIP (openai/clip-vit-base-patch32), PyTorch, HuggingFace
Entry point: app.py
Training script: train.py (created in Session 2)
```

Treat `CONTEXT.md` as the ground truth for what state the codebase is in.
Never delete old entries — only append.

---

## Constraints and rules for all tasks

- **Do not change the UI design** (colours, fonts, layout, CSS variables). Only
  modify logic and text content.
- **Do not change `metadata.json`** — it is correct as-is.
- **Keep `PROMPT_ENSEMBLES` in `app.py` as the single source of truth** for
  prompt text. `train.py` should import or copy it, not redefine it differently.
- **Colab T4 free tier must work** — do not introduce dependencies that require
  a paid GPU or more than 15GB of RAM.
- **Every function you add or modify must have a docstring** explaining what it
  does, its arguments, and its return value.
- **Do not use `st.experimental_*` APIs** — they are deprecated.
- If you are unsure about a design decision, implement the simpler option and
  leave a `# TODO:` comment explaining the trade-off.