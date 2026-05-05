# T12.4 Mughal Monument Identifier - Context Log

This file is maintained by Codex. Read it at the start of every session before
touching any code. It records every significant change to the codebase.

Project: SMAI Assignment 3, T12.4, IIIT Hyderabad 2025-26
Stack: Streamlit, CLIP (openai/clip-vit-base-patch32), PyTorch, HuggingFace
Entry point: app.py
Training script: train.py (created in Session 2)

## 2026-05-05 - Task 1: CLIP Fine-Tuning Pipeline

**What changed:** Created `train.py` to fine-tune the CLIP vision side on monument images from `./data/`, using the requested augmentation stack, contrastive loss, freezing policy, cosine scheduler, AMP support, zero-shot validation, best-checkpoint saving, and early stopping.
**Files modified:** `train.py`
**Key decisions:** Imported `PROMPT_ENSEMBLES` from `app.py` so training and inference use the same class prompts; froze everything first and then explicitly unfroze only vision layers 9-11 plus `visual_projection`; added a fallback class-aware split when strict stratification is impossible because some classes may have too few images.
**Known issues / TODO:** Training was scaffolded but not run end-to-end in this session because no local `./data/` training corpus was available for a real fit.

## 2026-05-05 - Task 2: Load Fine-Tuned Weights in app.py

**What changed:** Added checkpoint detection for `./clip_mughal_finetuned/`, loading of fine-tuned weights when `config.json` is present, loading of `training_log.json`, and a model-status badge below the title.
**Files modified:** `app.py`
**Key decisions:** Added a `ModelStatus` dataclass and made `load_clip()` the active loader function so the UI can show either a green fine-tuned badge with best val top-1 or a grey base-model badge.
**Known issues / TODO:** The app still falls back to base CLIP unless a valid saved fine-tuned checkpoint exists locally.

## 2026-05-05 - Task 3: Out-of-Domain Detection

**What changed:** Added `OOD_THRESHOLD`, OOD flagging inside `predict()`, and a dedicated OOD result panel with supported-monument chips and a collapsed raw-score expander.
**Files modified:** `app.py`
**Key decisions:** Kept the current visual language and two-column layout while swapping the right-hand result panel when the top probability stays below the threshold.
**Known issues / TODO:** The threshold is intentionally exposed as a top-level constant and may still need tuning after real fine-tuning and real-world testing.

## 2026-05-05 - Task 4: requirements.txt

**What changed:** Added `torchvision`, `scikit-learn`, and `tqdm` for the fine-tuning pipeline.
**Files modified:** `requirements.txt`
**Key decisions:** Left versions unpinned because there was no session-specific compatibility issue that required pinning.
**Known issues / TODO:** None.

## 2026-05-05 - Task 5: Context Log Maintenance

**What changed:** Created this `CONTEXT.md` file and appended entries for each task completed in this session.
**Files modified:** `CONTEXT.md`
**Key decisions:** Kept entries append-only and task-scoped so future sessions can quickly reconstruct what changed and why.
**Known issues / TODO:** Future sessions should continue appending here instead of splitting context across multiple logs.

## 2026-05-05 - India-Only Class Update and Hybrid Specialist Inference

**What changed:** Replaced the Pakistan classes `Lahore Fort` and `Badshahi Mosque` with `Buland Darwaza` and `Tomb of Salim Chishti` across `app.py` and `metadata.json`. Added `SPECIALIST_CLUSTER`, changed inference to always run base zero-shot CLIP first and then optionally refine cluster predictions with a fine-tuned specialist model checkpoint if one exists.
**Files modified:** `app.py`, `metadata.json`
**Key decisions:** Kept the base zero-shot model as the universal first-pass classifier for all 15 monuments, then restricted specialist refinement to the nine visually confusing classes named in the new brief. The status badge now reflects hybrid specialist availability instead of implying that the entire app runs only on the fine-tuned checkpoint.
**Known issues / TODO:** The specialist refinement path depends on a partial fine-tuned checkpoint being available in `./clip_mughal_finetuned/`; otherwise the app remains base zero-shot only.

## 2026-05-05 - Partial Fine-Tuning Dataset Logic

**What changed:** Updated `train.py` so training records are discovered from the dataset structure itself, limited to the specialist cluster, and the actual training/evaluation class list is now derived dynamically from `sorted(set(record.monument_name for record in records))`.
**Files modified:** `train.py`
**Key decisions:** Kept under-population warnings aligned to the expected specialist training folders, but kept train/val/eval class handling aligned to the classes that are actually present on disk.
**Known issues / TODO:** If no training images exist, the script exits cleanly after warnings; real specialist training still requires the dataset to be populated manually.

## 2026-05-05 - Dataset Skeleton and Report Sync

**What changed:** Created the empty `data/` folder skeleton for the nine specialist classes and synchronized the implementation notes with `REPORT_WORKLOG.md`.
**Files modified:** `CONTEXT.md`, `REPORT_WORKLOG.md`, `data/`
**Key decisions:** Used exact folder names from the code so training can read the dataset without any rename step.
**Known issues / TODO:** The directories are empty placeholders only and must be populated with images before fine-tuning.

## 2026-05-05 - Repeated Seeded Evaluation for train.py

**What changed:** Updated `train.py` to support repeated seeded runs via `--seed` and `--runs`, aggregate mean and standard deviation of validation top-1 across runs, print per-class accuracy, and cap each class at 80 images during collection to reduce imbalance effects.
**Files modified:** `train.py`, `requirements.txt`
**Key decisions:** Kept the existing fine-tuning setup intact and wrapped it in a small `run_training_once()` helper rather than rewriting the full pipeline. Checkpoint saving now preserves the globally best run across repeated executions, while `training_log.json` stores aggregate run statistics afterward.
**Known issues / TODO:** Repeated runs still reuse the same validation strategy rather than adding a separate test set, so the results remain an estimate rather than a final held-out benchmark.

## 2026-05-05 - Data-Aware Specialist Clusters and Rotating K-Fold Sync

**What changed:** Updated `app.py` so specialist prompt groups are no longer limited to the earlier white-marble subset. The app now defines four specialist clusters, filters them down to the monuments that actually have populated folders in `data/`, and uses those active groups for specialist reranking and fine-tuned specialist fallback loading. Updated `train.py` so under-population warnings now align with the classes that are actually present on disk, and clarified the logging around the existing rotating k-fold 70/15/15 train/val/test workflow.
**Files modified:** `app.py`, `train.py`, `CONTEXT.md`, `REPORT_WORKLOG.md`
**Key decisions:** Kept the full four-cluster prompt design in code so absent monuments can become active later without another refactor, but restricted the active specialist path to populated dataset folders to avoid training/app mismatch. Preserved the existing 7-fold rotating split design because it is the cleanest full k-fold approximation of a 70/15/15 train/validation/test allocation.
**Known issues / TODO:** Some requested cluster members still have no local images in `data/`, so they remain defined but inactive until those folders are populated.
