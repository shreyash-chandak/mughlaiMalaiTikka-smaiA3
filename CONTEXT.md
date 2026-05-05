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

## 2026-05-06 - Baseline Evaluator and Safer Specialist Training Objective

**What changed:** Added `evaluate.py` for local/Kaggle evaluation of base CLIP or a saved checkpoint, including all-class and specialist-only scopes, duplicate hash reporting, per-class accuracy, confusion matrix export, and prediction JSON output. Updated `train.py` to fine-tune with supervised image-to-class-prompt cross entropy instead of batchwise image-text InfoNCE, and aligned the freeze policy to the requested last three vision blocks plus `visual_projection`. Added `KAGGLE_RUN.md` with copy-paste Kaggle cells.
**Files modified:** `train.py`, `evaluate.py`, `KAGGLE_RUN.md`, `CONTEXT.md`, `REPORT_WORKLOG.md`
**Key decisions:** Kept `Qutub Minar` and `Shalimar Bagh` outside specialist fine-tuning because they are visually distinctive and already handled well by base zero-shot CLIP. The 13-class specialist model now trains images against one prompt-ensemble embedding per class, avoiding false-negative pressure between same-class images in the same batch.
**Known issues / TODO:** Need Kaggle results from `evaluate.py` before and after fine-tuning to confirm whether the new objective improves the specialist confusion pairs. Exact duplicate images still need dataset cleanup or grouped splitting if they inflate validation/test scores.

## 2026-05-06 - Colab Evaluation Run, Akbar's Tomb Root Cause, Offline Augmentation Plan

**What changed:** Ran the baseline evaluation and first training run on Colab T4 GPU. Confirmed the following:
- Base CLIP zero-shot top-1 on all 15 classes: **66.76%** (725/1086)
- Base CLIP on 13-class specialist scope: **63.90%** (632/989)
- 796 images found across 13 specialist classes for training (Qutub Minar and Shalimar Bagh correctly excluded as zero-shot-only)
- Fine-tuned specialist checkpoint (1 run, 7 folds, 5 epochs each) best val top-1: **78.1%** (Fold 1 Epoch 4)
- Training was still running at Fold 5 Epoch 3 when the notebook was shared

**Akbar's Tomb 0.00% root cause identified:** Two compounding issues:
1. **Character encoding mismatch** — the folder on disk may use a curly RIGHT SINGLE QUOTATION MARK (U+2019) in the name `Akbar's Tomb` while the Python code uses a straight apostrophe (U+0027). `collect_image_records()` does a set-membership check against `SPECIALIST_CLUSTER` and silently skips the class if the apostrophes don't match, resulting in zero training examples.
2. **Insufficient images** — only 36 images total. With 7-fold K-fold: ~5 images per val/test fold. Even without the encoding bug, accuracy is reported in steps of ~20% and any bad fold yields 0%.
**Both issues must be fixed together.**

**Key confusion pairs from the baseline run:**
- Agra Fort → Buland Darwaza: 33 images misclassified (biggest error)
- Red Fort → Buland Darwaza: 25
- Itmad-ud-Daulah → Tomb of Salim Chishti: 17
- Safdarjung Tomb ↔ Humayun's Tomb: 17/16 (symmetric confusion)

**Dataset quality:** 49 duplicate hash groups, 50 extra duplicate files, 0 cross-class duplicates.

**Fix plan (new Colab cells):** 
- Cell A: Audit folder names with repr() to detect encoding issues
- Cell B: Rename curly-apostrophe folders to straight-apostrophe; then augment all classes with < 80 images up to 80 using RandomResizedCrop + flip + rotation + ColorJitter + GaussianBlur, saving aug_XXXX.jpg copies on disk
- Cell C: Verify counts
- Cell D: Retrain with --runs 3 --epochs 10 on the balanced dataset
- Cell E/F: Re-evaluate base and fine-tuned models
- Cell G: Zip and download checkpoints + eval artifacts

**Files modified:** `CONTEXT.md`, `REPORT_WORKLOG.md`
**Known issues / TODO:** Need the fine-tuned evaluation results (eval_outputs/finetuned_specialist/metrics.json) to confirm improvement over base CLIP. Augmented images must be excluded from evaluation to avoid inflated test scores — the evaluate.py script currently reads all images in a folder, so either use --limit_per_class or separate augmented images into a subfolder.

## 2026-05-06 - Fine-Tuned Model Results (Colab Run Complete)

**What changed:** Completed training on balanced dataset (offline augmentation to 80 images/class, apostrophe fix applied). Model checkpoint extracted and placed in `clip_mughal_finetuned/`.

**Results:**
- Mean validation top-1: **88.3%** (up from 66.76% base CLIP)
- Mean test top-1: **88.4%**
- Best single fold val top-1: **92.1%** (Fold 1, Epoch 9)
- All 13 specialist classes now functional including Akbar's Tomb (86.2%, up from 0%)
- Tomb of Salim Chishti: 100% (previously 72%)
- Taj Mahal: 98.1% (previously 93.6%)
- Agra Fort: 73.6% (previously 17.7%) — still weakest class but dramatically improved

**Key training details:** 1 run, 7 folds, 10 epochs per fold, lr=2e-6, label_smoothing=0.1, augmented images in training only (val/test clean).

**Files modified:** `train.py` (augmentation leakage fix, lr bump, label smoothing)
**Known issues / TODO:** Agra Fort remains the weakest class (73.6%). Consider adding more distinctive prompts for fort ramparts vs gateways.

## 2026-05-06 - App Integration: Metadata Info Card, Dynamic OOD, Nested Contexts

**What changed:** Completed the Streamlit app to match the full spec: upload photo → predicted monument → history paragraph → visit info card → "Open in Google Maps".

**Changes to `app.py`:**
1. **Metadata info card** — `load_monument_metadata()` reads `metadata.json` and renders: history paragraph with fun fact, 8-card visit info grid (location, built by, period, style, hours, ticket Indian, ticket foreign, Google Maps button).
2. **Dynamic OOD threshold** — `predict()` now accepts `ood_threshold` parameter. Base model uses 0.15, specialist uses 0.35 (`SPECIALIST_OOD_THRESHOLD`) since the fine-tuned model outputs higher confidence values.
3. **Nested context notes** — `NESTED_CONTEXTS` dict maps sub-monuments to their parent complex. Buland Darwaza and Tomb of Salim Chishti show "Part of the Fatehpur Sikri complex"; Moti Masjid Agra shows "Located inside Agra Fort".
4. **Top scores moved to expander** — Raw score list is now inside a collapsible `st.expander("See top scores")` so the info card takes visual priority.

**Files modified:** `app.py`, `CONTEXT.md`, `REPORT_WORKLOG.md`
**Known issues / TODO:** App is feature-complete for the assignment spec. Remaining polish: test with non-monument images to verify OOD rejection at the new specialist threshold.
