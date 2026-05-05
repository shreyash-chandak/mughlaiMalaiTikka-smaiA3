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
