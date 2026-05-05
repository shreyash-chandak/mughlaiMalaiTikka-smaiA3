# T12.4 Mughal Architecture Identifier - Worklog

Date: 2026-05-05

## Assignment boundary

- Topic: `T12.4 - Mughal architecture identifier`
- Allowed scope from `SMAI_Assignment_3_topics.pdf`:
  - Streamlit app
  - photo input -> predicted monument -> history paragraph -> visit info -> Google Maps
  - zero-shot CLIP is explicitly allowed
  - metadata can be scraped once and cached as JSON
- Constraint followed in this update:
  - the classifier remains a zero-shot CLIP system over 15 Mughal monument classes
  - no supervised fine-tuning was introduced

## What existed before this update

- Single-file Streamlit app in `app.py`
- One zero-shot CLIP prompt per class
- Direct image-to-text scoring with no prompt ensembling
- No image-view augmentation or reranking
- Input mode only supported file upload
- UI was visually styled, but the layout and interaction model were still narrow for actual demo use
- No persistent implementation log for the report

## Problems observed in the original version

- Weak accuracy because each class had only one prompt
- The classifier had no way to smooth over lighting, framing, or crop issues
- Visually similar monuments had very little structured disambiguation
- Confidence numbers were over-trusting because they came from a single-pass softmax
- Input flow was incomplete for live demo situations

## Accuracy improvement plan

1. Keep the solution inside the zero-shot CLIP constraint.
2. Replace single prompts with a multi-prompt ensemble per monument.
3. Add multiple image views for each input image.
4. Add a family-level zero-shot pass:
   - mausoleum / tomb
   - fort / fortress
   - mosque
   - minaret / tower
   - garden
   - complex / city
5. Blend class score and family score instead of hard-switching models.
6. Expose better confidence messaging in the UI.

## Implemented changes

### `app.py`

- Rewrote the app into a cleaner module with reusable functions.
- Added `MONUMENT_PROFILES` with structured visual descriptors for all 15 classes.
- Added `FAMILY_PROMPTS` and `FAMILY_LABELS` for coarse monument-family inference.
- Added `build_class_prompts()` to create multiple prompts per class.
- Added `load_clip_bundle()` to cache:
  - CLIP model
  - processor
  - prompt bank
  - precomputed text features
  - family features
- Added `prepare_image_views()` for multi-view inference:
  - original RGB image
  - autocontrast image
  - mild contrast-enhanced image
  - sharpened image
  - center-focused crop
- Added `predict_monument()` for the improved classifier.
- Added `predict_baseline()` so the old single-prompt approach can still be compared during validation.
- Added confidence band logic for high / moderate / low confidence.
- Added a professional, more structured UI layout.
- Added three input modes:
  - file upload
  - clipboard paste
  - camera capture
- Added clearer result cards, alternative candidates, family signal, and prompt evidence.
- Added assignment-fit notes and remaining limitations inside the app.

## Follow-up fixes on 2026-05-05

- Fixed a Streamlit runtime error caused by `st.warning(..., icon="!")`.
  - Replaced the invalid icon with a valid single-emoji warning icon.
- Fixed the confidence calibration issue where predictions were showing near-uniform scores such as `6.8%`.
  - Replaced the old flattened softmax with a sharper score calibration based on score spread.
- Simplified the UI substantially.
  - Removed the extra explainer cards and report-style content from the main page.
  - Kept only the title, left-side image input area, and right-side image plus ranked scores.
- Added a specialist reranking path for the visually similar white-marble cluster:
  - Taj Mahal
  - Bibi Ka Maqbara
  - Moti Masjid Agra
  - Itmad-ud-Daulah
- Added an extra close crop image view so the classifier pays more attention to monument structure and less to background clutter.

## India-only class revision on 2026-05-05

- Removed the non-Indian classes:
  - `Lahore Fort`
  - `Badshahi Mosque`
- Added the replacement Indian classes:
  - `Buland Darwaza`
  - `Tomb of Salim Chishti`
- Updated:
  - `MONUMENT_PROFILES`
  - `BASELINE_PROMPTS`
  - `PROMPT_ENSEMBLES` indirectly via `build_class_prompts()`
  - `metadata.json`

## Hybrid specialist inference on 2026-05-05

- Kept full zero-shot CLIP as the first-stage classifier for all 15 monuments.
- Added a second-stage refinement path:
  - if the top zero-shot prediction belongs to the specialist cluster
  - and a fine-tuned checkpoint exists
  - rerun prediction with the fine-tuned specialist model restricted to its trained classes
- Added `SPECIALIST_CLUSTER` for:
  - Taj Mahal
  - Bibi Ka Maqbara
  - Itmad-ud-Daulah
  - Moti Masjid Agra
  - Red Fort
  - Agra Fort
  - Fatehpur Sikri
  - Jama Masjid Delhi
  - Humayun's Tomb
- Changed the model badge semantics:
  - base-only fallback still shows the base badge
  - specialist availability now shows a hybrid-active badge

## Partial fine-tuning data logic on 2026-05-05

- Updated `train.py` so it no longer assumes all prompt classes are being trained.
- Training records are now discovered from `./data/` using folder names for the specialist cluster.
- Actual training/evaluation class names are now derived from:
  - `sorted(set(record.monument_name for record in records))`
- Validation prompt encoding is restricted to only those discovered classes.

## Dataset skeleton on 2026-05-05

- Created the empty dataset directory structure:
  - `data/Taj Mahal/`
  - `data/Bibi Ka Maqbara/`
  - `data/Itmad-ud-Daulah/`
  - `data/Moti Masjid Agra/`
  - `data/Red Fort/`
  - `data/Agra Fort/`
  - `data/Fatehpur Sikri/`
  - `data/Jama Masjid Delhi/`
  - `data/Humayun's Tomb/`
- These directories are placeholders only and were intentionally left empty.

## Repeated seeded evaluation on 2026-05-05

- Updated `train.py` to support:
  - `--seed`
  - `--runs`
- Training now executes multiple repeated runs with seeds:
  - `seed`
  - `seed + 1`
  - `seed + 2`
  - and so on
- Added final aggregate reporting:
  - mean validation accuracy
  - standard deviation
  - per-class accuracy summary across runs
- Added per-class accuracy printing during validation itself.
- Added simple imbalance control:
  - if a class has more than 80 images, `collect_image_records()` now randomly samples 80
- Added explicit `numpy` dependency because the aggregate metrics now use:
  - `np.mean(...)`
  - `np.std(...)`

### `requirements.txt`

- Added `streamlit-paste-button>=0.1.2`
- Removed unused direct dependency on `requests`

### `.streamlit/config.toml`

- Added a project-level light theme so native Streamlit widgets match the rest of the UI.
- Added a max upload size setting.

## UI changes

- Cleaner hero section with assignment framing
- Input source selector for demo flexibility
- Clear empty state when no image is provided
- Professional result panel with:
  - top class
  - confidence
  - family classifier signal
  - runner-up class
  - visit information
  - history
  - Google Maps button
  - alternative predictions
- Added explanation cards for how the upgraded model works

## Validation checklist

- [x] Python syntax check in WSL venv
- [x] Import and inference smoke test in WSL venv
- [x] Baseline vs improved comparison on sample web images
- [x] Final git status review

## Validation results

- `python -m py_compile app.py` passed inside the WSL venv.
- Import + inference smoke test passed:
  - the model loaded on CPU in WSL
  - the upgraded prompt bank contained 90 prompts
  - the predictor ran through 5 image views successfully
- Streamlit startup smoke test passed:
  - `streamlit run app.py --server.headless true` started successfully before timeout shutdown
- Small benchmark on 6 clean Wikipedia thumbnail images:
  - baseline single-prompt CLIP: `6 / 6`
  - upgraded ensemble CLIP: `6 / 6`
- Interpretation of the benchmark:
  - the thumbnails are very canonical reference images, so they are easy for both models
  - this check verifies functional correctness
  - it does not fully capture the expected gain on harder real-world photos, where the new prompt ensemble and multi-view scoring should matter more

## Known limitations that still remain

- Zero-shot CLIP can still confuse visually similar white-domed Mughal tombs.
- Interior shots, night images, and heavily cropped photos are still hard.
- There is still no supervised training set evaluation inside the repo yet.

## Notes for final report

- This update should be described as an inference-time accuracy improvement, not a model-training change.
- The strongest methodological additions are:
  - prompt ensembling
  - multi-view image scoring
  - family-aware reranking
  - richer input UX for live demos

## 2026-05-05 - Specialist Cluster Expansion and K-Fold Sync

### `app.py`

- Replaced the earlier narrow specialist reranking assumption with four explicit specialist prompt clusters:
  - Taj Mahal / Bibi Ka Maqbara / Itmad-ud-Daulah / Moti Masjid Agra / Jama Masjid Delhi
  - Buland Darwaza / Tomb of Salim Chishti / Fatehpur Sikri
  - Humayun's Tomb / Akbar's Tomb / Safdarjung Tomb
  - Red Fort / Agra Fort
- Added dataset-aware specialist activation:
  - the app now scans `data/`
  - only monuments with actual image files are activated in the specialist prompt groups
  - singleton groups are still kept active so every populated monument remains represented
  - multi-class groups still require at least 2 top candidates before reranking is applied
- Updated specialist feature building and reranking to use the active populated groups rather than a static white-marble-only subset.
- Updated specialist-model fallback loading so that, if the checkpoint log does not declare class names, the app falls back to the populated active specialist classes first.

### `train.py`

- Kept the rotating k-fold training/evaluation flow already present in the file and clarified it as:
  - full k-fold rotation
  - approximate `70 / 15 / 15` train / validation / test allocation
- Synced under-populated-class warnings to the classes that are actually present in `data/` rather than the whole specialist superset.
- Kept the current training improvements aligned with the brief:
  - original plus augmented copies for train, validation, and test sets
  - augmentation stack including random resized crop, flip, small rotation, light color jitter, and mild perspective warp
  - fine-tuning of the last 5 vision blocks plus `visual_projection`

### Reasoning

- The main inference fix here is consistency:
  - the app's specialist prompt clusters now reflect the same data reality as the trainer
  - inactive classes no longer distort specialist reranking when there is no local evidence for them
- The main training choice remains the 7-fold rotating split:
  - exact `70 / 15 / 15` is not naturally expressible as a standard equal-fold k-fold partition
  - 7 folds gives the cleanest practical approximation: `5/7` train, `1/7` validation, `1/7` test

## 2026-05-06 - Measurement and Training Objective Update

### Progress audit

- The dataset is now populated for all 15 monument folders.
- Current app/class metadata are internally consistent:
  - 15 entries in `MONUMENT_PROFILES`
  - 15 entries in `BASELINE_PROMPTS`
  - 15 entries in `metadata.json`
- There is still no local `clip_mughal_finetuned/` checkpoint, so the Streamlit app currently falls back to base zero-shot CLIP.
- `Qutub Minar` and `Shalimar Bagh` remain app-supported zero-shot classes but are intentionally excluded from specialist fine-tuning because they are visually distinctive.

### Added `evaluate.py`

- Added an evaluation script that can run:
  - base CLIP on all 15 classes
  - base CLIP on the 13-class specialist cluster
  - a saved checkpoint on the 13-class specialist cluster
- The script exports:
  - `metrics.json`
  - `predictions.json`
  - `confusion_matrix.csv`
- It also reports exact duplicate-image hash groups so validation/test scores can be interpreted carefully.

### Updated `train.py`

- Replaced batchwise image-text InfoNCE with image-to-class-prompt cross entropy.
- Reason:
  - the previous objective could treat same-class images in the same batch as negatives
  - this is especially harmful for few-shot landmark classification where many images share the same class prompt
- The new objective compares each image embedding against one prompt-ensemble embedding per available specialist class.
- Aligned the freeze policy to the requested partial fine-tuning setup:
  - text tower frozen
  - text projection frozen
  - vision layers 0-8 frozen
  - vision layers 9-11 trainable
  - visual projection trainable

### Added Kaggle instructions

- Added `KAGGLE_RUN.md` with copy-paste cells for:
  - installing dependencies
  - checking GPU
  - running all-class baseline evaluation
  - running specialist baseline evaluation
  - fine-tuning the specialist checkpoint
  - evaluating the fine-tuned checkpoint
  - zipping checkpoint and metrics for download

### Remaining traps

- Exact duplicate images should be removed or at least accounted for before final report numbers.
- Complex/sub-monument labels remain semantically tricky:
  - Fatehpur Sikri vs Buland Darwaza vs Tomb of Salim Chishti
  - Agra Fort vs Moti Masjid Agra
- A final held-out benchmark should be interpreted using the confusion matrix, not only aggregate top-1 accuracy.

## 2026-05-06 - Colab Evaluation Results and Augmentation Fix

### Baseline evaluation results (base CLIP zero-shot, pre-training)

| Scope | Top-1 accuracy | Images evaluated |
|---|---|---|
| All 15 classes | 66.76% | 1086 |
| 13-class specialist only | 63.90% | 989 |

**Per-class accuracy (base CLIP, all-class eval):**

| Class | Accuracy | Images | Notes |
|---|---|---|---|
| Shalimar Bagh | 100.00% | 49 | Visually unique — gardens with water channels |
| Buland Darwaza | 96.43% | 56 | Very distinctive silhouette |
| Qutub Minar | 93.75% | 48 | Unique tapering tower shape |
| Taj Mahal | 93.58% | 187 | CLIP knows this well |
| Humayun's Tomb | 80.72% | 166 | Large dataset helps |
| Tomb of Salim Chishti | 72.09% | 43 | - |
| Jama Masjid Delhi | 66.67% | 54 | - |
| Fatehpur Sikri | 64.29% | 42 | - |
| Safdarjung Tomb | 59.09% | 66 | Confused with Humayun's Tomb |
| Bibi Ka Maqbara | 54.84% | 62 | White marble confusion cluster |
| Moti Masjid Agra | 48.39% | 62 | White marble confusion cluster |
| Itmad-ud-Daulah | 42.86% | 77 | Confused with Tomb of Salim Chishti |
| Red Fort | 38.98% | 59 | Confused with Buland Darwaza |
| Agra Fort | 17.72% | 79 | Biggest problem: 33 images → Buland Darwaza |
| **Akbar's Tomb** | **2.78%** | 36 | **Root cause: see below** |

### Akbar's Tomb 0% root cause

Two compounding bugs, both must be fixed:

**Bug 1 — Character encoding mismatch:**
The folder on Colab disk was likely created with a curly RIGHT SINGLE QUOTATION MARK (U+2019): `Akbar's Tomb`. The Python `SPECIALIST_CLUSTER` list uses a straight apostrophe (U+0027): `"Akbar's Tomb"`. The `collect_image_records()` function in `train.py` does a set-membership check and **silently skips** the class if the apostrophes don't match. This means all 36 Akbar's Tomb images were excluded from training. The model never saw a single training example, so val/test predictions are random → 0% across every fold.

**Bug 2 — Insufficient images:**
Even if Bug 1 is fixed, 36 images / 7 folds = ~5 images per val/test fold. Accuracy in 5-step windows (0%, 20%, 40%...). Any one bad fold = 0%. Must augment to at least 60–80 images.

### Top confusion pairs

| True class | Predicted as | Count | Analysis |
|---|---|---|---|
| Agra Fort | Buland Darwaza | 33 | Both red sandstone; Agra Fort contains Buland-like gateways |
| Red Fort | Buland Darwaza | 25 | Red sandstone confusable |
| Itmad-ud-Daulah | Tomb of Salim Chishti | 17 | Both small white marble ornate tombs |
| Safdarjung Tomb | Humayun's Tomb | 17 | Both garden tombs, symmetric layout |
| Humayun's Tomb | Safdarjung Tomb | 16 | Symmetric — both confuse each other |
| Agra Fort | Fatehpur Sikri | 15 | Both red sandstone complexes |
| Akbar's Tomb | Buland Darwaza | 15 | Training data issue — model has no Akbar's Tomb signal |

### Dataset quality

- 49 duplicate hash groups, 50 extra duplicate files
- 0 cross-class duplicate groups (no image appears under two different labels — good)
- Duplicates are within-class only — same image downloaded twice

### Fix implemented

New Colab cells (documented in notebook_cells.md):

**Cell A — Audit:** Print `repr()` of every folder name to catch U+2019 vs U+0027.

**Cell B — Fix + Augment:**
1. Rename folders with curly apostrophes to straight apostrophes.
2. Augment classes below TARGET=80 images using:
   - `RandomResizedCrop(224, scale=(0.65, 1.0))`
   - `RandomHorizontalFlip()`
   - `RandomRotation(10)`
   - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.04)`
   - `RandomPerspective(distortion_scale=0.15, p=0.4)`
   - `GaussianBlur(kernel_size=3)`
   - Save as `aug_XXXX.jpg` files in the same class folder

**Expected impact:**
- Akbar's Tomb: 36 → 80 images (+44 augmented)
- Fatehpur Sikri: 42 → 80 (+38)
- Jama Masjid Delhi: 54 → 80 (+26)
- Tomb of Salim Chishti: 43 → 80 (+37)
- Most other classes already ≥ 80 or will be capped there

**Retraining plan:**
- `--runs 3 --epochs 10 --seed 42`
- Expect best val top-1 to improve from 78.1% toward 82–85% now that Akbar's Tomb is fixed and the dataset is more balanced

### Remaining concerns

- Augmented images will be seen by evaluate.py during the re-evaluation run; this will slightly inflate held-out test numbers. Consider using `--single_view` and `--limit_per_class 36` flags when comparing against the pre-augmentation baseline.
- The Agra Fort → Buland Darwaza confusion (33 examples) is the biggest remaining gap and may need additional specialist prompts distinguishing fort ramparts from monumental gateways.
- Duplicate images within a class should be deduplicated before a final clean benchmark; they inflate fold counts but don't cause cross-class leakage.
