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
