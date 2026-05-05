You are working on an existing CLIP-based Mughal monument classification codebase (`app.py`, `train.py`). Modify and extend it as follows.

---

## 1. Replace non-Indian monuments

Remove the following classes completely from:

* `MONUMENT_PROFILES`

* `BASELINE_PROMPTS`

* any prompt banks / metadata

* Badshahi Mosque

* Lahore Fort

Replace them with:

1. Buland Darwaza (Fatehpur Sikri, India)
2. Tomb of Salim Chishti (Fatehpur Sikri, India)

### Add profiles:

Buland Darwaza:

* family: "complex"
* material: red sandstone
* style: monumental gateway
* signatures:

  * massive arched gateway
  * grand staircase
  * high central arch
  * Mughal calligraphy and ornamentation

Tomb of Salim Chishti:

* family: "mausoleum"
* material: white marble
* style: Mughal tomb
* signatures:

  * small white marble structure
  * intricate lattice screens
  * square layout
  * delicate carvings

Update all prompt systems:

* `PROMPT_ENSEMBLES`
* `BASELINE_PROMPTS`
* any specialist prompts if needed

---

## 2. Partial Fine-Tuning Logic (IMPORTANT)

We are NOT training on all classes.

We are only fine-tuning CLIP for the visually confusing cluster:

* Taj Mahal
* Bibi Ka Maqbara
* Itmad-ud-Daulah
* Moti Masjid Agra
* Red Fort
* Agra Fort
* Fatehpur Sikri
* Jama Masjid Delhi
* Humayun's Tomb

### Modify training:

* Detect available classes dynamically from dataset instead of using all classes:
  Replace:
  `class_names = list(PROMPT_ENSEMBLES.keys())`

  With:
  `class_names = sorted(set(record.monument_name for record in records))`

* Ensure:

  * training
  * validation
  * evaluation

  only operate on these available classes.

---

## 3. Hybrid Inference Design

Keep full zero-shot CLIP for all monuments.

Add a specialist refinement step:

Pipeline:

1. Run zero-shot CLIP → get top prediction
2. If prediction ∈ cluster:

   * run fine-tuned model
   * restrict predictions to those given classes
3. Otherwise:

   * use zero-shot result

Implement this cleanly inside `app.py`.

---

## 4. Dataset Folder Structure

Create the dataset directory (do not populate images):

```
data/
├── Taj Mahal/
├── Bibi Ka Maqbara/
├── Itmad-ud-Daulah/
├── Moti Masjid Agra/
├── Red Fort/
├── Agra Fort/
├── Fatehpur Sikri/
├── Jama Masjid Delhi/
├── Humayun's Tomb/

```

Ensure:

* folder names EXACTLY match class names in code
* `train.py` reads from this structure

Optional: allow recursive loading using `os.walk`

---

## 5. Training Setup

* Keep existing partial fine-tuning strategy:

  * freeze all layers except last 3 vision encoder blocks
  * keep visual projection trainable

* Do not modify loss or optimizer unless necessary

---

## 6. Environment Requirements

All code execution must assume:

* WSL (Ubuntu)
* Python virtual environment (`venv`)

Use:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ensure paths are Linux-compatible.

---

## 7. Output Requirements

Make only necessary changes:

* update class definitions
* update prompts
* modify training class handling
* add hybrid inference logic

Do NOT rewrite the entire codebase.

Keep changes minimal, clean, and consistent with existing style.


Treat `CONTEXT.md` as the ground truth for what state the codebase is in.
Never delete old entries — only append.

Also Update `REPORT_WORKLOG.md` as per the state of the repo.