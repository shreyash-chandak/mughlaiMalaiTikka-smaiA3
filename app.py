from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import streamlit as st
import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import CLIPModel, CLIPProcessor

try:
    from streamlit_paste_button import paste_image_button
except ModuleNotFoundError:
    paste_image_button = None

MODEL_NAME = "openai/clip-vit-base-patch32"
FINETUNED_MODEL_DIR = "platynator/clip-mughal-model"
DATA_DIR = "data"
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")
OOD_THRESHOLD = 0.15  # Random chance is about 6.7% for 15 classes; 15% is a safer boundary between noise and a plausible match.
SPECIALIST_OOD_THRESHOLD = 0.35  # Fine-tuned models are more confident; raise threshold accordingly.

# Nested context: sub-monuments that are physically inside a larger complex
NESTED_CONTEXTS: dict[str, str] = {
    "Buland Darwaza": "Part of the Fatehpur Sikri complex",
    "Tomb of Salim Chishti": "Part of the Fatehpur Sikri complex",
    "Moti Masjid Agra": "Located inside Agra Fort",
}
SPECIALIST_GROUPS: dict[str, list[str]] = {
    "marble_and_mosque_cluster": [
        "Taj Mahal",
        "Bibi Ka Maqbara",
        "Itmad-ud-Daulah",
        "Moti Masjid Agra",
        "Jama Masjid Delhi",
    ],
    "fatehpur_cluster": [
        "Buland Darwaza",
        "Tomb of Salim Chishti",
        "Fatehpur Sikri",
    ],
    "garden_tomb_cluster": [
        "Humayun's Tomb",
        "Akbar's Tomb",
        "Safdarjung Tomb",
    ],
    "fort_cluster": [
        "Red Fort",
        "Agra Fort",
    ],
}
SPECIALIST_CLUSTER = [monument for cluster in SPECIALIST_GROUPS.values() for monument in cluster]


MONUMENT_PROFILES: dict[str, dict[str, Any]] = {
    "Taj Mahal": {
        "aliases": ["Taj Mahal"],
        "family": "mausoleum",
        "place": "Agra, Uttar Pradesh, India",
        "material": "white marble",
        "style": "Mughal mausoleum",
        "signatures": [
            "a large central onion dome",
            "four tall minarets",
            "a reflecting pool",
            "a perfectly symmetrical garden axis",
        ],
    },
    "Humayun's Tomb": {
        "aliases": ["Humayun's Tomb"],
        "family": "mausoleum",
        "place": "New Delhi, India",
        "material": "red sandstone with white marble",
        "style": "garden tomb",
        "signatures": [
            "a high plinth",
            "a large white dome",
            "charbagh gardens",
            "arched red sandstone facades",
        ],
    },
    "Red Fort": {
        "aliases": ["Red Fort", "Lal Qila"],
        "family": "fort",
        "place": "Old Delhi, India",
        "material": "red sandstone",
        "style": "fortress-palace",
        "signatures": [
            "massive fortified walls",
            "battlements",
            "the Lahori Gate",
            "broad ceremonial facades",
        ],
    },
    "Agra Fort": {
        "aliases": ["Agra Fort"],
        "family": "fort",
        "place": "Agra, Uttar Pradesh, India",
        "material": "red sandstone with marble palaces",
        "style": "riverfront fort",
        "signatures": [
            "thick ramparts",
            "fortified gateways",
            "curving red walls",
            "palace courtyards inside the fort",
        ],
    },
    "Fatehpur Sikri": {
        "aliases": ["Fatehpur Sikri"],
        "family": "complex",
        "place": "Agra district, Uttar Pradesh, India",
        "material": "red sandstone",
        "style": "imperial city complex",
        "signatures": [
            "Buland Darwaza",
            "vast courtyards",
            "ornate pavilions",
            "a historic Mughal city layout",
        ],
    },
    "Itmad-ud-Daulah": {
        "aliases": ["Itmad-ud-Daulah", "Baby Taj", "Itimad-ud-Daulah"],
        "family": "mausoleum",
        "place": "Agra, Uttar Pradesh, India",
        "material": "white marble with pietra dura inlay",
        "style": "ornamental tomb",
        "signatures": [
            "delicate inlay work",
            "corner towers",
            "a jewel-box scale",
            "fine lattice screens",
        ],
    },
    "Jama Masjid Delhi": {
        "aliases": ["Jama Masjid", "Jama Masjid Delhi"],
        "family": "mosque",
        "place": "Old Delhi, India",
        "material": "red sandstone and white marble",
        "style": "congregational mosque",
        "signatures": [
            "three white domes",
            "two tall minarets",
            "a monumental staircase",
            "a large prayer courtyard",
        ],
    },
    "Qutub Minar": {
        "aliases": ["Qutub Minar", "Qutb Minar"],
        "family": "minaret",
        "place": "Mehrauli, Delhi, India",
        "material": "red sandstone and buff stone",
        "style": "tapering minaret tower",
        "signatures": [
            "a fluted tapering tower",
            "circular balconies",
            "ornamental calligraphic bands",
            "a very tall vertical silhouette",
        ],
    },
    "Bibi Ka Maqbara": {
        "aliases": ["Bibi Ka Maqbara"],
        "family": "mausoleum",
        "place": "Aurangabad, Maharashtra, India",
        "material": "white marble over a Deccan plinth",
        "style": "Mughal mausoleum",
        "signatures": [
            "a central white dome",
            "four corner minarets",
            "a long axial garden",
            "a Taj Mahal-like profile",
        ],
    },
    "Akbar's Tomb": {
        "aliases": ["Akbar's Tomb", "Akbar Tomb Sikandra"],
        "family": "mausoleum",
        "place": "Sikandra, Agra, India",
        "material": "red sandstone with marble details",
        "style": "multi-tiered tomb",
        "signatures": [
            "a grand entry gate",
            "stacked terraces",
            "white marble chhatris",
            "open pavilions at the top",
        ],
    },
    "Buland Darwaza": {
        "aliases": ["Buland Darwaza"],
        "family": "complex",
        "place": "Fatehpur Sikri, Uttar Pradesh, India",
        "material": "red sandstone",
        "style": "monumental gateway",
        "signatures": [
            "massive arched gateway",
            "grand staircase",
            "high central arch",
            "Mughal calligraphy and ornamentation",
        ],
    },
    "Tomb of Salim Chishti": {
        "aliases": ["Tomb of Salim Chishti", "Salim Chishti Tomb"],
        "family": "mausoleum",
        "place": "Fatehpur Sikri, Uttar Pradesh, India",
        "material": "white marble",
        "style": "Mughal tomb",
        "signatures": [
            "small white marble structure",
            "intricate lattice screens",
            "square layout",
            "delicate carvings",
        ],
    },
    "Shalimar Bagh": {
        "aliases": ["Shalimar Bagh", "Shalimar Garden", "Shalimar Gardens"],
        "family": "garden",
        "place": "Srinagar, Kashmir, India",
        "material": "stone channels and terraced greenery",
        "style": "Mughal garden",
        "signatures": [
            "terraced lawns",
            "water channels",
            "fountains",
            "tree-lined garden walkways",
        ],
    },
    "Moti Masjid Agra": {
        "aliases": ["Moti Masjid Agra", "Moti Masjid"],
        "family": "mosque",
        "place": "Agra Fort, Agra, India",
        "material": "white marble",
        "style": "small imperial mosque",
        "signatures": [
            "three white domes",
            "clean marble facades",
            "graceful arches",
            "a compact prayer courtyard",
        ],
    },
    "Safdarjung Tomb": {
        "aliases": ["Safdarjung Tomb", "Safdarjung's Tomb"],
        "family": "mausoleum",
        "place": "New Delhi, India",
        "material": "sandstone with marble trim",
        "style": "late Mughal tomb",
        "signatures": [
            "a bulbous central dome",
            "four corner towers",
            "charbagh gardens",
            "warm sandstone walls",
        ],
    },
}


FAMILY_PROMPTS: dict[str, list[str]] = {
    "mausoleum": [
        "a Mughal mausoleum or tomb with a dominant dome and formal garden planning",
        "an Indo-Islamic funerary monument with charbagh symmetry and ornamental arches",
    ],
    "fort": [
        "a Mughal fort or fortress with heavy walls, gateways, battlements, and palace courts",
        "a large red sandstone defensive complex from the Mughal period",
    ],
    "mosque": [
        "a Mughal mosque with domes, minarets, prayer arches, and a congregational courtyard",
        "an imperial mosque built in red sandstone or white marble",
    ],
    "minaret": [
        "a tall tapering Islamic minaret tower with balconies and carved bands",
        "a monumental victory tower from the Delhi Sultanate or Mughal visual tradition",
    ],
    "garden": [
        "a formal Mughal garden with terraces, fountains, water channels, and tree-lined paths",
        "a heritage landscape focused on water geometry and garden planning",
    ],
    "complex": [
        "a Mughal imperial city complex with gateways, courtyards, pavilions, and ceremonial spaces",
        "a monumental heritage complex built in red sandstone with multiple architectural elements",
    ],
}


FAMILY_LABELS: dict[str, str] = {
    "mausoleum": "Mausoleum / Tomb",
    "fort": "Fort / Fortress",
    "mosque": "Mosque",
    "minaret": "Minaret / Tower",
    "garden": "Garden",
    "complex": "Complex / City",
}


BASELINE_PROMPTS: dict[str, str] = {
    "Taj Mahal": "a photograph of the Taj Mahal, a white marble Mughal mausoleum with a central dome and four minarets in Agra India",
    "Humayun's Tomb": "a photograph of Humayun's Tomb, a red sandstone Mughal mausoleum with a large central dome surrounded by gardens in Delhi India",
    "Red Fort": "a photograph of the Red Fort, a massive Mughal fortress with red sandstone walls and Lahori Gate in Old Delhi India",
    "Agra Fort": "a photograph of Agra Fort, a large Mughal fort with red sandstone ramparts and palace buildings along the Yamuna river",
    "Fatehpur Sikri": "a photograph of Fatehpur Sikri, a Mughal abandoned city with the Buland Darwaza gateway and Jama Masjid in Uttar Pradesh India",
    "Itmad-ud-Daulah": "a photograph of Itmad-ud-Daulah also called Baby Taj, a small white marble Mughal tomb with corner towers and pietra dura inlay in Agra",
    "Jama Masjid Delhi": "a photograph of Jama Masjid in Delhi, a large Mughal mosque with three white marble onion domes and two red sandstone minarets",
    "Qutub Minar": "a photograph of Qutub Minar, a tall tapering brick minaret with ornate bands and balconies in Mehrauli Delhi India",
    "Bibi Ka Maqbara": "a photograph of Bibi Ka Maqbara also known as Taj of the Deccan, a white marble Mughal mausoleum in Aurangabad Maharashtra resembling the Taj Mahal",
    "Akbar's Tomb": "a photograph of Akbar's Tomb in Sikandra, a multi-story Mughal mausoleum with red sandstone terraces and white marble pavilions near Agra",
    "Buland Darwaza": "a photograph of Buland Darwaza, a monumental red sandstone Mughal gateway at Fatehpur Sikri with a grand staircase and a high central arch",
    "Tomb of Salim Chishti": "a photograph of the Tomb of Salim Chishti, a small white marble Mughal tomb at Fatehpur Sikri with intricate lattice screens and delicate carvings",
    "Shalimar Bagh": "a photograph of Shalimar Bagh in Srinagar, a terraced Mughal garden with fountains, water channels and chinar trees beside Dal Lake in Kashmir",
    "Moti Masjid Agra": "a photograph of Moti Masjid inside Agra Fort, a small pure white marble Mughal mosque with three marble domes and graceful arches",
    "Safdarjung Tomb": "a photograph of Safdarjung Tomb, a late Mughal sandstone mausoleum with a central dome and four corner towers surrounded by gardens in New Delhi",
}


SPECIALIST_PROMPTS: dict[str, dict[str, list[str]]] = {
    "marble_and_mosque_cluster": {
        "Taj Mahal": [
            "a huge white marble mausoleum with one dominant central dome, four free-standing minarets, and a long reflecting pool",
            "the Taj Mahal, a very large symmetrical Mughal tomb with four corner minarets around the main building",
        ],
        "Bibi Ka Maqbara": [
            "a smaller Taj-like white mausoleum with four minarets and a narrower profile",
            "Bibi Ka Maqbara, a white-domed Mughal tomb that resembles the Taj Mahal but is smaller and less monumental",
        ],
        "Moti Masjid Agra": [
            "a compact white marble mosque with three domes, repeated prayer arches, and no huge free-standing minarets",
            "the Moti Masjid in Agra Fort, a small marble mosque with three bulbous domes and a prayer courtyard",
        ],
        "Itmad-ud-Daulah": [
            "a small jewel-box marble tomb with corner towers, delicate inlay, and fine lattice work",
            "Itmad-ud-Daulah, a compact white marble mausoleum with slender corner turrets and ornate surface details",
        ],
        "Jama Masjid Delhi": [
            "a grand congregational mosque with three marble domes, tall minarets, and a wide courtyard",
            "Jama Masjid Delhi, a monumental red sandstone and white marble mosque reached by broad stairways",
        ],
    },
    "fatehpur_cluster": {
        "Buland Darwaza": [
            "a colossal red sandstone gateway with a grand staircase and very high central arch",
            "Buland Darwaza, a monumental Mughal entrance facade with calligraphy and towering gateway proportions",
        ],
        "Tomb of Salim Chishti": [
            "a small white marble tomb with intricate jali screens and delicate carved surfaces",
            "the Tomb of Salim Chishti, a compact square shrine in white marble set within the Fatehpur Sikri complex",
        ],
        "Fatehpur Sikri": [
            "a wide Mughal imperial complex with courtyards, pavilions, and multiple red sandstone structures",
            "Fatehpur Sikri, a large historic city complex rather than a single gateway or shrine",
        ],
    },
    "garden_tomb_cluster": {
        "Humayun's Tomb": [
            "a grand red sandstone garden tomb with a high plinth and large white double dome",
            "Humayun's Tomb, a monumental charbagh mausoleum with broad symmetrical facades",
        ],
        "Akbar's Tomb": [
            "a multi-tiered Mughal tomb with terraces, chhatris, and a less dominant central dome",
            "Akbar's Tomb, a sprawling sandstone mausoleum complex with layered pavilion-like upper levels",
        ],
        "Safdarjung Tomb": [
            "a late Mughal garden tomb with a bulbous dome, four corner towers, and warm sandstone walls",
            "Safdarjung Tomb, a symmetrical walled garden mausoleum with a central domed block and corner pavilions",
        ],
    },
    "fort_cluster": {
        "Red Fort": [
            "a very large ceremonial Mughal fort with massive red sandstone walls and the Lahori Gate",
            "Red Fort, a fortress-palace complex with long straight defensive walls and imperial gates",
        ],
        "Agra Fort": [
            "a Mughal fort with curved red sandstone ramparts, layered gateways, and palace courtyards inside",
            "Agra Fort, a sprawling fortified complex with heavy walls and internal marble palaces",
        ],
    }
}


def discover_populated_data_classes(data_dir: str = DATA_DIR) -> set[str]:
    """Return class-folder names in ``data/`` that contain at least one image file.

    Args:
        data_dir: Dataset root used by the training pipeline.

    Returns:
        set[str]: Populated class names discovered on disk.
    """

    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    populated_classes: set[str] = set()

    if not os.path.isdir(data_dir):
        return populated_classes

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if any(
            os.path.isfile(os.path.join(class_dir, file_name))
            and os.path.splitext(file_name)[1].lower() in valid_suffixes
            for file_name in os.listdir(class_dir)
        ):
            populated_classes.add(class_name)

    return populated_classes


def get_active_specialist_prompts(data_dir: str = DATA_DIR) -> dict[str, dict[str, list[str]]]:
    """Filter specialist clusters to the monuments that currently have data on disk.

    If no populated dataset folders are found, the full prompt map is returned so the
    app still works in pure inference/demo mode.

    Args:
        data_dir: Dataset root used to discover populated classes.

    Returns:
        dict[str, dict[str, list[str]]]: Active specialist prompt groups.
    """

    populated_classes = discover_populated_data_classes(data_dir)
    if not populated_classes:
        return SPECIALIST_PROMPTS

    active_prompts: dict[str, dict[str, list[str]]] = {}
    for group_name, group_prompts in SPECIALIST_PROMPTS.items():
        filtered_group = {
            monument_name: prompts
            for monument_name, prompts in group_prompts.items()
            if monument_name in populated_classes
        }
        if filtered_group:
            active_prompts[group_name] = filtered_group

    return active_prompts


ACTIVE_SPECIALIST_PROMPTS = get_active_specialist_prompts()
ACTIVE_SPECIALIST_CLUSTER = [
    monument_name
    for group_prompts in ACTIVE_SPECIALIST_PROMPTS.values()
    for monument_name in group_prompts
]


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #f7f2e8;
    --surface: #fffaf2;
    --surface-2: #efe4d2;
    --ink: #1f1a14;
    --muted: #6d6258;
    --accent: #9b6b2f;
    --accent-2: #c99f62;
    --accent-3: #7e2330;
    --line: #deceb6;
    --success: #355f46;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(circle at top right, rgba(201, 159, 98, 0.20), transparent 28%),
        linear-gradient(180deg, #fbf7ef 0%, #f6efe2 100%);
    color: var(--ink);
}

#MainMenu, footer, header {
    visibility: hidden;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

h1, h2, h3, h4, .serif {
    font-family: 'Cormorant Garamond', serif;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1220px;
}

.hero-shell {
    background: linear-gradient(135deg, rgba(20, 17, 12, 0.98), rgba(72, 38, 21, 0.95) 55%, rgba(126, 35, 48, 0.92));
    border: 1px solid rgba(233, 206, 167, 0.18);
    color: #f7ead1;
    border-radius: 24px;
    padding: 2.4rem 2.3rem;
    overflow: hidden;
    position: relative;
    box-shadow: 0 18px 42px rgba(36, 27, 19, 0.18);
    margin-bottom: 1.3rem;
}

.hero-shell::after {
    content: "MUGHAL";
    position: absolute;
    top: 1rem;
    right: 1.25rem;
    font-size: 0.8rem;
    letter-spacing: 0.44em;
    color: rgba(247, 234, 209, 0.22);
}

.hero-title {
    font-size: clamp(2.5rem, 4vw, 4.3rem);
    line-height: 0.95;
    margin: 0;
    font-weight: 600;
}

.hero-kicker {
    letter-spacing: 0.28em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: rgba(247, 234, 209, 0.72);
    margin-bottom: 0.8rem;
}

.hero-copy {
    max-width: 720px;
    line-height: 1.65;
    color: rgba(247, 234, 209, 0.86);
    margin-top: 1rem;
    font-size: 0.98rem;
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 1.4rem;
}

.hero-chip {
    background: rgba(255, 250, 242, 0.08);
    border: 1px solid rgba(247, 234, 209, 0.12);
    border-radius: 16px;
    padding: 0.9rem 1rem;
}

.hero-chip-label {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(247, 234, 209, 0.66);
    margin-bottom: 0.3rem;
}

.hero-chip-value {
    font-size: 0.98rem;
    color: #fff7e8;
}

.panel {
    background: rgba(255, 250, 242, 0.90);
    border: 1px solid rgba(183, 155, 122, 0.24);
    border-radius: 22px;
    padding: 1.2rem;
    box-shadow: 0 10px 28px rgba(43, 30, 20, 0.07);
}

.panel-tight {
    padding: 1rem;
}

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.85rem;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.25rem 0 1rem;
}

.model-badge-active {
    color: var(--success);
    background: rgba(53, 95, 70, 0.10);
    border: 1px solid rgba(53, 95, 70, 0.18);
}

.model-badge-base {
    color: var(--muted);
    background: rgba(109, 98, 88, 0.10);
    border: 1px solid rgba(109, 98, 88, 0.18);
}

.result-title {
    margin: 0;
    font-size: 2.5rem;
    line-height: 1;
    color: var(--accent-3);
}

.result-subtitle {
    color: var(--muted);
    margin-top: 0.35rem;
    font-size: 0.94rem;
}

.signal-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1rem;
}

.signal-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.9rem 1rem;
}

.signal-label {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.66rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

.signal-value {
    font-size: 1.1rem;
    color: var(--ink);
}

.soft-card {
    background: white;
    border: 1px solid rgba(183, 155, 122, 0.22);
    border-radius: 18px;
    padding: 1rem;
}

.mini-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    background: rgba(155, 107, 47, 0.10);
    color: var(--accent);
    padding: 0.35rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.25rem 0.35rem 0 0;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 0.8rem;
}

.detail-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.95rem 1rem;
}

.detail-label {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.64rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

.detail-value {
    font-size: 0.95rem;
    color: var(--ink);
    line-height: 1.55;
}

.candidate-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    margin-top: 0.7rem;
}

.candidate-name {
    font-weight: 600;
    color: var(--ink);
}

.candidate-meta {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.15rem;
}

.candidate-score {
    color: var(--accent-3);
    font-weight: 700;
    white-space: nowrap;
}

.empty-state {
    min-height: 420px;
    border: 1.5px dashed rgba(155, 107, 47, 0.35);
    border-radius: 22px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(255, 250, 242, 0.72);
    text-align: center;
    padding: 2rem;
}

.empty-state-title {
    font-size: 2rem;
    margin-top: 0.6rem;
    color: var(--accent-3);
}

.empty-state-copy {
    max-width: 380px;
    line-height: 1.6;
    color: var(--muted);
}

.method-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
}

.method-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem;
}

.method-title {
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.35rem;
}

.method-copy {
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.88rem;
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    border: 1.4px dashed rgba(155, 107, 47, 0.35);
    padding: 0.65rem;
}

[data-testid="stCameraInput"] {
    background: white;
    border-radius: 18px;
    border: 1px solid var(--line);
    padding: 0.4rem;
}

[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.2rem 0.8rem 0.8rem;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

@media (max-width: 900px) {
    .hero-grid,
    .signal-strip,
    .method-row,
    .detail-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


@dataclass
class PromptBank:
    """Cache precomputed text features and prompt metadata for inference."""

    class_names: list[str]
    prompts_by_class: dict[str, list[str]]
    flat_prompts: list[str]
    class_prompt_indices: dict[str, list[int]]
    text_features: torch.Tensor
    family_names: list[str]
    family_features: torch.Tensor
    family_lookup: dict[str, str]
    specialist_features: dict[str, dict[str, torch.Tensor]]


@dataclass
class ModelStatus:
    """Describe whether inference is using the base or fine-tuned CLIP checkpoint."""

    is_finetuned: bool
    badge_text: str
    badge_class: str
    best_val_top1: float | None = None


@dataclass
class SpecialistBundle:
    """Hold an optional fine-tuned specialist model and cluster-specific prompt banks."""

    model: CLIPModel | None
    processor: CLIPProcessor | None
    banks_by_group: dict[str, PromptBank]
    class_to_group: dict[str, str]


def load_metadata() -> dict[str, Any]:
    """Load cached monument metadata from disk.

    Returns:
        dict[str, Any]: Metadata keyed by monument name.
    """

    meta_path = os.path.join(os.path.dirname(__file__), "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_class_prompts(metadata: dict[str, Any]) -> dict[str, list[str]]:
    """Build the prompt ensemble used by both inference and fine-tuning validation.

    Args:
        metadata: Monument metadata loaded from ``metadata.json``.

    Returns:
        dict[str, list[str]]: Prompt ensemble keyed by monument name.
    """

    prompts_by_class: dict[str, list[str]] = {}

    for monument, profile in MONUMENT_PROFILES.items():
        monument_meta = metadata.get(monument, {})
        place = profile["place"]
        style = monument_meta.get("style", profile["style"])
        aliases = profile["aliases"]
        material = profile["material"]
        signatures = profile["signatures"]
        family_label = FAMILY_LABELS[profile["family"]].lower()
        alias_phrase = ", also known as ".join(aliases[:2]) if len(aliases) > 1 else aliases[0]

        prompt_candidates = [
            f"a travel photograph of {aliases[0]}, a Mughal {family_label} in {place}",
            f"an architecture photo of {alias_phrase} with {signatures[0]}, {signatures[1]}, and {material}",
            f"a heritage site photograph of {aliases[0]} in {place}, known for {signatures[2]} and {signatures[3]}",
            f"a wide outdoor view of {aliases[0]}, a {style} built in {material}",
            f"a tourist photo of {aliases[0]} showing {signatures[0]} and {signatures[2]}",
            f"a daylight photograph of {aliases[0]} with distinctive Mughal details such as {signatures[1]} and {signatures[3]}",
        ]

        deduped: list[str] = []
        seen: set[str] = set()
        for prompt in prompt_candidates:
            clean_prompt = " ".join(prompt.split())
            if clean_prompt not in seen:
                deduped.append(clean_prompt)
                seen.add(clean_prompt)
        prompts_by_class[monument] = deduped

    return prompts_by_class


PROMPT_ENSEMBLES = build_class_prompts(load_metadata())


def build_family_features(model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """Encode coarse monument-family prompts for lightweight reranking.

    Args:
        model: Loaded CLIP model.
        processor: Matching CLIP processor.
        device: Torch device for feature extraction.

    Returns:
        tuple[list[str], torch.Tensor]: Family names and normalized text features.
    """

    family_names = list(FAMILY_PROMPTS.keys())
    family_prompt_texts = [" ".join(prompts) for prompts in FAMILY_PROMPTS.values()]
    encoded = processor(text=family_prompt_texts, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        features = extract_text_features(model, encoded)
    return family_names, normalize(features)


def build_specialist_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode specialist prompts for visually confusing monument clusters.

    Args:
        model: Loaded CLIP model.
        processor: Matching CLIP processor.
        device: Torch device for feature extraction.

    Returns:
        dict[str, dict[str, torch.Tensor]]: Specialist prompt features keyed by group and class.
    """

    specialist_features: dict[str, dict[str, torch.Tensor]] = {}

    for group_name, group_prompts in ACTIVE_SPECIALIST_PROMPTS.items():
        specialist_features[group_name] = {}
        for monument_name, prompts in group_prompts.items():
            encoded = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                features = extract_text_features(model, encoded)
            specialist_features[group_name][monument_name] = normalize(features)

    return specialist_features


def build_prompt_bank(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    prompts_by_class: dict[str, list[str]],
) -> PromptBank:
    """Build a prompt bank for a specific class subset.

    Args:
        model: Loaded CLIP model.
        processor: Matching CLIP processor.
        device: Torch device used for feature extraction.
        prompts_by_class: Prompt ensemble keyed by class name.

    Returns:
        PromptBank: Cached prompt features and lookup metadata.
    """

    class_names = list(prompts_by_class.keys())
    flat_prompts: list[str] = []
    class_prompt_indices: dict[str, list[int]] = {}

    for class_name in class_names:
        indices: list[int] = []
        for prompt in prompts_by_class[class_name]:
            indices.append(len(flat_prompts))
            flat_prompts.append(prompt)
        class_prompt_indices[class_name] = indices

    encoded = processor(text=flat_prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        text_features = extract_text_features(model, encoded)
    text_features = normalize(text_features)

    family_names, family_features = build_family_features(model, processor, device)
    specialist_features = build_specialist_features(model, processor, device)
    return PromptBank(
        class_names=class_names,
        prompts_by_class=prompts_by_class,
        flat_prompts=flat_prompts,
        class_prompt_indices=class_prompt_indices,
        text_features=text_features,
        family_names=family_names,
        family_features=family_features,
        family_lookup={name: MONUMENT_PROFILES[name]["family"] for name in class_names},
        specialist_features=specialist_features,
    )


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature vectors along the last dimension.

    Args:
        tensor: Feature tensor to normalize.

    Returns:
        torch.Tensor: Normalized feature tensor.
    """

    return tensor / tensor.norm(dim=-1, keepdim=True)


def get_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU.

    Returns:
        torch.device: Runtime device for inference or training helpers.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_text_features(model: CLIPModel, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Project CLIP text encoder outputs into the shared embedding space.

    Args:
        model: Loaded CLIP model.
        encoded_inputs: Tokenized text inputs from the processor.

    Returns:
        torch.Tensor: Text embeddings in CLIP's joint feature space.
    """

    text_outputs = model.text_model(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs.get("attention_mask"),
        position_ids=encoded_inputs.get("position_ids"),
        return_dict=True,
    )
    pooled_output = text_outputs.pooler_output
    return model.text_projection(pooled_output)


def extract_image_features(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Project CLIP vision encoder outputs into the shared embedding space.

    Args:
        model: Loaded CLIP model.
        pixel_values: Batched image tensor ready for the vision tower.

    Returns:
        torch.Tensor: Image embeddings in CLIP's joint feature space.
    """

    vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
    pooled_output = vision_outputs.pooler_output
    return model.visual_projection(pooled_output)


def calibrate_probabilities(class_scores: torch.Tensor) -> torch.Tensor:
    """Convert raw class scores into a sharper calibrated probability distribution.

    Args:
        class_scores: Raw class compatibility scores.

    Returns:
        torch.Tensor: Softmax probabilities after scale calibration.
    """

    centered_scores = class_scores - class_scores.mean()
    score_scale = torch.clamp(class_scores.std(), min=0.015)
    calibrated_logits = centered_scores / score_scale * 1.8
    return torch.softmax(calibrated_logits, dim=0)


def specialist_adjustment(
    image_features: torch.Tensor,
    candidate_scores: dict[str, float],
    bank: PromptBank,
) -> dict[str, float]:
    """Apply specialist reranking for monuments that are visually easy to confuse.

    Args:
        image_features: Normalized image embeddings for the current input views.
        candidate_scores: Base class scores before specialist reranking.
        bank: Prompt bank containing specialist prompt features.

    Returns:
        dict[str, float]: Adjusted class scores.
    """

    adjusted_scores = candidate_scores.copy()
    top_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    top_names = {name for name, _ in top_candidates}

    for group_name, group_features in bank.specialist_features.items():
        group_names = set(group_features.keys())
        minimum_overlap = 1 if len(group_names) == 1 else 2
        if len(top_names & group_names) < minimum_overlap:
            continue

        group_bonus: dict[str, float] = {}

        for monument_name, features in group_features.items():
            specialist_scores = image_features @ features.T
            mean_score = specialist_scores.mean().item()
            max_score = specialist_scores.max().item()
            group_bonus[monument_name] = 0.65 * mean_score + 0.35 * max_score

        for monument_name, bonus in group_bonus.items():
            if monument_name in adjusted_scores:
                adjusted_scores[monument_name] += 0.45 * bonus

    return adjusted_scores


def load_training_log(model_dir: str) -> dict[str, Any]:
    """Read training metadata for a fine-tuned checkpoint when available.

    Args:
        model_dir: Directory that may contain ``training_log.json``.

    Returns:
        dict[str, Any]: Parsed training log, or an empty dict if unavailable.
    """

    log_path = os.path.join(model_dir, "training_log.json")
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_cluster_name(monument_name: str) -> str | None:
    """Return the specialist cluster name for a monument, if any.

    Args:
        monument_name: Predicted or candidate monument class name.

    Returns:
        str | None: Cluster key from ``SPECIALIST_GROUPS`` when applicable.
    """

    for group_name, class_names in SPECIALIST_GROUPS.items():
        if monument_name in class_names:
            return group_name
    return None


def resolve_model_status(model_dir: str) -> ModelStatus:
    try:
        from transformers import CLIPModel
        CLIPModel.from_pretrained(model_dir)
        return ModelStatus(
            is_finetuned=True,
            badge_text="✦ Hybrid model active",
            badge_class="model-badge-active",
        )
    except Exception:
        return ModelStatus(
            is_finetuned=False,
            badge_text="◦ Base CLIP model (zero-shot)",
            badge_class="model-badge-base",
        )


@st.cache_resource(show_spinner=False)
def load_clip() -> tuple[CLIPModel, CLIPProcessor, PromptBank, SpecialistBundle, torch.device, ModelStatus]:
    """Load the base zero-shot CLIP model and an optional local fine-tuned specialist model.

    Returns:
        tuple[CLIPModel, CLIPProcessor, PromptBank, SpecialistBundle, torch.device, ModelStatus]:
            Base model assets, optional specialist assets, runtime device, and badge metadata.
    """

    device = get_device()
    model_status = resolve_model_status(FINETUNED_MODEL_DIR)

    base_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    base_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    base_model.eval()
    base_bank = build_prompt_bank(base_model, base_processor, device, PROMPT_ENSEMBLES)

    specialist_bundle = SpecialistBundle(model=None, processor=None, banks_by_group={}, class_to_group={})
    if model_status.is_finetuned:
        specialist_log = load_training_log(FINETUNED_MODEL_DIR)
        specialist_class_names = specialist_log.get("class_names") or specialist_log.get("expected_class_names") or []
        available_specialist_classes = set(specialist_class_names) if specialist_class_names else set(PROMPT_ENSEMBLES)

        specialist_banks: dict[str, PromptBank] = {}
        class_to_group: dict[str, str] = {}

        specialist_model = CLIPModel.from_pretrained(FINETUNED_MODEL_DIR).to(device)
        specialist_processor = CLIPProcessor.from_pretrained(FINETUNED_MODEL_DIR)
        specialist_model.eval()

        for group_name, group_prompts in ACTIVE_SPECIALIST_PROMPTS.items():
            filtered_prompts = {
                class_name: prompts
                for class_name, prompts in group_prompts.items()
                if class_name in available_specialist_classes
            }
            if not filtered_prompts:
                continue

            specialist_banks[group_name] = build_prompt_bank(
                specialist_model,
                specialist_processor,
                device,
                filtered_prompts,
            )
            for class_name in filtered_prompts:
                class_to_group[class_name] = group_name

        if specialist_banks:
            specialist_model.eval()
            specialist_bundle = SpecialistBundle(
                model=specialist_model,
                processor=specialist_processor,
                banks_by_group=specialist_banks,
                class_to_group=class_to_group,
            )

    return base_model, base_processor, base_bank, specialist_bundle, device, model_status


def center_crop(image: Image.Image, crop_ratio: float) -> Image.Image:
    """Crop the center region of an image using a relative crop ratio.

    Args:
        image: Input PIL image.
        crop_ratio: Fraction of width and height to retain.

    Returns:
        Image.Image: Center-cropped image.
    """

    width, height = image.size
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    left = max((width - crop_w) // 2, 0)
    top = max((height - crop_h) // 2, 0)
    return image.crop((left, top, left + crop_w, top + crop_h))


def prepare_image_views(image: Image.Image) -> list[Image.Image]:
    """Create multiple image views to make zero-shot inference less brittle.

    Args:
        image: Input monument image.

    Returns:
        list[Image.Image]: Augmented image views used during inference.
    """

    rgb = image.convert("RGB")
    autocontrast = ImageOps.autocontrast(rgb)
    contrast = ImageEnhance.Contrast(autocontrast).enhance(1.08)
    sharpened = ImageEnhance.Sharpness(autocontrast).enhance(1.25)
    focused = center_crop(rgb, 0.88)
    focused = ImageOps.autocontrast(focused)
    close_crop = center_crop(rgb, 0.72)
    close_crop = ImageEnhance.Sharpness(ImageOps.autocontrast(close_crop)).enhance(1.2)
    return [rgb, autocontrast, contrast, sharpened, focused, close_crop]


def score_prompt_groups(
    prompt_scores: torch.Tensor,
    bank: PromptBank,
    family_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, str]]:
    """Aggregate prompt-level scores into class-level scores.

    Args:
        prompt_scores: Similarity matrix between image views and prompt embeddings.
        bank: Prompt bank containing prompt-to-class mappings.
        family_probs: Coarse family probabilities used for gentle reranking.

    Returns:
        tuple[torch.Tensor, dict[str, str]]: Class scores and the best prompt per class.
    """

    class_scores: list[torch.Tensor] = []
    best_prompts: dict[str, str] = {}

    for class_name in bank.class_names:
        indices = bank.class_prompt_indices[class_name]
        group_scores = prompt_scores[:, indices]
        mean_score = group_scores.mean()
        max_position = group_scores.argmax().item()
        max_score = group_scores.reshape(-1)[max_position]
        family_name = bank.family_lookup[class_name]
        family_bonus = 0.8 * family_probs[bank.family_names.index(family_name)]
        class_score = 0.72 * mean_score + 0.28 * max_score + family_bonus
        class_scores.append(class_score)

        best_prompt_flat_index = indices[max_position % len(indices)]
        best_prompts[class_name] = bank.flat_prompts[best_prompt_flat_index]

    return torch.stack(class_scores), best_prompts


@torch.no_grad()
def predict(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    bank: PromptBank,
    device: torch.device,
    ood_threshold: float = OOD_THRESHOLD,
    use_specialist_adjustment: bool = False,
) -> dict[str, Any]:
    """Predict the most likely monument class and flag out-of-domain inputs.

    Args:
        image: Input monument image.
        model: Active CLIP model.
        processor: Matching CLIP processor.
        bank: Prompt bank containing cached text features.
        device: Torch device used for inference.
        ood_threshold: Minimum confidence required to be considered in-domain.
        use_specialist_adjustment: Whether cluster specialist reranking should be applied.

    Returns:
        dict[str, Any]: Ranked results, confidence metadata, and an OOD flag.
    """

    views = prepare_image_views(image)
    encoded = processor(images=views, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)

    image_features = extract_image_features(model, pixel_values)
    image_features = normalize(image_features)
    prompt_scores = image_features @ bank.text_features.T
    family_scores = image_features @ bank.family_features.T
    family_probs = torch.softmax(family_scores.mean(dim=0) / 0.55, dim=0)

    class_scores, best_prompts = score_prompt_groups(prompt_scores, bank, family_probs)
    candidate_scores = {
        bank.class_names[idx]: float(class_scores[idx].item())
        for idx in range(len(bank.class_names))
    }
    if use_specialist_adjustment:
        candidate_scores = specialist_adjustment(image_features, candidate_scores, bank)
    adjusted_class_scores = torch.tensor(
        [candidate_scores[name] for name in bank.class_names],
        device=device,
        dtype=class_scores.dtype,
    )
    probs = calibrate_probabilities(adjusted_class_scores)
    ranked_indices = torch.argsort(adjusted_class_scores, descending=True)

    results: list[dict[str, Any]] = []
    for idx in ranked_indices.tolist():
        name = bank.class_names[idx]
        results.append(
            {
                "name": name,
                "probability": float(probs[idx].item()),
                "score": float(adjusted_class_scores[idx].item()),
                "family": bank.family_lookup[name],
                "best_prompt": best_prompts[name],
            }
        )

    top_margin = results[0]["score"] - results[1]["score"] if len(results) > 1 else results[0]["score"]
    family_idx = torch.argmax(family_probs).item()
    top_prob = results[0]["probability"]
    is_ood = top_prob < ood_threshold
    return {
        "results": results,
        "family_prediction": bank.family_names[family_idx],
        "family_confidence": float(family_probs[family_idx].item()),
        "top_margin": float(top_margin),
        "view_count": len(views),
        "prompt_count": len(bank.flat_prompts),
        "is_ood": is_ood,
    }


def refine_with_specialist(
    image: Image.Image,
    zero_shot_prediction: dict[str, Any],
    specialist_bundle: SpecialistBundle,
    device: torch.device,
) -> dict[str, Any]:
    """Refine zero-shot predictions only when the top class belongs to a specialist cluster.

    Args:
        image: Input monument image.
        zero_shot_prediction: Base zero-shot prediction payload.
        specialist_bundle: Optional specialist model assets.
        device: Torch device used for inference.

    Returns:
        dict[str, Any]: Final prediction payload after optional specialist refinement.
    """

    if zero_shot_prediction["is_ood"]:
        return zero_shot_prediction

    if specialist_bundle.model is None or specialist_bundle.processor is None or not specialist_bundle.banks_by_group:
        return zero_shot_prediction

    top_name = zero_shot_prediction["results"][0]["name"]
    cluster_name = specialist_bundle.class_to_group.get(top_name)
    if cluster_name is None:
        return zero_shot_prediction

    specialist_bank = specialist_bundle.banks_by_group.get(cluster_name)
    if specialist_bank is None:
        return zero_shot_prediction

    specialist_prediction = predict(
        image=image,
        model=specialist_bundle.model,
        processor=specialist_bundle.processor,
        bank=specialist_bank,
        device=device,
        ood_threshold=SPECIALIST_OOD_THRESHOLD,
        use_specialist_adjustment=False,
    )

    if specialist_prediction["is_ood"]:
        return zero_shot_prediction

    cluster_class_names = set(specialist_bank.class_names)
    specialist_prediction["results"].extend(
        result
        for result in zero_shot_prediction["results"]
        if result["name"] not in cluster_class_names
    )
    specialist_prediction["refined_by_specialist"] = True
    specialist_prediction["specialist_cluster"] = cluster_name
    return specialist_prediction


@torch.no_grad()
def predict_baseline(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> list[tuple[str, float]]:
    """Run the original single-prompt baseline for quick sanity checks.

    Args:
        image: Input monument image.
        model: Active CLIP model.
        processor: Matching CLIP processor.
        device: Torch device used for inference.

    Returns:
        list[tuple[str, float]]: Ranked baseline predictions.
    """

    texts = [BASELINE_PROMPTS[name] for name in MONUMENT_PROFILES]
    encoded = processor(text=texts, images=image.convert("RGB"), return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    probs = outputs.logits_per_image[0].softmax(dim=-1).detach().cpu().tolist()
    return sorted(zip(MONUMENT_PROFILES.keys(), probs), key=lambda item: item[1], reverse=True)


def confidence_band(probability: float, margin: float) -> str:
    """Map numeric confidence signals into a human-readable confidence band.

    Args:
        probability: Top predicted probability.
        margin: Score gap between the top two predictions.

    Returns:
        str: ``High``, ``Moderate``, or ``Low``.
    """

    if probability >= 0.60 and margin >= 0.20:
        return "High"
    if probability >= 0.35 and margin >= 0.08:
        return "Moderate"
    return "Low"


def image_from_upload(uploaded_file: Any) -> Image.Image | None:
    """Convert an uploaded file into an RGB PIL image.

    Args:
        uploaded_file: Streamlit upload object.

    Returns:
        Image.Image | None: Decoded image, or ``None`` if no file was uploaded.
    """

    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def image_from_camera(camera_file: Any) -> Image.Image | None:
    """Convert a camera capture into an RGB PIL image.

    Args:
        camera_file: Streamlit camera capture object.

    Returns:
        Image.Image | None: Decoded image, or ``None`` if no capture exists.
    """

    if camera_file is None:
        return None
    return Image.open(camera_file).convert("RGB")


def image_from_clipboard() -> Image.Image | None:
    """Read an image pasted via the clipboard widget.

    Returns:
        Image.Image | None: Clipboard image, or ``None`` if nothing has been pasted.
    """

    if paste_image_button is None:
        st.info("Clipboard paste support is unavailable until streamlit-paste-button is installed.")
        return st.session_state.get("pasted_image")

    result = paste_image_button(
        label="Paste screenshot from clipboard",
        text_color="#fffaf2",
        background_color="#9b6b2f",
        hover_background_color="#835621",
        key="paste_button",
    )
    if result.image_data is not None:
        st.session_state["pasted_image"] = result.image_data.convert("RGB")
    return st.session_state.get("pasted_image")


def render_model_status_badge(model_status: ModelStatus) -> None:
    """Render the active-model badge just below the page title.

    Args:
        model_status: Badge metadata describing the active checkpoint.
    """

    st.markdown(
        f'<div class="model-badge {model_status.badge_class}">{model_status.badge_text}</div>',
        unsafe_allow_html=True,
    )


def render_monument_chips() -> str:
    """Build the supported-monuments chip list as reusable HTML.

    Returns:
        str: Chip markup for all supported monument classes.
    """

    return "".join(f'<span class="mini-tag">{name}</span>' for name in PROMPT_ENSEMBLES)


def render_ood_panel(prediction: dict[str, Any]) -> None:
    """Render the out-of-domain message when no monument class is convincing.

    Args:
        prediction: Prediction payload containing ranked low-confidence scores.
    """

    st.markdown(
        f"""
        <div class="panel" style="text-align:center;">
            <div style="font-size:3.4rem; line-height:1;">🕌</div>
            <h2 class="result-title serif" style="text-align:center; margin-top:0.6rem;">Not a Mughal Monument</h2>
            <div class="result-subtitle" style="max-width:620px; margin:0.8rem auto 0;">
                This image does not appear to match any of the 15 Mughal monuments in our database.
                For best results, upload a clear exterior or interior photograph of one of the supported monuments.
            </div>
            <div style="margin-top:1rem;">{render_monument_chips()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See raw scores (all low)"):
        for candidate in prediction["results"]:
            st.markdown(
                f"""
                <div class="candidate-row">
                    <div class="candidate-name">{candidate["name"]}</div>
                    <div class="candidate-score">{candidate["probability"] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def load_monument_metadata() -> dict[str, Any]:
    """Load monument metadata from metadata.json.

    Returns:
        dict[str, Any]: Metadata keyed by monument name, or empty dict if file is missing.
    """
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_result_panel(image: Image.Image, prediction: dict[str, Any]) -> None:
    """Render the standard prediction panel for in-domain monument images.

    Args:
        image: Input image displayed back to the user.
        prediction: Prediction payload from ``predict``.
    """

    top_result = prediction["results"][0]
    monument_name = top_result["name"]
    conf_band = confidence_band(top_result["probability"], prediction["top_margin"])
    runner_up = prediction["results"][1] if len(prediction["results"]) > 1 else top_result

    # Nested context sub-note for sub-monuments
    nested_note = NESTED_CONTEXTS.get(monument_name, "")

    st.image(image, use_container_width=True)

    # ── Title block ──
    st.markdown(
        f"""<div style="background:rgba(255,250,242,0.90); border:1px solid rgba(183,155,122,0.24); border-radius:22px; padding:1.2rem; box-shadow:0 10px 28px rgba(43,30,20,0.07);">
        <div style="text-transform:uppercase; letter-spacing:0.22em; font-size:0.72rem; color:#6d6258; margin-bottom:0.85rem;">Prediction</div>
        <h2 style="margin:0; font-size:2.5rem; line-height:1; color:#7e2330; font-family:'Cormorant Garamond',serif;">{monument_name}</h2>
        {
            f'<div style="font-size:0.88rem; color:var(--muted); margin-top:0.15rem; font-style:italic;">📍 {nested_note}</div>'
            if nested_note
            else ""
        }
        <div style="color:#6d6258; margin-top:0.35rem; font-size:0.94rem;">Runner-up: {runner_up['name']}</div>
        </div> """,
        unsafe_allow_html=True,
    )

    # ── Signal strip — separate call so formatting issues don't silently suppress it ──
    top_margin_str = f"{prediction['top_margin']:.3f}"   # compute outside the f-string
    confidence_pct = f"{top_result['probability'] * 100:.1f}%"

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.85rem; margin-top:1rem;">
            <div style="background:white; border:1px solid #deceb6; border-radius:16px; padding:0.9rem 1rem;">
                <div style="text-transform:uppercase; letter-spacing:0.18em; font-size:0.66rem; color:#6d6258; margin-bottom:0.35rem;">Confidence</div>
                <div style="font-size:1.1rem; color:#1f1a14;">{confidence_pct}</div>
            </div>
            <div style="background:white; border:1px solid #deceb6; border-radius:16px; padding:0.9rem 1rem;">
                <div style="text-transform:uppercase; letter-spacing:0.18em; font-size:0.66rem; color:#6d6258; margin-bottom:0.35rem;">Confidence Band</div>
                <div style="font-size:1.1rem; color:#1f1a14;">{conf_band}</div>
            </div>
            <div style="background:white; border:1px solid #deceb6; border-radius:16px; padding:0.9rem 1rem;">
                <div style="text-transform:uppercase; letter-spacing:0.18em; font-size:0.66rem; color:#6d6258; margin-bottom:0.35rem;">Top Gap</div>
                <div style="font-size:1.1rem; color:#1f1a14;">{top_margin_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(float(top_result["probability"]))
    caption_parts = [
        f"{prediction['prompt_count']} prompts",
        f"{prediction['view_count']} image views",
        f"runner-up: {runner_up['name']}",
        f"Model Used: {prediction.get('model_used', 'CLIP')}",
    ]
    if prediction.get("refined_by_specialist"):
        caption_parts.append(f"specialist refinement active ({prediction.get('specialist_cluster', 'cluster')})")
    st.caption(" | ".join(caption_parts))

    if False and conf_band == "Low":
        st.warning(
            "Low-confidence result. The top two classes are close, so treat this prediction cautiously.",
            icon="⚠️",
        )

    # ── Monument info card from metadata.json ──────────────────────────────
    metadata = load_monument_metadata()
    info = metadata.get(monument_name)
    if info:
        maps_url = f"https://www.google.com/maps/search/{info.get('maps_query', monument_name.replace(' ', '+'))}"
        history_text = info.get("history", "")
        fun_fact = info.get("fun_fact", "")
        location = info.get("location", "")
        built_by = info.get("built_by", "")
        year = info.get("year", "")
        style = info.get("style", "")
        hours = info.get("opening_hours", "")
        ticket_indian = info.get("ticket_price_indian", "")
        ticket_foreign = info.get("ticket_price_foreign", "")

        # History paragraph
        if history_text:
            st.markdown(
                f"""
                <div class="panel" style="margin-top: 1rem;">
                    <div class="section-label">History</div>
                    <p style="line-height: 1.7; color: var(--ink); font-size: 0.95rem; margin: 0;">{history_text}</p>
                    {f'<p style="margin-top: 0.8rem; font-size: 0.88rem; color: var(--muted); font-style: italic;">💡 {fun_fact}</p>' if fun_fact else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visit info card
        st.markdown(
            f"""
            <div class="panel" style="margin-top: 1rem;">
                <div class="section-label">Visit Information</div>
                <div class="detail-grid">
                    <div class="detail-card">
                        <div class="detail-label">Location</div>
                        <div class="detail-value">{location}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Built By</div>
                        <div class="detail-value">{built_by}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Period</div>
                        <div class="detail-value">{year}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Style</div>
                        <div class="detail-value">{style}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Opening Hours</div>
                        <div class="detail-value">{hours}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Ticket (Indian)</div>
                        <div class="detail-value">{ticket_indian}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Ticket (Foreign)</div>
                        <div class="detail-value">{ticket_foreign}</div>
                    </div>
                    <div class="detail-card" style="display: flex; align-items: center; justify-content: center;">
                        <a href="{maps_url}" target="_blank" style="
                            display: inline-flex; align-items: center; gap: 0.5rem;
                            background: linear-gradient(135deg, var(--accent), var(--accent-3));
                            color: white; padding: 0.65rem 1.2rem; border-radius: 12px;
                            text-decoration: none; font-weight: 600; font-size: 0.88rem;
                            transition: opacity 0.2s;
                        " onmouseover="this.style.opacity='0.85'" onmouseout="this.style.opacity='1'">
                            📍 Open in Google Maps
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Top scores ─────────────────────────────────────────────────────────
    with st.expander("See top scores"):
        for candidate in prediction["results"][:5]:
            st.markdown(
                f"""
                <div class="candidate-row">
                    <div>
                        <div class="candidate-name">{candidate["name"]}</div>
                        <div class="candidate-meta">{candidate["best_prompt"]}</div>
                    </div>
                    <div class="candidate-score">{candidate["probability"] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Render the Streamlit UI and run model inference for the selected image."""

    st.set_page_config(
        page_title="Mughal Architecture Identifier",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Mughal Architecture Identifier")
    render_model_status_badge(resolve_model_status(FINETUNED_MODEL_DIR))

    left_col, right_col = st.columns([1.0, 1.25], gap="large")

    with left_col:
        st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
        source = st.radio(
            "Choose how to provide the monument image",
            ["Upload file", "Paste screenshot", "Use camera"],
            horizontal=True,
            label_visibility="collapsed",
        )

        selected_image: Image.Image | None = None

        if source == "Upload file":
            uploaded = st.file_uploader(
                "Upload a monument photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Best results usually come from a clear exterior photo with the monument visible.",
            )
            selected_image = image_from_upload(uploaded)

        elif source == "Paste screenshot":
            st.caption("Click the button, then paste a copied screenshot or image from your clipboard.")
            selected_image = image_from_clipboard()
            if selected_image is not None and st.button("Clear pasted image", use_container_width=True):
                st.session_state.pop("pasted_image", None)
                st.rerun()

        else:
            camera_capture = st.camera_input("Take a monument photo")
            selected_image = image_from_camera(camera_capture)

    with right_col:
        if selected_image is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="section-label">Awaiting Image</div>
                    <div class="empty-state-title serif">Upload a monument photo</div>
                    <div class="empty-state-copy">
                        Use upload, paste, or camera on the left. The right panel will show the image and ranked class scores.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            model, processor, bank, specialist_bundle, device, _ = load_clip()
            with st.spinner("Running monument analysis..."):
                prediction = predict(selected_image, model, processor, bank, device)
                prediction["model_used"] = "CLIP"
                prediction = refine_with_specialist(selected_image, prediction, specialist_bundle, device)
                if prediction.get("refined_by_specialist"):
                    prediction["model_used"] = "Finetuned"
            if prediction["is_ood"]:
                render_ood_panel(prediction)
            else:
                render_result_panel(selected_image, prediction)


if __name__ == "__main__":
    main()
