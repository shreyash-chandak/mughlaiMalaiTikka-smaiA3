"""
T12.4 — Mughal Architecture Identifier
SMAI Assignment 3 | IIIT Hyderabad

Zero-shot classification using CLIP (openai/clip-vit-base-patch32)
"""

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import os
import requests
from io import BytesIO

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mughal Monument Identifier",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Josefin+Sans:wght@300;400;600&display=swap');

:root {
    --ivory:    #f5f0e8;
    --gold:     #c9a84c;
    --gold-lt:  #e8d49a;
    --maroon:   #6b1f24;
    --stone:    #8c7b6b;
    --dark:     #1a1410;
    --marble:   #ece9e0;
}

html, body, [data-testid="stApp"] {
    background-color: var(--ivory);
    color: var(--dark);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1, h2, h3, .monument-name {
    font-family: 'Cormorant Garamond', serif;
}
p, li, div, label, span, button {
    font-family: 'Josefin Sans', sans-serif;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1410 0%, #3d2b1a 50%, #6b1f24 100%);
    border-radius: 2px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '۞';
    position: absolute;
    font-size: 280px;
    color: rgba(201,168,76,0.07);
    top: -60px; right: -40px;
    line-height: 1;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 300;
    color: var(--gold-lt);
    letter-spacing: 0.04em;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem;
    color: rgba(232,212,154,0.6);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin: 0;
}
.hero-divider {
    width: 60px;
    height: 1px;
    background: var(--gold);
    margin: 1rem 0;
    opacity: 0.7;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--marble);
    border: 1.5px dashed var(--gold);
    border-radius: 2px;
    padding: 1.5rem;
}
[data-testid="stFileUploader"] label {
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    color: var(--stone) !important;
}

/* ── Prediction card ── */
.pred-card {
    background: white;
    border-top: 3px solid var(--gold);
    padding: 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(26,20,16,0.08);
}
.pred-rank {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--stone);
    margin-bottom: 0.3rem;
}
.pred-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--maroon);
    margin: 0 0 0.15rem;
    line-height: 1;
}
.pred-location {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--stone);
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}
.confidence-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--stone);
    margin-bottom: 0.3rem;
}

/* ── Info sections ── */
.info-section {
    background: var(--marble);
    border-left: 3px solid var(--gold);
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #3a2e26;
}
.info-tag {
    display: inline-block;
    background: var(--maroon);
    color: var(--gold-lt);
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.6rem;
}
.visit-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.visit-item {
    background: white;
    padding: 0.8rem 1rem;
    border-bottom: 2px solid var(--gold-lt);
}
.visit-label {
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--stone);
    font-family: 'Josefin Sans', sans-serif;
}
.visit-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    color: var(--dark);
    line-height: 1.3;
}

/* ── Fun fact callout ── */
.fun-fact {
    background: linear-gradient(135deg, #6b1f24 0%, #3d2b1a 100%);
    color: var(--gold-lt);
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-style: italic;
    line-height: 1.6;
}
.fun-fact-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
    font-style: normal;
}

/* ── Other predictions ── */
.other-card {
    background: white;
    border-left: 2px solid var(--gold-lt);
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.other-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    color: var(--dark);
}
.other-pct {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--stone);
}

/* ── Maps button ── */
.maps-btn {
    display: inline-block;
    background: var(--gold);
    color: var(--dark) !important;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    text-decoration: none;
    padding: 0.6rem 1.4rem;
    margin-top: 1rem;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
}
.maps-btn:hover { background: var(--gold-lt); }

/* ── How it works ── */
.how-card {
    background: white;
    border-top: 2px solid var(--gold-lt);
    padding: 1.2rem 1.2rem 1rem;
    text-align: center;
}
.how-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.how-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    color: var(--maroon);
    margin-bottom: 0.3rem;
}
.how-body {
    font-size: 0.75rem;
    color: var(--stone);
    line-height: 1.5;
}

/* ── Monument chips ── */
.monument-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.chip {
    background: var(--marble);
    border: 1px solid var(--gold-lt);
    color: var(--stone);
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 0.25rem 0.6rem;
}

/* ── Streamlit progress bar tint ── */
[data-testid="stProgressBar"] > div > div {
    background-color: var(--gold) !important;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--stone);
    border-bottom: 1px solid var(--gold-lt);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MONUMENTS = [
    "Taj Mahal",
    "Humayun's Tomb",
    "Red Fort",
    "Agra Fort",
    "Fatehpur Sikri",
    "Itmad-ud-Daulah",
    "Jama Masjid Delhi",
    "Qutub Minar",
    "Bibi Ka Maqbara",
    "Akbar's Tomb",
    "Lahore Fort",
    "Badshahi Mosque",
    "Shalimar Bagh",
    "Moti Masjid Agra",
    "Safdarjung Tomb",
]

# Descriptive text prompts improve CLIP accuracy
PROMPTS = {
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
    "Lahore Fort": "a photograph of Lahore Fort also called Shahi Qila, a large Mughal fortress with decorated facades and the Sheesh Mahal palace in Lahore Pakistan",
    "Badshahi Mosque": "a photograph of Badshahi Mosque in Lahore, a grand Mughal mosque with large red sandstone courtyard and three white marble onion domes in Pakistan",
    "Shalimar Bagh": "a photograph of Shalimar Bagh in Srinagar, a terraced Mughal garden with fountains, water channels and chinar trees beside Dal Lake in Kashmir",
    "Moti Masjid Agra": "a photograph of Moti Masjid inside Agra Fort, a small pure white marble Mughal mosque with three marble domes and graceful arches",
    "Safdarjung Tomb": "a photograph of Safdarjung Tomb, a late Mughal sandstone mausoleum with a central dome and four corner towers surrounded by gardens in New Delhi",
}

# ─── Load metadata ───────────────────────────────────────────────────────────
@st.cache_data
def load_metadata():
    meta_path = os.path.join(os.path.dirname(__file__), "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Load CLIP model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_clip():
    with st.spinner("Loading CLIP model (first run only)…"):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    return model, processor

# ─── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image: Image.Image, model, processor):
    texts = [PROMPTS[m] for m in MONUMENTS]
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()

    results = sorted(
        zip(MONUMENTS, probs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return results  # list of (name, prob)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-sub">SMAI Assignment 3 · T12.4 · IIIT Hyderabad</div>
    <div class="hero-divider"></div>
    <div class="hero-title">Mughal Monument<br>Identifier</div>
    <div style="margin-top:1rem; font-family:'Josefin Sans',sans-serif; font-size:0.8rem; color:rgba(232,212,154,0.55); max-width:480px; line-height:1.6;">
        Upload a photograph of any Mughal-era monument and the app will identify it using CLIP zero-shot classification — no model training required.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ──────────────────────────────────────────────────────────────────
meta = load_metadata()

col_left, col_right = st.columns([1, 1.35], gap="large")

with col_left:
    st.markdown('<div class="section-header">Upload Monument Photo</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a JPG / PNG image here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Photograph of a Mughal monument — taken from any angle.",
    )

    # "How it works" mini-cards
    st.markdown('<div class="section-header" style="margin-top:2rem;">How It Works</div>', unsafe_allow_html=True)
    hw1, hw2, hw3 = st.columns(3)
    for col, icon, title, body in [
        (hw1, "🖼️", "Upload", "Any photo of a Mughal monument"),
        (hw2, "🧠", "CLIP", "Zero-shot image-text similarity"),
        (hw3, "📜", "Identify", "Name, history & visit info"),
    ]:
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-icon">{icon}</div>
                <div class="how-title">{title}</div>
                <div class="how-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:2rem;">Supported Monuments</div>', unsafe_allow_html=True)
    chips_html = '<div class="monument-chips">'
    for m in MONUMENTS:
        chips_html += f'<span class="chip">{m}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


with col_right:
    if uploaded is None:
        st.markdown("""
        <div style="height:380px; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; background:white; border:1px solid #e8d49a;">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">🕌</div>
            <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; color:#8c7b6b; opacity:0.6;">
                Awaiting photograph…
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")

        # Show image
        st.image(image, use_container_width=True)

        # Load model & predict
        model, processor = load_clip()

        with st.spinner("Analysing monument…"):
            results = predict(image, model, processor)

        top_name, top_prob = results[0]
        info = meta.get(top_name, {})

        # ── Top prediction card ──
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-rank">Top Prediction</div>
            <div class="pred-name">{top_name}</div>
            <div class="pred-location">📍 {info.get('location', 'Mughal Empire')}</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(top_prob))
        st.markdown(f"""<div style="font-family:'Josefin Sans',sans-serif; font-size:0.85rem; color:#6b1f24; margin-top:-0.5rem;">
            <strong>{top_prob*100:.1f}%</strong> confidence
        </div>""", unsafe_allow_html=True)

        # ── History ──
        st.markdown(f"""
        <div class="info-section">
            <div class="info-tag">History</div><br>
            {info.get('history', 'Historical information not available.')}
        </div>
        """, unsafe_allow_html=True)

        # ── Visit info grid ──
        st.markdown(f"""
        <div style="margin:0.8rem 0 0.3rem;">
            <div class="info-tag">Visit Information</div>
        </div>
        <div class="visit-grid">
            <div class="visit-item">
                <div class="visit-label">Built By</div>
                <div class="visit-value">{info.get('built_by', '—')}</div>
            </div>
            <div class="visit-item">
                <div class="visit-label">Year</div>
                <div class="visit-value">{info.get('year', '—')}</div>
            </div>
            <div class="visit-item">
                <div class="visit-label">Opening Hours</div>
                <div class="visit-value">{info.get('opening_hours', '—')}</div>
            </div>
            <div class="visit-item">
                <div class="visit-label">Entry (Indian / Foreign)</div>
                <div class="visit-value">{info.get('ticket_price_indian', '—')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Fun Fact ──
        if info.get("fun_fact"):
            st.markdown(f"""
            <div class="fun-fact">
                <div class="fun-fact-label">✦ Did You Know?</div>
                {info['fun_fact']}
            </div>
            """, unsafe_allow_html=True)

        # ── Google Maps link ──
        maps_url = f"https://www.google.com/maps/search/?api=1&query={info.get('maps_query', top_name.replace(' ', '+'))}"
        st.markdown(f'<a href="{maps_url}" target="_blank" class="maps-btn">🗺 Open in Google Maps</a>', unsafe_allow_html=True)

        # ── Other predictions ──
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Other Candidates</div>', unsafe_allow_html=True)
        for name, prob in results[1:4]:
            pct = prob * 100
            loc = meta.get(name, {}).get("location", "")
            st.markdown(f"""
            <div class="other-card">
                <div>
                    <div class="other-name">{name}</div>
                    <div style="font-family:'Josefin Sans',sans-serif; font-size:0.68rem; color:#8c7b6b; letter-spacing:0.08em;">{loc}</div>
                </div>
                <div class="other-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none; border-top:1px solid #e8d49a; margin:3rem 0 1rem;">
<div style="font-family:'Josefin Sans',sans-serif; font-size:0.65rem; color:#8c7b6b;
            letter-spacing:0.2em; text-align:center; text-transform:uppercase; padding-bottom:1rem;">
    T12.4 · Mughal Architecture Identifier · SMAI Assignment 3 · IIIT Hyderabad 2025–26 ·
    Model: CLIP ViT-B/32 (Zero-Shot) · Metadata sourced from Wikipedia &amp; cached as JSON
</div>
""", unsafe_allow_html=True)
