import html as _html
import json
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="PAM50 Classifier | TCGA-BRCA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "final_model"
MODEL_PATH = MODEL_DIR / "tuned_mlp_final.keras"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"
GENES_PATH = MODEL_DIR / "selected_gene_names.json"
LABELS_PATH = MODEL_DIR / "label_mapping.json"


# ------------------------------------------------------------
# CONSTANTS & EVAL/INTERPRETATION HELPERS
# ------------------------------------------------------------
PAM50_SYMBOLS = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "PGR", "PHGDH", "PTTG1",
    "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
    "NUF2", "NDC80", "ORC6",            # simbol terbaru
    "CDCA1", "KNTC2", "ORC6L",          # alias lama pada publikasi 2009
}

def is_pam50(name):
    sym = str(name).split("|")[0].strip().upper()
    return sym in PAM50_SYMBOLS

def compute_metrics(y_true, y_pred, class_names):
    labels = list(class_names)
    present = [c for c in labels if (y_true == c).any()]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "Subtipe": labels, "Presisi": prec * 100, "Recall": rec * 100, "F1": f1 * 100, "Dukungan": sup.astype(int),
    })
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "n": len(y_true),
        "n_wrong": int((y_true != y_pred).sum()),
        "per_class": per_class,
        "cm": cm
    }

def first_layer_importance(model, n_genes):
    for layer in model.layers:
        ws = layer.get_weights()
        if ws and np.ndim(ws[0]) == 2 and ws[0].shape[0] == n_genes:
            return np.abs(ws[0]).mean(axis=1)
    return None

def model_layers_table(model):
    rows = []
    for layer in model.layers:
        try:
            shape = str(layer.output_shape) if hasattr(layer, "output_shape") else str(tuple(layer.output.shape))
        except:
            shape = "-"
        try:
            params = int(layer.count_params())
        except:
            params = 0
        rows.append({"Lapisan": layer.name, "Tipe": type(layer).__name__, "Bentuk keluaran": shape, "Parameter": params})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# VISUAL THEME
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --ink:#3F1D34; --rose:#D9568A; --muted:#8C6577;
  --line:#F4DCE5; --canvas:#FFF6F9; --white:#FFFFFF;
  --track:#F8E6ED;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(900px 380px at 88% -6%,#FFE0EB 0%,rgba(255,224,235,0) 70%),var(--canvas); color:#43243A;}
[data-testid="stHeader"] {background:rgba(255,246,249,.9);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#FDE8F0 0%,#F9D6E4 100%);border-right:1px solid #F3C9DA;}
[data-testid="stSidebar"] * {color:#4B2640;}
[data-testid="stSidebar"] hr {border-color:rgba(75,38,64,.14);}
.block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;color:var(--ink)!important;letter-spacing:-.035em;}
h4 {font-family:'Manrope',sans-serif!important;color:var(--ink)!important;font-weight:700!important;letter-spacing:-.02em;font-size:1.05rem!important;}
h1 {font-weight:800!important;} h2,h3 {font-weight:700!important;}
h2 {font-size:1.85rem!important;}
hr {margin:2.4rem 0 1.8rem!important;border:none!important;border-top:1px solid var(--line)!important;}
footer, #MainMenu {visibility:hidden;}
.hero {
  padding:30px 32px;border-radius:24px;color:#3F1D34;margin-bottom:22px;border:1px solid #F6CADB;
  background:radial-gradient(circle at 1px 1px,rgba(217,86,138,.13) 1px,transparent 0) 0 0/22px 22px,linear-gradient(120deg,#FFE9F0 0%,#FCD6E4 55%,#F8C2D6 100%);
  box-shadow:0 12px 35px rgba(217,86,138,.16);position:relative;overflow:hidden;
}
.hero:after {
  content:"";position:absolute;width:230px;height:230px;border:1px solid rgba(217,86,138,.24);
  border-radius:50%;right:-45px;top:-100px;box-shadow:0 0 0 30px rgba(217,86,138,.07),0 0 0 60px rgba(217,86,138,.05);
}
.kicker {color:#B83A68;font-size:.75rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;}
.hero-title {color:#3F1D34!important;font-family:'Manrope',sans-serif;font-weight:800;font-size:clamp(1.7rem,3vw,2.45rem);line-height:1.2;letter-spacing:-.045em;max-width:800px;margin:0 0 12px;}
.hero-copy {color:#6A4A5C;line-height:1.65;max-width:780px;margin:0;}
.tag-row {display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;}
.tag {padding:6px 11px;border:1px solid rgba(184,58,104,.28);border-radius:99px;background:rgba(255,255,255,.72);color:#8A2D55;font-size:.78rem;font-weight:700;}
.eyebrow {display:flex;align-items:center;gap:10px;color:var(--rose);font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:5px;}
.eyebrow:before {content:"";width:24px;height:2px;border-radius:2px;background:var(--rose);}
.section-note {color:var(--muted);font-size:.92rem;line-height:1.6;margin:0 0 16px;}
.section-note b {color:var(--ink);}
.card {background:#fff;border:1px solid var(--line);border-radius:17px;padding:20px 22px;box-shadow:0 4px 16px #3f1d3408;height:100%;}
.card-title {color:var(--ink);font-weight:800;margin-bottom:7px;}
.card-copy {color:var(--muted);font-size:.9rem;line-height:1.6;}
.mini-label {color:var(--muted);font-size:.76rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;}
.mini-value {color:var(--ink);font-family:'Manrope',sans-serif;font-weight:800;font-size:1.6rem;margin-top:5px;}
.pill {display:inline-block;padding:6px 10px;border-radius:99px;background:#E8F6EE;color:#1E7B4F;font-size:.78rem;font-weight:800;}
div[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-left:4px solid var(--rose);padding:16px 18px;border-radius:15px;box-shadow:0 4px 16px #3f1d3408;}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {color:var(--muted)!important;font-weight:600;}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {color:var(--ink)!important;font-family:'Manrope',sans-serif;font-weight:800;}
div[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color:var(--muted)!important;line-height:1.55;}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"] {background:linear-gradient(135deg,#D9568A,#C23E73);border:none;color:#fff;border-radius:12px;font-weight:800;min-height:48px;box-shadow:0 6px 16px rgba(217,86,138,.28);transition:transform .15s ease,box-shadow .15s ease;}
.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="stBaseButton-primary"]:hover {transform:translateY(-1px);box-shadow:0 10px 22px rgba(217,86,138,.34);color:#fff;}
.stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p {color:#fff!important;}
.stDownloadButton>button {border-radius:12px;font-weight:700;min-height:46px;border:1.5px solid var(--rose);background:#fff;}
.stDownloadButton>button, .stDownloadButton>button p {color:#C23E73!important;}
.stDownloadButton>button:hover {background:#FDE7EF;border-color:#C23E73;}
[data-testid="stFileUploader"] {background:#fff;border:1px solid var(--line);border-radius:17px;padding:14px;box-shadow:0 4px 16px #3f1d3408;}
[data-testid="stFileUploaderDropzone"] {background:#FFF3F7;border:1.5px dashed #EFA9C2;border-radius:13px;}
[data-testid="stFileUploaderDropzone"]:hover {background:#FDE6EE;border-color:var(--rose);}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 4px 16px #3f1d3408;}
[data-testid="stExpander"] {background:#fff;border:1px solid var(--line)!important;border-radius:14px;}
[data-testid="stExpander"] summary {font-weight:700;color:var(--ink);}
/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap:4px;background:#F9E3EB;padding:4px;border-radius:13px;width:fit-content;max-width:100%;margin-bottom:16px;border-bottom:none;}
.stTabs button[data-baseweb="tab"] {padding:9px 18px;height:auto;border-radius:10px;font-weight:700;background:transparent;}
.stTabs button[data-baseweb="tab"] p {color:var(--muted)!important;font-weight:700;}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {background:#fff;box-shadow:0 2px 8px rgba(63,29,52,.10);}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {color:var(--ink)!important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none!important;}
/* ---------- Hasil: distribusi subtipe ---------- */
.dist-row {display:grid;grid-template-columns:10px minmax(96px,128px) 1fr auto;align-items:center;gap:12px;padding:9px 0;}
.dist-row + .dist-row {border-top:1px solid #F8E8EE;}
.dist-row.is-zero {opacity:.45;}
.dot {width:10px;height:10px;border-radius:50%;display:block;}
.dist-name {color:var(--ink);font-weight:700;font-size:.9rem;}
.dist-track {height:10px;background:var(--track);border-radius:99px;overflow:hidden;display:block;}
.dist-fill {display:block;height:100%;border-radius:99px;}
.dist-num {color:var(--ink);font-weight:800;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;}
.dist-num small {color:var(--muted);font-weight:600;margin-left:8px;font-size:.8rem;}
/* ---------- Hasil: detail sampel ---------- */
.result-card {min-height:158px;}
.subtype-name {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.6rem;letter-spacing:-.03em;color:var(--ink);margin:6px 0 4px;line-height:1.2;}
.runner {margin-top:12px;padding-top:11px;border-top:1px solid var(--line);color:var(--muted);font-size:.86rem;}
.runner b {color:var(--ink);}
.ring-card {display:flex;align-items:center;gap:20px;}
.ring {width:100px;height:100px;border-radius:50%;display:grid;place-items:center;flex:none;}
.ring-inner {width:78px;height:78px;border-radius:50%;background:#fff;display:grid;place-items:center;font-family:'Manrope',sans-serif;font-weight:800;font-size:1.08rem;color:var(--ink);letter-spacing:-.02em;}
.flag {display:inline-block;background:#FFF4DE;color:#8A5A00;border-radius:9px;padding:6px 10px;font-size:.8rem;font-weight:600;line-height:1.45;margin-top:8px;}
.ok {display:inline-block;background:#E8F6EE;color:#1E7B4F;border-radius:9px;padding:6px 10px;font-size:.8rem;font-weight:600;line-height:1.45;margin-top:8px;}
.prob-row {display:grid;grid-template-columns:minmax(96px,128px) 1fr 62px;align-items:center;gap:14px;padding:8px 0;}
.prob-name {color:var(--muted);font-size:.9rem;font-weight:600;}
.prob-track {height:14px;background:var(--track);border-radius:99px;overflow:hidden;display:block;}
.prob-fill {display:block;height:100%;border-radius:99px;opacity:.5;}
.prob-val {text-align:right;color:var(--muted);font-weight:700;font-variant-numeric:tabular-nums;}
.prob-row.is-top .prob-name {color:var(--ink);font-weight:800;}
.prob-row.is-top .prob-fill {opacity:1;}
.prob-row.is-top .prob-val {color:var(--ink);font-weight:800;}
/* ---------- Hasil: unduh ---------- */
.stat-row {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0 18px;}
.stat {background:#fff;border:1px solid var(--line);border-radius:13px;padding:13px 16px;}
.stat-val {color:var(--ink);font-family:'Manrope',sans-serif;font-weight:800;font-size:1.25rem;margin-top:4px;}
.footer {color:#A3808F;font-size:.78rem;line-height:1.6;text-align:center;padding-top:22px;margin-top:40px;border-top:1px solid var(--line);}
/* ---------- Pemberitahuan penting ---------- */
.notice {background:#FFFAEE;border:1px solid #F0DEB0;border-left:6px solid #E0A030;border-radius:17px;padding:20px 24px 22px;margin:4px 0 18px;}
.notice-head {display:flex;align-items:center;gap:12px;font-family:'Manrope',sans-serif;font-weight:800;font-size:1.08rem;color:#6B4300;margin-bottom:16px;letter-spacing:-.02em;}
.notice-icon {width:26px;height:26px;border-radius:50%;background:#E0A030;color:#fff;display:grid;place-items:center;font-weight:800;font-size:.95rem;flex:none;}
.notice-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:0;}
.notice-item {padding:0 20px;}
.notice-item:first-child {padding-left:0;}
.notice-item + .notice-item {border-left:1px solid #EFE0B8;}
.notice-item b {display:block;color:var(--ink);font-size:.95rem;margin-bottom:5px;}
.notice-item span {display:block;color:#6A4A5C;font-size:.88rem;line-height:1.65;}
/* ---------- Edukasi ---------- */
.guide-intro {color:#5B3A4E;line-height:1.75;font-size:.95rem;margin:6px 0 16px;}
.guide-intro b {color:var(--ink);}
.guide-source {color:var(--muted);font-size:.8rem;line-height:1.6;margin-top:14px;}
.info-grid {display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:4px;}
.bullets {margin:8px 0 0;padding-left:18px;color:#5B3A4E;font-size:.9rem;line-height:1.7;}
.bullets li {margin-bottom:5px;}
.glossary {background:#FCEAF1;border-radius:12px;padding:12px 16px;color:#6B2A4A;font-size:.86rem;line-height:1.7;margin-bottom:16px;}
.glossary b {color:#B83A68;}
.sub-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:14px;margin:4px 0 4px;}
.sub-card {background:#fff;border:1px solid var(--line);border-top:5px solid #ccc;border-radius:15px;padding:16px 18px 15px;box-shadow:0 4px 16px #3f1d3408;}
.sub-name {font-family:'Manrope',sans-serif;font-weight:800;color:var(--ink);font-size:1.1rem;letter-spacing:-.02em;}
.sub-tag {color:var(--muted);font-size:.8rem;font-weight:700;margin:2px 0 9px;}
.sub-copy {color:#5B3A4E;font-size:.9rem;line-height:1.65;}
.sub-meta {margin-top:11px;padding-top:10px;border-top:1px solid #F8E8EE;font-size:.82rem;line-height:1.65;color:var(--muted);}
.sub-meta b {color:var(--ink);}
.about-card {margin:16px 0 6px;}
.about-title {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.2rem;color:var(--ink);letter-spacing:-.02em;margin:5px 0 8px;}
.about-title span {color:var(--muted);font-weight:600;font-size:.86rem;font-family:'DM Sans',sans-serif;margin-left:8px;}
/* ---------- Tambahan tampilan ---------- */
.live {display:inline-block;width:8px;height:8px;border-radius:50%;background:#12B76A;margin-right:8px;animation:pulse 2.2s ease-out infinite;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(18,183,106,.55);}70%{box-shadow:0 0 0 8px rgba(18,183,106,0);}100%{box-shadow:0 0 0 0 rgba(18,183,106,0);}}
.stack {display:flex;gap:3px;height:14px;margin:10px 0 8px;}
.stack span {display:block;border-radius:99px;min-width:6px;}
.sub-card {transition:transform .18s ease,box-shadow .18s ease;}
.sub-card:hover {transform:translateY(-3px);box-shadow:0 12px 26px #3f1d3414;}
.empty {border-style:dashed!important;background:#FFFAFC!important;}
.side-label {color:#B83A68!important;font-size:.72rem;font-weight:800;letter-spacing:.14em;margin:18px 0 10px;}
.side-step {display:flex;align-items:center;gap:11px;font-size:.9rem;padding:6px 0;}
.side-step span {width:22px;height:22px;border-radius:50%;border:1px solid rgba(184,58,104,.45);color:#B83A68!important;font-size:.74rem;font-weight:800;display:grid;place-items:center;flex:none;}
.side-sub {display:flex;align-items:center;gap:11px;font-size:.88rem;padding:5px 0;}
.side-sub i {width:9px;height:9px;border-radius:50%;display:block;flex:none;}
/* ---------- Animasi loading ---------- */
.loader {position:relative;overflow:hidden;display:flex;align-items:center;gap:24px;background:#fff;border:1px solid var(--line);border-radius:17px;padding:20px 26px 22px;box-shadow:0 4px 16px #3f1d3408;margin:12px 0 16px;}
.loader:after {content:"";position:absolute;left:-40%;bottom:0;height:3px;width:40%;background:linear-gradient(90deg,transparent,var(--rose),transparent);animation:sweep 1.7s ease-in-out infinite;}
.helix {display:flex;gap:7px;flex:none;}
.hx {position:relative;display:block;width:8px;height:46px;}
.hx:before {content:"";position:absolute;left:3px;top:5px;bottom:5px;width:2px;border-radius:2px;background:rgba(217,86,138,.16);}
.hx b {position:absolute;left:0;width:8px;height:8px;border-radius:50%;}
.hx b:nth-child(1) {background:#D9568A;animation:strandA 1.5s ease-in-out infinite;animation-delay:calc(var(--i) * -.15s);}
.hx b:nth-child(2) {background:#F5A9C5;animation:strandB 1.5s ease-in-out infinite;animation-delay:calc(var(--i) * -.15s);}
@keyframes strandA {0%,100%{top:0;transform:scale(1);opacity:1;}50%{top:38px;transform:scale(.6);opacity:.55;}}
@keyframes strandB {0%,100%{top:38px;transform:scale(.6);opacity:.55;}50%{top:0;transform:scale(1);opacity:1;}}
@keyframes sweep {0%{left:-40%;}100%{left:100%;}}
.loader-title {font-family:'Manrope',sans-serif;font-weight:800;color:var(--ink);font-size:1.02rem;letter-spacing:-.02em;}
.loader-sub {color:var(--muted);font-size:.88rem;line-height:1.55;margin-top:3px;}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
@media(max-width:768px){
  .block-container{padding-top:1rem}.hero{padding:24px 20px;border-radius:18px;}
  .dist-row{grid-template-columns:10px 88px 1fr auto;gap:8px;}
  .prob-row{grid-template-columns:88px 1fr 54px;gap:8px;}
  .ring-card{flex-direction:column;align-items:flex-start;}
  .stat-row{grid-template-columns:1fr;}
  .notice-grid{grid-template-columns:1fr;gap:14px;}
  .notice-item,.notice-item:first-child{padding:0;}
  .notice-item + .notice-item{border-left:none;border-top:1px solid #EFE0B8;padding-top:14px;}
  .info-grid{grid-template-columns:1fr;}
  .loader{flex-direction:column;align-items:flex-start;gap:14px;}
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DISPLAY HELPERS (hanya untuk tampilan, tidak memengaruhi model)
# ------------------------------------------------------------
_FALLBACK_COLORS = ["#D9568A", "#6B7FD7", "#D9962B", "#8A6BBE", "#5B6B7A"]


def esc(value):
    return _html.escape(str(value))


SUBTYPE_COLORS = {
    "basal": "#C8433F", "her2": "#B04A9C", "luma": "#1F5F99",
    "lumb": "#4DA3D9", "normal": "#3E9B6B",
}


def subtype_key(name):
    key = str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
    if "basal" in key: return "basal"
    if "her2" in key: return "her2"
    if "luminala" in key or "luma" in key: return "luma"
    if "luminalb" in key or "lumb" in key: return "lumb"
    if "normal" in key: return "normal"
    return None

def subtype_color(name):
    k = subtype_key(name)
    if k: return SUBTYPE_COLORS[k]
    key = str(name).lower()
    return _FALLBACK_COLORS[sum(map(ord, key)) % len(_FALLBACK_COLORS)]

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def subtype_cell(value):
    return f"background-color:{hex_to_rgba(subtype_color(value), .16)};color:#3F1D34;font-weight:700"

def style_map(styler, func, subset):
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)

def _fill(color):
    return f"linear-gradient(90deg,{hex_to_rgba(color, .55)},{color})"

def distribution_html(counts, total):
    segs = "".join(
        f'<span style="flex:{int(c)};background:{subtype_color(n)}" title="{esc(n)}: {int(c)}"></span>'
        for n, c in counts.items() if c > 0
    )
    rows = ""
    for name, count in counts.items():
        pct = count / total * 100 if total else 0
        color = subtype_color(name)
        zero = " is-zero" if count == 0 else ""
        rows += (
            f'<div class="dist-row{zero}"><span class="dot" style="background:{color}"></span>'
            f'<span class="dist-name">{esc(name)}</span>'
            f'<span class="dist-track"><span class="dist-fill" style="width:{pct:.2f}%;background:{_fill(color)}"></span></span>'
            f'<span class="dist-num">{int(count):,}<small>{pct:.1f}%</small></span></div>'
        )
    return f'<div class="card"><div class="mini-label">Komposisi sampel</div><div class="stack">{segs}</div>{rows}</div>'

def probability_html(names, probs, highlight):
    order = sorted(range(len(names)), key=lambda i: -probs[i])
    rows = ""
    for i in order:
        name, p = names[i], float(probs[i])
        top = " is-top" if name == highlight else ""
        rows += (
            f'<div class="prob-row{top}"><span class="prob-name">{esc(name)}</span>'
            f'<span class="prob-track"><span class="prob-fill" style="width:{max(p * 100, 0.6):.2f}%;background:{_fill(subtype_color(name))}"></span></span>'
            f'<span class="prob-val">{p:.1%}</span></div>'
        )
    return f'<div class="card">{rows}</div>'

def confidence_ring_html(conf, color):
    return (
        f'<div class="ring" style="background:conic-gradient({color} {conf * 360:.1f}deg,#F6E3EA 0)">'
        f'<div class="ring-inner">{conf:.1%}</div></div>'
    )

def loader_html(title, subtitle=""):
    cols = "".join(f'<span class="hx" style="--i:{i}"><b></b><b></b></span>' for i in range(12))
    sub = f'<div class="loader-sub">{esc(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="loader" role="status" aria-live="polite"><div class="helix">{cols}</div>'
        f'<div><div class="loader-title">{esc(title)}</div>{sub}</div></div>'
    )

@contextmanager
def loading_panel(title, subtitle=""):
    slot = st.empty()
    slot.markdown(loader_html(title, subtitle), unsafe_allow_html=True)
    try:
        yield
    finally:
        slot.empty()

def harmonize_labels(series, class_names, label_mapping):
    raw = series.astype("object")
    valid = set(class_names)
    key_by_class = {c: subtype_key(c) for c in class_names}
    mapping = {}
    for v in pd.unique(raw.dropna()):
        s = str(v).strip()
        target = None
        if s in valid:
            target = s
        elif s.isdigit() and str(int(s)) in label_mapping:
            target = label_mapping[str(int(s))]
        else:
            k = subtype_key(s)
            if k: target = next((c for c in class_names if key_by_class[c] == k), None)
        if target is not None:
            mapping[v] = target
    return raw.map(mapping)

SUBTYPE_GUIDE = {
    "luma": {
        "name": "Luminal A",
        "tag": "Tumbuh lambat · prognosis relatif terbaik",
        "markers": "ER+ / PR+ / HER2−, proliferasi rendah",
        "desc": "Subtipe yang paling sering ditemukan. Sel kanker masih bergantung pada hormon estrogen dan membelah relatif lambat. Risiko kambuhnya paling rendah dibanding subtipe lain.",
        "care": "Terapi hormonal; kemoterapi tidak selalu diperlukan.",
    },
    "lumb": {
        "name": "Luminal B",
        "tag": "Lebih cepat tumbuh · risiko di atas Luminal A",
        "markers": "ER+ (PR sering lebih rendah), HER2− atau HER2+, proliferasi tinggi",
        "desc": "Juga bergantung pada hormon, tetapi selnya membelah lebih cepat dan gen proliferasinya lebih aktif. Risiko kambuh lebih tinggi daripada Luminal A.",
        "care": "Terapi hormonal, sering dikombinasikan dengan kemoterapi; terapi anti-HER2 bila HER2-positif.",
    },
    "her2": {
        "name": "HER2-enriched",
        "tag": "Agresif · tersedia terapi target",
        "markers": "Gen HER2 sangat aktif; ER/PR sering rendah atau negatif",
        "desc": "Aktivitas gen HER2 yang tinggi membuat sel tumbuh cepat. Dahulu tergolong berat, tetapi terapi target anti-HER2 telah memperbaiki hasil pengobatan secara nyata.",
        "care": "Terapi anti-HER2 (misalnya trastuzumab) bersama kemoterapi.",
    },
    "basal": {
        "name": "Basal-like",
        "tag": "Agresif · tumbuh cepat",
        "markers": "Sering triple-negative (ER−, PR−, HER2−)",
        "desc": "Pola ekspresi gennya mirip sel basal pada saluran susu. Tumbuh cepat dan lebih sering kambuh dalam beberapa tahun pertama. Karena tidak bergantung pada hormon maupun HER2, terapi hormonal dan anti-HER2 tidak efektif. Basal-like dan triple-negative banyak tumpang tindih, tetapi tidak identik.",
        "care": "Kemoterapi sebagai pilar utama; terapi tambahan bergantung pada pemeriksaan lanjutan.",
    },
    "normal": {
        "name": "Normal-like",
        "tag": "Kategori kecil · sering terkait kualitas sampel",
        "markers": "Profil gen mirip jaringan payudara normal",
        "desc": "Ekspresi gennya menyerupai jaringan payudara normal. Pada banyak kasus ini muncul karena sampel hanya sedikit mengandung sel tumor, sehingga sinyal jaringan normal mendominasi. Ini bukan bukti bahwa jaringan sampel normal, dan sebagian peneliti menilainya lebih sebagai artefak pengambilan sampel.",
        "care": "Tidak ada pendekatan khusus; keputusan bergantung pada pemeriksaan patologi dan klinis.",
    },
}

CANCER_INFO_HTML = (
    '<p class="guide-intro"><b>Apa itu kanker payudara?</b> Kanker ini muncul ketika sel di jaringan payudara, '
    'paling sering di saluran atau kelenjar susu, tumbuh tak terkendali. Sel tersebut dapat menyerang jaringan di sekitarnya '
    'dan, pada tahap lanjut, menyebar (metastasis) lewat kelenjar getah bening dan aliran darah ke organ lain seperti tulang, paru, hati, dan otak. '
    'Penyakit ini sangat beragam: dua pasien bisa memiliki perilaku tumor dan pengobatan yang berbeda, '
    'sehingga tumor dikelompokkan ke dalam subtipe molekuler seperti PAM50.</p>'
    '<div class="stat-row">'
    '<div class="stat"><div class="mini-label">Kasus baru 2022</div><div class="stat-val">2,3 juta</div>'
    '<div class="card-copy">perempuan didiagnosis di seluruh dunia</div></div>'
    '<div class="stat"><div class="mini-label">Kematian 2022</div><div class="stat-val">670 ribu</div>'
    '<div class="card-copy">perkiraan kematian akibat kanker payudara</div></div>'
    '<div class="stat"><div class="mini-label">Bertahan hidup</div><div class="stat-val">83%</div>'
    '<div class="card-copy">di negara berpendapatan tinggi; di negara berpendapatan rendah kurang dari 50%</div></div>'
    '</div>'
    '<div class="info-grid">'
    '<div class="card"><div class="card-title">Mengapa berbahaya?</div><ul class="bullets">'
    '<li>Kanker payudara adalah kanker tersering pada perempuan di 157 dari 185 negara (WHO, 2022).</li>'
    '<li>Pada tahap dini biasanya tanpa gejala, sehingga mudah terlewat.</li>'
    '<li>Makin lanjut stadiumnya, makin sulit diobati. Setelah menyebar ke organ lain, penyakit umumnya sulit disembuhkan tuntas dan lebih sering hanya dapat dikendalikan.</li>'
    '<li>Sebagian subtipe (Luminal B, HER2-enriched, basal-like) memiliki risiko kambuh lebih tinggi daripada Luminal A.</li>'
    '<li>Pengobatan lebih efektif dan lebih ringan bila dimulai sejak dini.</li>'
    '</ul></div>'
    '<div class="card"><div class="card-title">Tanda yang perlu diwaspadai</div><ul class="bullets">'
    '<li>Benjolan atau penebalan baru di payudara atau ketiak.</li>'
    '<li>Perubahan ukuran atau bentuk payudara.</li>'
    '<li>Kulit berkerut, melesak, kemerahan, atau tampak seperti kulit jeruk.</li>'
    '<li>Puting tertarik ke dalam, atau keluar cairan yang tidak biasa (terutama berdarah).</li>'
    '</ul><div class="card-copy" style="margin-top:10px">Sebagian besar benjolan payudara bukan kanker, tetapi tetap perlu diperiksa. '
    'Bila menemukan perubahan, segera periksakan diri ke dokter atau fasilitas kesehatan, dan lakukan skrining sesuai usia dan faktor risiko.</div></div>'
    '</div>'
    '<div class="guide-source">Sumber ringkasan: WHO Breast Cancer Fact Sheet dan IARC (data 2022). Informasi umum, bukan nasihat medis.</div>'
)

def subtype_guide_html():
    cards = ""
    for key in ["luma", "lumb", "her2", "basal", "normal"]:
        g = SUBTYPE_GUIDE[key]
        cards += (
            f'<div class="sub-card" style="border-top-color:{SUBTYPE_COLORS[key]}">'
            f'<div class="sub-name">{esc(g["name"])}</div><div class="sub-tag">{esc(g["tag"])}</div>'
            f'<div class="sub-copy">{esc(g["desc"])}</div>'
            f'<div class="sub-meta"><b>Penanda khas:</b> {esc(g["markers"])}<br>'
            f'<b>Pendekatan terapi umum:</b> {esc(g["care"])}</div></div>'
        )
    return (
        '<p class="guide-intro"><b>Apa itu PAM50?</b> PAM50 adalah panel 50 gen yang membaca pola ekspresi gen tumor '
        'untuk menggolongkannya ke lima subtipe intrinsik. Setiap subtipe berbeda dalam kecepatan tumbuh, ketergantungan pada hormon, '
        'dan respons terhadap terapi, sehingga membantu memperkirakan perjalanan penyakit.</p>'
        '<div class="glossary"><b>Istilah:</b> ER dan PR = reseptor hormon estrogen dan progesteron. '
        'HER2 = protein pemacu pertumbuhan sel. Proliferasi = seberapa cepat sel kanker membelah. '
        'Triple-negative = ER−, PR−, dan HER2− sekaligus.</div>'
        f'<div class="sub-grid">{cards}</div>'
        '<div class="guide-source">Contoh proporsi dari satu kohort besar (1.319 perempuan di AS): Luminal A 53,1%, Luminal B 20,5%, '
        'HER2-enriched 13,0%, basal-like 9,8%, normal-like 3,6%. Proporsi berbeda antar populasi dan metode. '
        'Informasi umum, bukan nasihat medis; keputusan terapi selalu ditentukan oleh dokter.</div>'
    )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 2px 14px">
      <div style="font-size:2rem">🧬</div>
      <div style="font-family:Manrope,sans-serif;font-size:1.25rem;font-weight:800;line-height:1.25;margin-top:4px">
        PAM50<br>Classifier
      </div>
      <div style="color:#8C6577;font-size:.82rem;margin-top:7px">TCGA-BRCA · Research Prototype</div>
    </div><hr>
    <div class="side-label">ALUR PENGGUNAAN</div>
    <div class="side-step"><span>1</span>Unggah data ekspresi gen</div>
    <div class="side-step"><span>2</span>Periksa format sampel</div>
    <div class="side-step"><span>3</span>Jalankan klasifikasi</div>
    <div class="side-step"><span>4</span>Tinjau hasil & Evaluasi Model</div>
    <hr>
    <div class="side-label">SUBTIPE PAM50</div>
    <div class="side-sub"><i style="background:#C8433F"></i>Basal-like</div>
    <div class="side-sub"><i style="background:#B04A9C"></i>HER2-enriched</div>
    <div class="side-sub"><i style="background:#1F5F99"></i>Luminal A</div>
    <div class="side-sub"><i style="background:#4DA3D9"></i>Luminal B</div>
    <div class="side-sub"><i style="background:#3E9B6B"></i>Normal-like</div>
    <hr>
    <div style="color:#8C6577;font-size:.78rem;line-height:1.6;margin-top:6px">
      Prototipe machine learning untuk penelitian dan eksplorasi data. Bukan pengganti pemeriksaan klinis.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="kicker">Bioinformatics · Machine Learning</div>
  <div class="hero-title">Klasifikasi Subtipe Molekuler Kanker Payudara</div>
  <p class="hero-copy">
    Analisis ekspresi gen TCGA-BRCA menggunakan model
    <b>Tuned Multi-Layer Perceptron (MLP)</b> untuk memperkirakan
    subtipe molekuler PAM50 pada setiap sampel, mengevaluasi kinerja klasifikasi, dan menginterpretasi gen fitur.
  </p>
  <div class="tag-row">
    <span class="tag">TCGA-BRCA</span><span class="tag">2.000 gen terpilih</span>
    <span class="tag">5 kelas PAM50</span><span class="tag">Tuned MLP</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="notice">'
    '<div class="notice-head"><span class="notice-icon">!</span>Baca dulu sebelum menggunakan hasil</div>'
    '<div class="notice-grid">'
    '<div class="notice-item"><b>Prototipe penelitian</b><span>Dibuat untuk eksplorasi data dan pembelajaran. '
    'Ini bukan alat diagnosis medis dan tidak menggantikan pemeriksaan dokter atau patologi.</span></div>'
    '<div class="notice-item"><b>Normal-like bukan berarti jaringan normal</b><span>Normal-like hanyalah nama salah satu subtipe molekuler PAM50. '
    'Hasil ini tidak membuktikan bahwa jaringan sampel normal; sering kali profil ini muncul karena sampel hanya sedikit mengandung sel tumor.</span></div>'
    '<div class="notice-item"><b>Confidence bukan kepastian</b><span>Angka confidence adalah probabilitas tertinggi menurut model. '
    'Nilai tinggi berarti model yakin, tetapi tidak menjamin hasilnya benar secara klinis.</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

with st.expander("📚 Pelajari: kanker payudara dan subtipe PAM50"):
    edu_cancer, edu_subtype = st.tabs(["Kanker payudara", "Subtipe PAM50"])
    with edu_cancer:
        st.markdown(CANCER_INFO_HTML, unsafe_allow_html=True)
    with edu_subtype:
        st.markdown(subtype_guide_html(), unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ------------------------------------------------------------
required_files = [MODEL_PATH, SCALER_PATH, GENES_PATH, LABELS_PATH]
missing_files = [str(p) for p in required_files if not p.exists()]
if missing_files:
    st.error("File model belum lengkap. Pastikan folder `results/final_model` berada di direktori yang sama dengan `app.py`.")
    st.code("\n".join(missing_files))
    st.stop()

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path, scaler_path, genes_path, labels_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(genes_path, "r", encoding="utf-8") as f:
        gene_names = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    label_mapping = {str(k): str(v) for k, v in raw_mapping.items()}
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    return model, scaler, gene_names, label_mapping, class_names

try:
    with loading_panel("Memuat model dan preprocessing", "Pemuatan pertama dapat memakan waktu beberapa detik."):
        model, scaler, gene_names, label_mapping, class_names = load_artifacts(
            str(MODEL_PATH), str(SCALER_PATH), str(GENES_PATH), str(LABELS_PATH)
        )
except Exception as error:
    st.error("Model atau file preprocessing gagal dimuat.")
    st.exception(error)
    st.stop()

status_col, genes_col, classes_col = st.columns([1.4, 1, 1])
with status_col:
    st.markdown("""
    <div class="card"><div class="mini-label">Status sistem</div>
    <div style="margin-top:9px"><span class="pill"><span class="live"></span>Model siap digunakan</span></div>
    <div class="card-copy" style="margin-top:8px">Model dan preprocessing berhasil dimuat.</div></div>
    """, unsafe_allow_html=True)
with genes_col:
    st.markdown(f"""
    <div class="card"><div class="mini-label">Fitur model</div>
    <div class="mini-value">{len(gene_names):,}</div><div class="card-copy">Gen terpilih</div></div>
    """, unsafe_allow_html=True)
with classes_col:
    st.markdown(f"""
    <div class="card"><div class="mini-label">Keluaran model</div>
    <div class="mini-value">{len(class_names)}</div><div class="card-copy">Subtipe PAM50</div></div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# UPLOAD AND PREPROCESS INPUT
# ------------------------------------------------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">LANGKAH 01</div>', unsafe_allow_html=True)
st.header("Unggah data ekspresi gen")
st.markdown(
    '<div class="section-note">Unggah CSV atau TSV berisi nama gen dan nilai ekspresi numerik. '
    'Orientasi data akan dideteksi otomatis. Sertakan file label jika ingin mengevaluasi model.</div>', unsafe_allow_html=True
)

upload_col, format_col = st.columns([1.45, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Pilih file CSV / TSV (Ekspresi Gen)", type=["csv", "tsv", "txt"],
        help="File perlu memuat ID sampel atau nama gen sebagai indeks."
    )
    with st.expander("Apakah Anda memiliki Label Aktual? (Opsional untuk Evaluasi Kinerja)"):
        label_file = st.file_uploader(
            "Pilih file label (CSV / TSV)", type=["csv", "tsv", "txt"],
            help="Kolom pertama = ID sampel; kolom berikutnya = label subtipe (PAM50)."
        )

with format_col:
    st.markdown("""
    <div class="card"><div class="card-title">Format yang didukung</div>
    <div class="card-copy">
      <b>Format A</b><br>Baris = sampel · Kolom = gen<br><br>
      <b>Format B</b><br>Baris = gen · Kolom = sampel<br><br>
      Nama gen harus sesuai dengan fitur model saat pelatihan.
    </div></div>
    """, unsafe_allow_html=True)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None
if "prediction_file_name" not in st.session_state:
    st.session_state.prediction_file_name = None

if uploaded_file is not None:
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.upload_signature != signature:
        st.session_state.prediction_results = None
        st.session_state.upload_signature = signature

    try:
        uploaded_file.seek(0)
        with loading_panel("Membaca data ekspresi gen", f"Memuat {uploaded_file.name}..."):
            raw_df = pd.read_csv(uploaded_file, sep=None, engine="python", index_col=0, encoding="utf-8-sig")
            raw_df.columns = raw_df.columns.astype(str).str.replace("\ufeff", "", regex=False)
            raw_df.index = raw_df.index.astype(str).str.replace("\ufeff", "", regex=False)
            raw_df.index.name = None
            raw_df.columns.name = None

        required_genes = set(map(str, gene_names))
        columns, index_values = set(raw_df.columns), set(raw_df.index)

        if required_genes.issubset(columns):
            expression_df = raw_df.copy()
            orientation = "Sampel pada baris, gen pada kolom"
        elif required_genes.issubset(index_values):
            expression_df = raw_df.T.copy()
            expression_df.columns = expression_df.columns.astype(str)
            orientation = "Gen pada baris, sampel pada kolom"
        else:
            found_columns = len(required_genes.intersection(columns))
            found_index = len(required_genes.intersection(index_values))
            st.error("File belum memuat seluruh gen yang dibutuhkan model.")
            c1, c2 = st.columns(2)
            c1.metric("Gen cocok pada kolom", f"{found_columns}/{len(gene_names)}")
            c2.metric("Gen cocok pada indeks", f"{found_index}/{len(gene_names)}")
            st.info("Periksa nama gen, delimiter file, dan kolom pertama (ID sampel atau nama gen).")
            st.stop()

        expression_df.columns = expression_df.columns.astype(str)
        missing_genes = [g for g in gene_names if g not in expression_df.columns]
        if missing_genes:
            st.error(f"Ada {len(missing_genes)} gen yang tidak tersedia.")
            st.code(", ".join(missing_genes[:20]))
            st.stop()

        X_input = expression_df.loc[:, gene_names].apply(pd.to_numeric, errors="coerce")
        X_array = X_input.to_numpy(dtype=np.float32)

        if len(X_input) == 0:
            st.error("Tidak ada sampel yang dapat diprediksi.")
            st.stop()
        if not np.isfinite(X_array).all():
            st.error("Data mengandung nilai kosong, nonnumerik, atau tak hingga pada gen terpilih. Bersihkan data terlebih dahulu.")
            st.stop()

        st.markdown('<div class="eyebrow">LANGKAH 02</div>', unsafe_allow_html=True)
        st.header("Periksa data sebelum prediksi")
        st.markdown('<div class="section-note">Pastikan jumlah sampel dan orientasi data sesuai sebelum menjalankan model.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sampel dalam file", f"{len(X_input):,} sampel")
        c1.caption("Jumlah baris data (satu baris = satu sampel) yang akan diklasifikasikan.")
        c2.metric("Gen yang dipakai model", f"{len(gene_names):,} gen")
        c2.caption("Gen diambil dari file sesuai fitur saat pelatihan. Gen lain di file diabaikan.")
        c3.metric("Subtipe keluaran", f"{len(class_names)} subtipe")
        c3.caption("Jumlah kemungkinan hasil prediksi: subtipe PAM50 yang dikenali model.")
        st.caption(f"**File:** {uploaded_file.name}  ·  **Orientasi:** {orientation}")

        with st.expander("Lihat cuplikan data ekspresi"):
            st.dataframe(X_input.head(8), use_container_width=True)

        run_col, info_col = st.columns([1, 1.5])
        with run_col:
            run_prediction = st.button("▶  Jalankan prediksi", type="primary", use_container_width=True)
        with info_col:
            st.caption("Model dan scaler tersimpan akan digunakan otomatis. Data tidak perlu dinormalisasi ulang secara manual.")

        if run_prediction:
            try:
                with loading_panel("Model sedang menganalisis ekspresi gen", "Menormalisasi data dan menghitung probabilitas tiap subtipe..."):
                    X_scaled = scaler.transform(X_array).astype(np.float32)
                    probabilities = model.predict(X_scaled, verbose=0)
                    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
                        raise ValueError("Dimensi keluaran model tidak sesuai dengan label_mapping.json.")

                    predicted_ids = np.argmax(probabilities, axis=1)
                    results = pd.DataFrame({
                        "Sample_ID": X_input.index.astype(str),
                        "Predicted_Subtype": [label_mapping[str(i)] for i in predicted_ids],
                        "Confidence": probabilities.max(axis=1),
                    })
                    for i, subtype in enumerate(class_names):
                        results[f"Probability_{subtype}"] = probabilities[:, i]

                    # Memasukkan Ground Truth jika pengguna menyertakan Label Uji
                    if label_file is not None:
                        label_file.seek(0)
                        label_df = pd.read_csv(label_file, index_col=0, engine="python")
                        if len(label_df.columns) > 0:
                            y_raw = label_df.iloc[:, 0].reindex(X_input.index)
                            y_harmonized = harmonize_labels(y_raw, class_names, label_mapping)
                            results["True_Subtype"] = y_harmonized.values

                    st.session_state.prediction_results = results
                    st.session_state.prediction_file_name = uploaded_file.name
            except Exception as error:
                st.error("Prediksi gagal dijalankan.")
                st.exception(error)

    except Exception as error:
        st.error("File tidak dapat dibaca atau formatnya tidak sesuai. Pastikan file berisi tabel ekspresi gen dengan header yang benar.")
        st.exception(error)

# ------------------------------------------------------------
# RESULTS & EVALUATION
# ------------------------------------------------------------
results = st.session_state.prediction_results
if results is not None and not results.empty:
    st.markdown("---")
    st.markdown('<div class="eyebrow">LANGKAH 03</div>', unsafe_allow_html=True)
    st.header("Hasil klasifikasi")

    n_samples = len(results)
    avg_confidence = results["Confidence"].mean()
    n_subtypes = results["Predicted_Subtype"].nunique()
    subtype_counts = results["Predicted_Subtype"].value_counts()
    top_subtype, top_count = subtype_counts.idxmax(), int(subtype_counts.max())

    st.markdown(
        '<div class="section-note">Ringkasan prediksi model untuk sampel pada file yang diunggah. '
        f'Subtipe terbanyak: <b>{esc(top_subtype)}</b> ({top_count:,} dari {n_samples:,} sampel).</div>',
        unsafe_allow_html=True,
    )

    detected_names = ", ".join(subtype_counts.index)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sampel diprediksi", f"{n_samples:,} sampel")
    c1.caption("Jumlah sampel yang selesai diklasifikasikan oleh model.")
    c2.metric("Subtipe yang muncul", f"{n_subtypes} dari {len(class_names)}")
    c2.caption(f"{n_subtypes} dari {len(class_names)} subtipe PAM50 muncul pada hasil prediksi: {detected_names}.")
    c3.metric("Rata-rata confidence", f"{avg_confidence:.1%}")
    c3.caption("Rata-rata probabilitas tertinggi model untuk tiap sampel. Makin tinggi, makin yakin model, tetapi bukan kepastian klinis.")

    tab_summary, tab_detail, tab_export = st.tabs(["📊 Ringkasan", "🔬 Detail sampel", "⬇️ Unduh hasil"])

    with tab_summary:
        chart_col, table_col = st.columns([1.15, 1])
        counts = results["Predicted_Subtype"].value_counts().reindex(class_names, fill_value=0)
        with chart_col:
            st.markdown("#### Distribusi subtipe")
            st.markdown(distribution_html(counts, n_samples), unsafe_allow_html=True)
            st.caption("Jumlah sampel pada setiap kategori hasil prediksi.")
        with table_col:
            st.markdown("#### Komposisi prediksi")
            distribution = counts.rename_axis("Subtipe").reset_index(name="Jumlah sampel")
            distribution["Persentase"] = distribution["Jumlah sampel"] / n_samples * 100
            dist_styler = style_map(distribution.style, subtype_cell, ["Subtipe"])
            st.dataframe(
                dist_styler,
                column_config={
                    "Jumlah sampel": st.column_config.NumberColumn("Jumlah sampel", format="%d"),
                    "Persentase": st.column_config.ProgressColumn(
                        "Persentase", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Tabel seluruh prediksi")
        display_df = results.copy()
        display_df["Confidence"] = display_df["Confidence"] * 100
        for s in class_names:
            display_df[f"Probability_{s}"] = display_df[f"Probability_{s}"] * 100
        
        cols_to_style = ["Predicted_Subtype"]
        if "True_Subtype" in display_df.columns:
            cols_to_style.append("True_Subtype")
            
        styler = style_map(display_df.style, subtype_cell, cols_to_style)
        st.dataframe(
            styler,
            column_config={
                "Sample_ID": st.column_config.TextColumn("ID sampel"),
                "Predicted_Subtype": st.column_config.TextColumn("Subtipe prediksi"),
                "True_Subtype": st.column_config.TextColumn("Label Aktual"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f%%"
                ),
                **{
                    f"Probability_{s}": st.column_config.ProgressColumn(
                        s, min_value=0, max_value=100, format="%.1f%%"
                    )
                    for s in class_names
                },
            },
            use_container_width=True,
            hide_index=True,
        )

    with tab_detail:
        st.markdown("#### Probabilitas untuk satu sampel")
        selected_sample = st.selectbox("Pilih ID sampel", results["Sample_ID"].astype(str).tolist())
        row = results.loc[results["Sample_ID"].astype(str) == selected_sample].iloc[0]

        probs = np.array([float(row[f"Probability_{s}"]) for s in class_names])
        order = np.argsort(probs)[::-1]
        pred_name = str(row["Predicted_Subtype"])
        pred_color = subtype_color(pred_name)
        confidence = float(row["Confidence"])
        runner_name = class_names[order[1]]
        runner_prob = probs[order[1]]
        margin = probs[order[0]] - runner_prob

        if confidence < 0.6 or margin < 0.2:
            conf_note = '<div class="flag">Selisih dengan kandidat kedua kecil. Tinjau hasil ini dengan lebih hati-hati.</div>'
        else:
            conf_note = '<div class="ok">Prediksi jauh di atas kandidat lain.</div>'

        d1, d2 = st.columns(2)
        with d1:
            true_label_html = f'<div class="card-copy" style="margin-top:4px">Label Aktual: <b>{esc(row["True_Subtype"])}</b></div>' if "True_Subtype" in row and pd.notna(row["True_Subtype"]) else ""
            st.markdown(
                f'<div class="card result-card" style="border-left:5px solid {pred_color}">'
                f'<div class="mini-label">Prediksi model</div>'
                f'<div class="subtype-name">{esc(pred_name)}</div>'
                f'<div class="card-copy">ID sampel: {esc(selected_sample)}</div>'
                f'{true_label_html}'
                f'<div class="runner">Kandidat kedua: <b>{esc(runner_name)}</b> ({runner_prob:.1%})</div></div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="card result-card ring-card">{confidence_ring_html(confidence, pred_color)}'
                f'<div><div class="mini-label">Confidence model</div>'
                f'<div class="card-copy" style="margin-top:6px">Probabilitas tertinggi dari model, bukan kepastian klinis.</div>'
                f'{conf_note}</div></div>',
                unsafe_allow_html=True,
            )

        prob_df = pd.DataFrame({
            "Subtipe": class_names,
            "Probabilitas": probs.tolist()
        }).set_index("Subtipe")
        st.markdown("#### Probabilitas per subtipe")
        st.markdown(probability_html(class_names, probs, pred_name), unsafe_allow_html=True)
        with st.expander("Lihat tabel probabilitas"):
            st.dataframe(
                prob_df.rename(columns={"Probabilitas": "Nilai probabilitas"}).style.format({"Nilai probabilitas": "{:.2%}"}),
                use_container_width=True,
            )

    with tab_export:
        st.markdown("#### Ekspor hasil klasifikasi")
        st.write("Unduh tabel prediksi beserta confidence dan probabilitas untuk setiap subtipe.")
        csv_data = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️  Unduh hasil prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_pam50.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # SECTION: EVALUASI KINERJA MODEL
    # --------------------------------------------------------
    if "True_Subtype" in results.columns and results["True_Subtype"].notna().any():
        eval_df = results.dropna(subset=["True_Subtype"])
        if len(eval_df) > 0:
            st.markdown("---")
            st.markdown('<div class="eyebrow">EVALUASI</div>', unsafe_allow_html=True)
            st.header("🎯 Evaluasi Kinerja Model")
            st.markdown('<div class="section-note">Mengukur akurasi, F1, dan confusion matrix berdasarkan perbandingan prediksi model dengan file label aktual yang telah Anda unggah.</div>', unsafe_allow_html=True)
            
            metrics = compute_metrics(eval_df["True_Subtype"], eval_df["Predicted_Subtype"], class_names)
            
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Akurasi", f"{metrics['accuracy']:.1%}", help="Persentase prediksi benar secara keseluruhan.")
            e2.metric("F1 Macro", f"{metrics['f1_macro']:.1%}", help="Rata-rata harmonik presisi dan recall secara merata.")
            e3.metric("MCC", f"{metrics['mcc']:.3f}", help="Matthews Correlation Coefficient (-1 sampai 1).")
            e4.metric("Sampel Dievaluasi", f"{metrics['n']:,}", help=f"Terdapat {metrics['n_wrong']} prediksi salah.")
            
            eval_t1, eval_t2 = st.tabs(["📊 Metrik per Kelas", "🧮 Confusion Matrix"])
            with eval_t1:
                styler = style_map(metrics["per_class"].style, subtype_cell, ["Subtipe"])
                st.dataframe(styler, hide_index=True, use_container_width=True, column_config={
                    "Presisi": st.column_config.NumberColumn(format="%.1f%%"),
                    "Recall": st.column_config.NumberColumn(format="%.1f%%"),
                    "F1": st.column_config.NumberColumn(format="%.1f%%")
                })
            with eval_t2:
                cm_df = pd.DataFrame(metrics["cm"], index=class_names, columns=class_names)
                st.dataframe(cm_df.style.background_gradient(cmap="RdPu", axis=None), use_container_width=True)

else:
    st.markdown("""
    <div class="card empty" style="margin-top:20px;text-align:center;padding:34px 28px">
      <div style="font-size:2rem;margin-bottom:8px">🧪</div>
      <div class="card-title" style="font-size:1.05rem">Hasil prediksi akan tampil di sini</div>
      <div class="card-copy">Unggah data ekspresi gen, periksa formatnya, lalu klik <b>Jalankan prediksi</b>.</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# SECTION: INTERPRETASI FITUR GEN
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="eyebrow">INTERPRETASI</div>', unsafe_allow_html=True)
st.header("🔍 Interpretasi Fitur Gen")
st.markdown('<div class="section-note">Melihat gen-gen mana saja yang paling kuat memengaruhi keputusan model secara global dengan menginspeksi bobot pada <i>dense layer</i> awal. Gen dengan rata-rata bobot tinggi adalah ekstraktor fitur terpenting.</div>', unsafe_allow_html=True)

importance = first_layer_importance(model, len(gene_names))
if importance is not None:
    df_imp = pd.DataFrame({
        "Gen": gene_names, 
        "Bobot": importance
    })
    df_imp["Gen PAM50"] = np.where([is_pam50(g) for g in df_imp["Gen"]], "Ya", "Tidak")
    df_imp = df_imp.sort_values("Bobot", ascending=False).head(40)
    
    # Configure Altair graph for Streamlit Dark/Light themes nicely
    chart = (
        alt.Chart(df_imp).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Bobot:Q", title="Rata-rata Absolut Bobot"),
            y=alt.Y("Gen:N", sort="-x", title=None),
            color=alt.Color(
                "Gen PAM50:N", 
                scale=alt.Scale(domain=["Ya", "Tidak"], range=["#C23E73", "#F2A7C2"])
            ),
            tooltip=["Gen", alt.Tooltip("Bobot:Q", format=".4f"), "Gen PAM50"]
        )
        .properties(height=max(300, 20 * len(df_imp)))
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#6A4A5C", titleColor="#6A4A5C", gridColor="#F4E1E9")
        .configure_legend(labelColor="#6A4A5C", titleColor="#6A4A5C")
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Bobot lapisan pertama tidak dapat diekstrak dari arsitektur model keras saat ini.")

with st.expander("Klik untuk melihat Arsitektur Lapisan Model (Tuned MLP)"):
    layers_df = model_layers_table(model)
    st.dataframe(layers_df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
  TCGA-BRCA PAM50 Classifier · Prototipe penelitian berbasis machine learning<br>
  Hasil prediksi bersifat eksperimental dan tidak ditujukan untuk diagnosis atau keputusan klinis.
</div>
""", unsafe_allow_html=True)