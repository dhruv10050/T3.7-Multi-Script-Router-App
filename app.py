"""T3.7 — Multi-Script Indic Handwriting Recognition
Streamlit app: draw a character or upload an image → get script + character prediction.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from inference import load_pipeline, predict, preprocess
from label_maps import get_label

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indic Script Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_INFO = {
    "devanagari": {"flag": "🔵", "label": "Devanagari", "classes": 46,  "scripts": "Hindi / Nepali / Sanskrit"},
    "tamil":      {"flag": "🟢", "label": "Tamil",       "classes": 156, "scripts": "Tamil"},
    "bengali":    {"flag": "🟠", "label": "Bengali",     "classes": 84,  "scripts": "Bengali / Assamese"},
    "telugu":     {"flag": "🟣", "label": "Telugu",      "classes": 6,   "scripts": "Telugu (vowels)"},
}
BADGE_COLOUR = {
    "devanagari": "#1565C0",
    "tamil":      "#2E7D32",
    "bengali":    "#E65100",
    "telugu":     "#6A1B9A",
}
CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints"))

# ── Cached model loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def _load_models():
    return load_pipeline(CKPT_DIR)


def _canvas_to_pil(canvas_data: np.ndarray) -> Image.Image | None:
    """Convert the RGBA numpy array from drawable-canvas to a grayscale PIL image."""
    if canvas_data is None:
        return None
    if canvas_data.shape[2] == 4:
        # alpha channel: 0 = transparent (background), >0 = drawn pixels
        alpha = canvas_data[:, :, 3]
        if alpha.max() == 0:
            return None  # nothing drawn
        # render onto white background
        bg = np.full(canvas_data.shape[:2], 255, dtype=np.uint8)
        mask = alpha > 0
        # invert drawn strokes: canvas draws with dark ink → characters are dark
        bg[mask] = (255 - canvas_data[:, :, :3][mask].mean(axis=1)).astype(np.uint8)
    else:
        bg = 255 - canvas_data.mean(axis=2).astype(np.uint8)
    return Image.fromarray(bg, mode="L")


def _auto_crop(img: Image.Image, margin: int = 10) -> Image.Image:
    """Tightly crop to the drawn ink region, then add a small margin."""
    arr = np.array(img)
    # pixels below 200 are considered ink (images are white-background)
    ink = arr < 200
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    if not rows.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0, r1 = max(0, r0 - margin), min(arr.shape[0], r1 + margin)
    c0, c1 = max(0, c0 - margin), min(arr.shape[1], c1 + margin)
    return img.crop((c0, r0, c1, r1))


def _render_predictions(result: dict, pipeline_scripts: list[str]) -> None:
    """Render the script badge and top-5 character predictions."""
    script = result["script_name"]
    if script not in pipeline_scripts:
        st.warning(
            f"Router predicted **{script.title()}** but that classifier is not loaded. "
            f"Available: {', '.join(pipeline_scripts)}.",
        )
        return

    info = SCRIPT_INFO.get(script, {"label": script.title(), "flag": "⬜", "classes": "?", "scripts": "—"})
    colour = BADGE_COLOUR.get(script, "#555")
    conf = result["script_conf"] * 100

    # ── Script badge ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='display:inline-block; background:{colour}; color:white;
                    padding:6px 16px; border-radius:20px; font-size:1.05rem;
                    font-weight:600; margin-bottom:12px;'>
            {info["flag"]}&nbsp;&nbsp;{info["label"]}
            &nbsp;<span style='opacity:0.8;font-size:0.9rem;'>({conf:.1f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Used for: {info['scripts']}  ·  {info['classes']} character classes")

    st.markdown("**Top-5 predictions**")

    top_ids   = result["top_ids"]
    top_probs = result["top_probs"]

    for rank, (idx, prob) in enumerate(zip(top_ids, top_probs), start=1):
        lbl = get_label(script, idx)
        pct = prob * 100
        char_html = (
            f"<span style='font-size:2rem; font-weight:700; "
            f"font-family:Noto Sans,serif;'>{lbl['char']}</span>"
        )
        bar_w = max(4, int(pct))
        col_char, col_info = st.columns([1, 5])
        with col_char:
            st.markdown(char_html, unsafe_allow_html=True)
        with col_info:
            st.markdown(
                f"<div style='margin-top:6px;'>"
                f"<b>{lbl['name']}</b> "
                f"<span style='color:#888; font-size:0.8rem;'>{lbl['unicode']}</span>"
                f"</div>"
                f"<div style='background:#e0e0e0; border-radius:4px; height:10px; margin-top:4px;'>"
                f"<div style='width:{bar_w}%; background:{colour}; height:10px; border-radius:4px;'></div>"
                f"</div>"
                f"<div style='font-size:0.85rem; color:#555;'>{pct:.1f}%</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='margin:4px 0; border-color:#eee;'>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✍️ Indic Script Recognition")
    st.markdown(
        """
        A two-stage CNN that first identifies the **script** then predicts the **character**.

        **Supported scripts**
        | | Script | Classes |
        |---|---|---|
        | 🔵 | Devanagari | 46 |
        | 🟢 | Tamil | 156 |
        | 🟠 | Bengali | 84 |
        | 🟣 | Telugu (vowels) | 6 |
        """
    )
    st.divider()

    # Model status
    pipeline, loaded_scripts = _load_models()
    if pipeline is None:
        st.error(
            "**No model checkpoints found.**\n\n"
            f"Expected directory: `{CKPT_DIR}/`\n\n"
            "Place `router_best.pth` and at least one "
            "`<script>_best.pth` file there, or set the `HF_MODEL_REPO` "
            "environment variable to auto-download from HuggingFace Hub."
        )
    else:
        st.success(f"Models loaded  ✓")
        for s in loaded_scripts:
            info = SCRIPT_INFO.get(s, {})
            st.markdown(f"- {info.get('flag','·')} {info.get('label', s)}")

    st.divider()
    st.caption("Architecture: ScriptRouter → ScriptCNN\nTrained from scratch · no transfer learning")


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("## Handwritten Indic Character Recognition")
st.markdown(
    "Draw a character in the canvas below **or** upload an image, then press **Predict**."
)

tab_draw, tab_upload = st.tabs(["✏️  Draw", "🖼️  Upload"])

# ────────────────────────────────────────────────────────────────────────────
# Tab 1 — Draw
# ────────────────────────────────────────────────────────────────────────────
with tab_draw:
    try:
        from streamlit_drawable_canvas import st_canvas
        _canvas_available = True
    except ImportError:
        _canvas_available = False
        st.error(
            "`streamlit-drawable-canvas` is not installed. "
            "Run `pip install streamlit-drawable-canvas` and restart."
        )

    if _canvas_available:
        col_canvas, col_result = st.columns([1, 1], gap="large")

        with col_canvas:
            stroke_width = st.slider("Pen width", min_value=8, max_value=40, value=18, step=2)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=300,
                height=300,
                drawing_mode="freedraw",
                key="draw_canvas",
            )
            predict_btn = st.button("Predict ▶", type="primary", key="btn_draw")

        with col_result:
            st.markdown("### Prediction")
            if predict_btn:
                if pipeline is None:
                    st.error("No model loaded — see sidebar.")
                else:
                    pil_img = _canvas_to_pil(canvas_result.image_data)
                    if pil_img is None:
                        st.warning("Canvas is empty — draw a character first.")
                    else:
                        pil_img = _auto_crop(pil_img)
                        with st.spinner("Running inference…"):
                            tensor = preprocess(pil_img)
                            result = predict(pipeline, tensor)
                        _render_predictions(result, loaded_scripts)
                        # show the preprocessed thumbnail
                        st.markdown("---")
                        st.caption("Input seen by model (64 × 64 grayscale):")
                        st.image(pil_img.resize((128, 128)), width=128)
            else:
                st.markdown(
                    "<div style='color:#aaa; margin-top:40px; text-align:center;'>"
                    "Draw something on the left, then press <b>Predict ▶</b></div>",
                    unsafe_allow_html=True,
                )

# ────────────────────────────────────────────────────────────────────────────
# Tab 2 — Upload
# ────────────────────────────────────────────────────────────────────────────
with tab_upload:
    col_up, col_up_result = st.columns([1, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Upload a character image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Any size; the app resizes to 64 × 64 grayscale.",
        )
        if uploaded:
            pil_up = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(pil_up, caption="Uploaded image", use_container_width=False, width=200)
        predict_up_btn = st.button("Predict ▶", type="primary", key="btn_upload")

    with col_up_result:
        st.markdown("### Prediction")
        if predict_up_btn:
            if pipeline is None:
                st.error("No model loaded — see sidebar.")
            elif not uploaded:
                st.warning("Please upload an image first.")
            else:
                with st.spinner("Running inference…"):
                    gray = pil_up.convert("L")
                    # invert if image is dark-background / light-ink
                    arr = np.array(gray)
                    if arr.mean() < 128:
                        gray = ImageOps.invert(gray)
                    gray = _auto_crop(gray)
                    tensor = preprocess(gray)
                    result = predict(pipeline, tensor)
                _render_predictions(result, loaded_scripts)
                st.markdown("---")
                st.caption("Input seen by model (64 × 64 grayscale):")
                st.image(gray.resize((128, 128)), width=128)
        else:
            st.markdown(
                "<div style='color:#aaa; margin-top:40px; text-align:center;'>"
                "Upload an image on the left, then press <b>Predict ▶</b></div>",
                unsafe_allow_html=True,
            )
