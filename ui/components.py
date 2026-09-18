"""Reusable modern UI components for GourNet Mango Leaf Disease Detector."""

import pathlib
import numpy as np
import pandas as pd
import streamlit as st


def get_sample_images(samples_dir: pathlib.Path) -> list[pathlib.Path]:
    """Scans samples_dir for valid images (.jpg, .jpeg, .png) and returns them sorted by name."""
    if not isinstance(samples_dir, pathlib.Path):
        samples_dir = pathlib.Path(samples_dir)
    if not samples_dir.exists() or not samples_dir.is_dir():
        return []

    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    return sorted(images, key=lambda p: p.name)


def format_consensus_data(
    pred_a: str,
    pred_b: str,
    conf_a: float,
    conf_b: float,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
) -> dict:
    """Formats consensus or divergence details between two model predictions."""
    agreed = pred_a == pred_b
    if agreed:
        title = f"Consensus Reached: Both models predict {pred_a}"
        detail = f"<strong>{short_a}</strong> ({conf_a * 100:.1f}%) vs <strong>{short_b}</strong> ({conf_b * 100:.1f}%)"
        css_class = "consensus-banner"
    else:
        title = "Models Disagree — Further Diagnostic Review Recommended"
        detail = (
            f"<strong>{short_a}</strong>: {pred_a} ({conf_a * 100:.1f}%) vs "
            f"<strong>{short_b}</strong>: {pred_b} ({conf_b * 100:.1f}%)"
        )
        css_class = "divergence-banner"

    return {
        "agreed": agreed,
        "title": title,
        "detail": detail,
        "css_class": css_class,
    }


def render_hero_header():
    """Renders modern botanical branding header with research subtext."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; color: #0F5132; display: flex; align-items: center; gap: 0.5rem;">
                🌿 GourNet Mango Leaf Disease Detector
            </h1>
            <p class="header-subtext" style="margin-top: 0.35rem;">
                Comparative Botanical Deep Learning Prototype & Benchmarking Dashboard
                <span class="status-pill" style="margin-left: 0.5rem;">Baseline vs v2</span>
                <span class="status-pill">8 Disease Classes</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status_ribbon(
    static_infos: dict,
    trained_flags: dict,
    variant: str,
    model_registry: dict,
):
    """Renders side-by-side model overview cards (Baseline vs v2)."""
    keys = list(model_registry.keys())
    if not keys:
        return

    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        meta = model_registry[key]
        short_name = meta.get("short", key)
        notes = meta.get("notes", "")
        params, size_mb = static_infos.get(key, (-1, 0.0))
        is_trained = trained_flags.get(key, False)

        with col:
            with st.container(border=True):
                header_html = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 700; font-size: 1.1rem; color: #0F172A;">{short_name}</span>
                    <span class="status-pill{' warning' if not is_trained else ''}">
                        {'● Trained' if is_trained else '⚠ Fallback / Missing'}
                    </span>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)
                if notes:
                    st.caption(notes)

                c1, c2, c3 = st.columns(3)
                param_str = f"{params:,}" if params >= 0 else "N/A"
                if "v2" in short_name.lower():
                    c1.metric("Parameters", param_str, delta="~7x smaller (-85.6%)", delta_color="off")
                else:
                    c1.metric("Parameters", param_str)

                c2.metric("Size", f"{size_mb:.1f} MB" if size_mb > 0 else "N/A")
                c3.metric("Weights", variant)


def render_consensus_banner(
    consensus_data: dict | str,
    pred_b: str | None = None,
    conf_a: float | None = None,
    conf_b: float | None = None,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
):
    """Renders the consensus/divergence banner using appropriate botanical styles."""
    if isinstance(consensus_data, dict):
        data = consensus_data
    else:
        data = format_consensus_data(
            pred_a=str(consensus_data),
            pred_b=str(pred_b or ""),
            conf_a=float(conf_a if conf_a is not None else 0.0),
            conf_b=float(conf_b if conf_b is not None else 0.0),
            short_a=short_a,
            short_b=short_b,
        )

    css_class = data.get("css_class", "consensus-banner")
    title = data.get("title", "")
    detail = data.get("detail", "")
    icon = "✅" if data.get("agreed", False) else "⚠️"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.25rem;">
                {icon} {title}
            </div>
            <div style="font-size: 0.95rem; opacity: 0.95;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparative_ribbon(
    a_ms: float,
    b_ms: float,
    a_conf: float,
    b_conf: float,
    a_margin: float,
    b_margin: float,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
):
    """Renders 3 metric comparison cards: Faster model, Higher confidence, Larger decision margin."""
    faster = short_a if a_ms <= b_ms else short_b
    ms_diff = abs(a_ms - b_ms)

    higher_conf = short_a if a_conf >= b_conf else short_b
    conf_diff = abs(a_conf - b_conf) * 100.0

    larger_margin = short_a if a_margin >= b_margin else short_b
    margin_diff = abs(a_margin - b_margin) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "⚡ Faster Model",
        faster,
        delta=f"{ms_diff:.1f} ms advantage" if ms_diff > 0 else "Identical latency",
        delta_color="off",
        border=True,
    )
    c2.metric(
        "🎯 Higher Confidence",
        higher_conf,
        delta=f"{conf_diff:.1f} pp higher" if conf_diff > 0 else "Identical confidence",
        delta_color="off",
        border=True,
    )
    c3.metric(
        "📏 Larger Decision Margin",
        larger_margin,
        delta=f"{margin_diff:.1f} pp margin" if margin_diff > 0 else "Identical margin",
        delta_color="off",
        border=True,
    )


def render_model_card(
    short_name: str,
    full_name: str,
    res: dict,
    top_k: int,
    class_names: list,
):
    """Renders a single model's prediction card with top class, metrics, and top-k breakdown."""
    if not res or res.get("probs") is None:
        with st.container(border=True):
            st.subheader(short_name)
            if full_name and full_name != short_name:
                st.caption(full_name)
            st.info("No prediction data available.")
        return

    probs = res.get("probs")
    ms = res.get("ms", 0.0)
    stats = res.get("stats")

    if stats is None and probs is not None:
        order = np.argsort(probs)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        conf1 = float(probs[top1])
        conf2 = float(probs[top2])
        clipped = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(clipped * np.log(clipped)))
        stats = {
            "top_idx": top1,
            "conf": conf1,
            "margin": conf1 - conf2,
            "entropy": entropy,
            "order": order,
        }

    top_idx = stats["top_idx"]
    conf = stats["conf"]
    margin = stats["margin"]
    entropy = stats["entropy"]
    order = stats["order"]

    top_class_name = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"

    with st.container(border=True):
        st.subheader(short_name)
        if full_name and full_name != short_name:
            st.caption(full_name)

        st.metric(
            label="Predicted Class",
            value=top_class_name,
            delta=f"{conf * 100.0:.1f}% confidence",
            delta_color="off",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{ms:.1f} ms")
        m2.metric("Margin", f"{margin * 100.0:.1f} pp")
        m3.metric("Entropy", f"{entropy:.2f}")

        st.markdown("**Top Predictions**")
        topk_indices = order[:top_k]
        topk_rows = [
            {
                "Class": class_names[int(i)] if int(i) < len(class_names) else f"Class {i}",
                "Probability": float(probs[int(i)]),
            }
            for i in topk_indices
        ]
        topk_df = pd.DataFrame(topk_rows)
        st.dataframe(
            topk_df,
            hide_index=True,
            column_config={
                "Class": st.column_config.TextColumn("Class"),
                "Probability": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            },
        )
