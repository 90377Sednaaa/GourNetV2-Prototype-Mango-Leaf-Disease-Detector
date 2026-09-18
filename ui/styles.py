"""Botanical & research design tokens and CSS styles for Streamlit."""
import streamlit as st

def get_botanical_css() -> str:
    return """
    <style>
    /* Design Tokens & Theme Variables */
    :root {
        --botanical-deep: #0F5132;
        --botanical-forest: #166534;
        --botanical-mint: #10B981;
        --botanical-bright-mint: #34D399;
        --botanical-light-mint: #E6F4EA;
        --slate-900: #0F172A;
        --slate-700: #334155;
        --slate-100: #F1F5F9;
        --amber-500: #F59E0B;
        --amber-600: #D97706;
        --amber-50: #FFFBEB;
        --coral-600: #DC2626;
        --coral-50: #FEF2F2;

        /* Universal Theme-Adaptive Defaults (Driven by Streamlit's native --text-color) */
        --card-border: rgba(16, 185, 129, 0.25);
        --card-shadow-hover: rgba(16, 185, 129, 0.12);
        --hero-title-color: #10B981;
        --header-subtext-color: var(--text-color, #E2E8F0);

        --status-pill-border: rgba(16, 185, 129, 0.45);
        --status-pill-bg: rgba(16, 185, 129, 0.15);
        --status-pill-color: var(--text-color, #E2E8F0);

        --status-pill-warning-border: rgba(245, 158, 11, 0.5);
        --status-pill-warning-bg: rgba(245, 158, 11, 0.15);
        --status-pill-warning-color: #FCD34D;

        --metric-badge-bg: rgba(255, 255, 255, 0.08);
        --metric-badge-color: var(--text-color, inherit);

        --consensus-bg: rgba(16, 185, 129, 0.12);
        --consensus-border: rgba(16, 185, 129, 0.45);
        --consensus-border-left: #10B981;
        --consensus-color: var(--text-color, #ECFDF5);

        --divergence-bg: rgba(245, 158, 11, 0.12);
        --divergence-border: rgba(245, 158, 11, 0.45);
        --divergence-border-left: #F59E0B;
        --divergence-color: var(--text-color, #FFFBEB);
    }

    /* Dark Mode OS preference Support */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-border: rgba(52, 211, 153, 0.3);
            --card-shadow-hover: rgba(52, 211, 153, 0.15);
            --hero-title-color: #34D399;
            --header-subtext-color: #CBD5E1;

            --status-pill-border: rgba(52, 211, 153, 0.4);
            --status-pill-bg: rgba(16, 185, 129, 0.2);
            --status-pill-color: #ECFDF5;

            --status-pill-warning-border: rgba(245, 158, 11, 0.45);
            --status-pill-warning-bg: rgba(245, 158, 11, 0.2);
            --status-pill-warning-color: #FCD34D;

            --metric-badge-bg: rgba(255, 255, 255, 0.08);
            --consensus-color: #ECFDF5;
            --divergence-color: #FFFBEB;
        }
    }

    /* Streamlit Dark Theme Attribute Support */
    [data-theme="dark"], [data-testid="stAppViewContainer"].stApp--dark, .stApp[data-theme="dark"] {
        --card-border: rgba(52, 211, 153, 0.3);
        --card-shadow-hover: rgba(52, 211, 153, 0.15);
        --hero-title-color: #34D399;
        --header-subtext-color: #CBD5E1;

        --status-pill-border: rgba(52, 211, 153, 0.4);
        --status-pill-bg: rgba(16, 185, 129, 0.2);
        --status-pill-color: #ECFDF5;

        --status-pill-warning-border: rgba(245, 158, 11, 0.45);
        --status-pill-warning-bg: rgba(245, 158, 11, 0.2);
        --status-pill-warning-color: #FCD34D;

        --metric-badge-bg: rgba(255, 255, 255, 0.08);
        --consensus-color: #ECFDF5;
        --divergence-color: #FFFBEB;
    }

    /* =========================================================
       Card, Box & Column Height Alignment System
       ========================================================= */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="column"] > div[data-testid="stVerticalBlock"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }

    [data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* Grad-CAM & Card Image Full-Width Alignment System */
    [data-testid="column"] [data-testid="stFullScreenFrame"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    [data-testid="column"] [data-testid="stFullScreenFrame"] > div {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }

    [data-testid="column"] [data-testid="stImage"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="column"] [data-testid="stImage"] > div,
    [data-testid="column"] [data-testid="stImageContainer"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }

    [data-testid="column"] [data-testid="stImage"] img,
    [data-testid="column"] [data-testid="stImageContainer"] img {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 420px !important;
        object-fit: contain !important;
        border-radius: 8px !important;
        margin: 0 auto !important;
        display: block !important;
    }

    [data-testid="column"] [data-testid="stImage"] [data-testid="stCaptionContainer"],
    [data-testid="column"] [data-testid="stImageCaption"] {
        text-align: center !important;
        width: 100% !important;
        margin-top: 0.35rem !important;
    }

    /* Card Containers */
    .gournet-card {
        background: transparent;
        border: 1px solid var(--card-border, rgba(16, 185, 129, 0.25));
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .gournet-card:hover {
        box-shadow: 0 6px 16px var(--card-shadow-hover, rgba(16, 185, 129, 0.12));
    }

    /* Model Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid var(--status-pill-border, rgba(16, 185, 129, 0.45));
        background: var(--status-pill-bg, rgba(16, 185, 129, 0.15));
        color: var(--status-pill-color, var(--text-color, #E2E8F0)) !important;
    }

    .status-pill.warning {
        border-color: var(--status-pill-warning-border, rgba(245, 158, 11, 0.5));
        background: var(--status-pill-warning-bg, rgba(245, 158, 11, 0.15));
        color: var(--status-pill-warning-color, #FCD34D) !important;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        background: var(--metric-badge-bg, rgba(255, 255, 255, 0.08));
        color: var(--metric-badge-color, var(--text-color, inherit));
    }

    /* High-Contrast Consensus and Divergence Banners */
    .consensus-banner {
        background: var(--consensus-bg, rgba(16, 185, 129, 0.12));
        border: 1px solid var(--consensus-border, rgba(16, 185, 129, 0.45));
        border-left: 5px solid var(--consensus-border-left, #10B981);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--consensus-color, var(--text-color, #ECFDF5));
    }

    .consensus-banner .banner-title {
        color: #10B981 !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .consensus-banner .banner-detail {
        color: var(--consensus-color, var(--text-color, #E2E8F0));
        font-size: 0.95rem;
        line-height: 1.4;
    }

    .consensus-banner strong {
        color: var(--text-color, #FFFFFF) !important;
        font-weight: 700;
    }

    .divergence-banner {
        background: var(--divergence-bg, rgba(245, 158, 11, 0.12));
        border: 1px solid var(--divergence-border, rgba(245, 158, 11, 0.45));
        border-left: 5px solid var(--divergence-border-left, #F59E0B);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--divergence-color, var(--text-color, #FFFBEB));
    }

    .divergence-banner .banner-title {
        color: #F59E0B !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .divergence-banner .banner-detail {
        color: var(--divergence-color, var(--text-color, #E2E8F0));
        font-size: 0.95rem;
        line-height: 1.4;
    }

    .divergence-banner strong {
        color: var(--text-color, #FFFFFF) !important;
        font-weight: 700;
    }

    /* Custom Header Subtitle */
    .header-subtext {
        color: var(--header-subtext-color, var(--text-color, #CBD5E1));
        font-size: 1.05rem;
        margin-top: -0.25rem;
        margin-bottom: 1.25rem;
        opacity: 0.9;
    }
    </style>
    """

def apply_theme_styles():
    st.markdown(get_botanical_css(), unsafe_allow_html=True)
