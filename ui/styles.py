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
        --botanical-light-mint: #E6F4EA;
        --slate-900: #0F172A;
        --slate-700: #334155;
        --slate-100: #F1F5F9;
        --amber-600: #D97706;
        --amber-50: #FFFBEB;
        --coral-600: #DC2626;
        --coral-50: #FEF2F2;

        /* Light Mode Defaults */
        --card-border: rgba(15, 81, 50, 0.18);
        --card-shadow-hover: rgba(15, 81, 50, 0.08);
        --hero-title-color: #0F5132;
        --header-subtext-color: #475569;

        --status-pill-border: rgba(15, 81, 50, 0.2);
        --status-pill-bg: rgba(16, 185, 129, 0.1);
        --status-pill-color: #0F5132;

        --status-pill-warning-border: rgba(217, 119, 6, 0.3);
        --status-pill-warning-bg: rgba(217, 119, 6, 0.1);
        --status-pill-warning-color: #B45309;

        --metric-badge-bg: rgba(15, 23, 42, 0.04);
        --metric-badge-color: inherit;

        --consensus-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 81, 50, 0.08) 100%);
        --consensus-border: #10B981;
        --consensus-border-left: #0F5132;
        --consensus-color: #064E3B;

        --divergence-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%);
        --divergence-border: #F59E0B;
        --divergence-border-left: #D97706;
        --divergence-color: #78350F;
    }

    /* Dark Mode Support: OS preference */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-border: rgba(52, 211, 153, 0.25);
            --card-shadow-hover: rgba(52, 211, 153, 0.15);
            --hero-title-color: #34D399;
            --header-subtext-color: #94A3B8;

            --status-pill-border: rgba(52, 211, 153, 0.35);
            --status-pill-bg: rgba(16, 185, 129, 0.2);
            --status-pill-color: #6EE7B7;

            --status-pill-warning-border: rgba(245, 158, 11, 0.45);
            --status-pill-warning-bg: rgba(245, 158, 11, 0.2);
            --status-pill-warning-color: #FCD34D;

            --metric-badge-bg: rgba(255, 255, 255, 0.08);

            --consensus-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(15, 81, 50, 0.3) 100%);
            --consensus-border: rgba(52, 211, 153, 0.6);
            --consensus-border-left: #34D399;
            --consensus-color: #ECFDF5;

            --divergence-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(217, 119, 6, 0.3) 100%);
            --divergence-border: rgba(251, 191, 36, 0.6);
            --divergence-border-left: #F59E0B;
            --divergence-color: #FFFBEB;
        }
    }

    /* Streamlit Dark Theme Attribute Support */
    [data-theme="dark"], [data-testid="stAppViewContainer"].stApp--dark, .stApp[data-theme="dark"] {
        --card-border: rgba(52, 211, 153, 0.25);
        --card-shadow-hover: rgba(52, 211, 153, 0.15);
        --hero-title-color: #34D399;
        --header-subtext-color: #94A3B8;

        --status-pill-border: rgba(52, 211, 153, 0.35);
        --status-pill-bg: rgba(16, 185, 129, 0.2);
        --status-pill-color: #6EE7B7;

        --status-pill-warning-border: rgba(245, 158, 11, 0.45);
        --status-pill-warning-bg: rgba(245, 158, 11, 0.2);
        --status-pill-warning-color: #FCD34D;

        --metric-badge-bg: rgba(255, 255, 255, 0.08);

        --consensus-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(15, 81, 50, 0.3) 100%);
        --consensus-border: rgba(52, 211, 153, 0.6);
        --consensus-border-left: #34D399;
        --consensus-color: #ECFDF5;

        --divergence-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(217, 119, 6, 0.3) 100%);
        --divergence-border: rgba(251, 191, 36, 0.6);
        --divergence-border-left: #F59E0B;
        --divergence-color: #FFFBEB;
    }

    /* Modern Card Container */
    .gournet-card {
        background: transparent;
        border: 1px solid var(--card-border, rgba(15, 81, 50, 0.18));
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .gournet-card:hover {
        box-shadow: 0 6px 16px var(--card-shadow-hover, rgba(15, 81, 50, 0.08));
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
        border: 1px solid var(--status-pill-border, rgba(15, 81, 50, 0.2));
        background: var(--status-pill-bg, rgba(16, 185, 129, 0.1));
        color: var(--status-pill-color, #0F5132);
    }

    .status-pill.warning {
        border-color: var(--status-pill-warning-border, rgba(217, 119, 6, 0.3));
        background: var(--status-pill-warning-bg, rgba(217, 119, 6, 0.1));
        color: var(--status-pill-warning-color, #B45309);
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        background: var(--metric-badge-bg, rgba(15, 23, 42, 0.04));
        color: var(--text-color, inherit);
    }

    /* Consensus and Divergence Banners */
    .consensus-banner {
        background: var(--consensus-bg, linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 81, 50, 0.08) 100%));
        border: 1px solid var(--consensus-border, #10B981);
        border-left: 5px solid var(--consensus-border-left, #0F5132);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--consensus-color, #064E3B);
    }

    .divergence-banner {
        background: var(--divergence-bg, linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%));
        border: 1px solid var(--divergence-border, #F59E0B);
        border-left: 5px solid var(--divergence-border-left, #D97706);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--divergence-color, #78350F);
    }

    /* Custom Header Subtitle */
    .header-subtext {
        color: var(--header-subtext-color, var(--text-color, #475569));
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.25rem;
    }
    </style>
    """

def apply_theme_styles():
    st.markdown(get_botanical_css(), unsafe_allow_html=True)
