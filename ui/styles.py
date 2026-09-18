"""Botanical & research design tokens and CSS styles for Streamlit."""
import streamlit as st

def get_botanical_css() -> str:
    return """
    <style>
    /* Design Tokens */
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
    }

    /* Modern Card Container */
    .gournet-card {
        background: transparent;
        border: 1px solid rgba(15, 81, 50, 0.18);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .gournet-card:hover {
        box-shadow: 0 6px 16px rgba(15, 81, 50, 0.08);
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
        border: 1px solid rgba(15, 81, 50, 0.2);
        background: rgba(16, 185, 129, 0.1);
        color: #0F5132;
    }

    .status-pill.warning {
        border-color: rgba(217, 119, 6, 0.3);
        background: rgba(217, 119, 6, 0.1);
        color: #B45309;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        background: rgba(15, 23, 42, 0.04);
    }

    /* Consensus and Divergence Banners */
    .consensus-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 81, 50, 0.08) 100%);
        border: 1px solid #10B981;
        border-left: 5px solid #0F5132;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #064E3B;
    }

    .divergence-banner {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%);
        border: 1px solid #F59E0B;
        border-left: 5px solid #D97706;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #78350F;
    }

    /* Custom Header Subtitle */
    .header-subtext {
        color: #475569;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.25rem;
    }
    </style>
    """

def apply_theme_styles():
    st.markdown(get_botanical_css(), unsafe_allow_html=True)
