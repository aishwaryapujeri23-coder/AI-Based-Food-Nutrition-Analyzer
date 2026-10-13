"""
app.py
------
AI-Based Food Nutrition Analyzer using RAG and LLM.

Run with:
    streamlit run app.py

Five features in one dashboard:
1. "Nutrition Lookup"           -> live nutrition facts (scaled to any
   quantity you enter) from the USDA FoodData Central API. This is the
   ONLY feature that calls an external API (USDA).
2. "Ask AI Nutrition Assistant" -> a RAG chatbot that answers free-form
   nutrition questions using your own PDF/TXT knowledge base in data/ +
   a language model hosted on Groq's cloud API.
3. "Personalized Diet Planner"  -> collects your profile (age, gender,
   height, weight, activity, goal) and asks the SAME RAG assistant to
   generate guidance, grounded in your data/ knowledge base.
4. "Meal Planner"                -> collects meal count + diet preference
   and asks the RAG assistant to build a day's meal plan, grounded in
   your data/ knowledge base.
5. "Disease Nutrition Advisor"  -> pick a condition (+ optional custom
   question) and asks the RAG assistant for condition-specific guidance,
   grounded in your data/ knowledge base.

Everything except the Nutrition Lookup tab is powered by Retrieval-
Augmented Generation over the files in data/ -- there is no hardcoded
diet-plan or meal-plan logic in this file.

VISUAL DESIGN NOTE: background "food imagery" is built from hand-crafted
inline SVG line-art (leaf / citrus / chili / carrot doodles) tiled across
the hero and page background, rather than hotlinked stock photography.
That keeps the app fully self-contained -- nothing breaks if you're
offline or a photo host goes down.
"""

import streamlit as st
from groq import APIError
from nutrition_api import search_food, get_food_details, NutritionAPIError


def md_html(content: str):
    """
    Render raw HTML/CSS via st.markdown.

    Every line is stripped of leading/trailing whitespace and the whole
    block is flattened onto a single line before being handed to
    Streamlit. Markdown's CommonMark parser treats 4+ leading spaces on a
    line as an indented code block, and can also terminate/misparse an
    HTML block on certain blank-line or nesting patterns -- flattening to
    one line removes every newline+indentation combination that could
    trigger either rule, regardless of how the original Python string was
    indented. CSS and HTML are both whitespace-insensitive between tokens,
    so this is a purely cosmetic transform: nothing about how the page
    looks or behaves changes.
    """
    flat = " ".join(line.strip() for line in content.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)



st.set_page_config(
    page_title="NutriSense AI | Food Nutrition Analyzer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CUSTOM STYLING
# Palette  : paper #F5F7EE · ink #202B1E · basil #2F5233 · turmeric #E2932A ·
#            clay #C1502E · basil-deep (hero) #1D3320
# Type     : Fraunces (display/serif, editorial food-label warmth) +
#            Manrope (body/UI) + JetBrains Mono (nutrition-fact numerals)
# Signature: metric cards styled like an FDA-style nutrition-facts panel;
#            big hoverable section headers per tab; a floating food-doodle
#            watermark instead of a generic stock photo.
# ---------------------------------------------------------------------------
md_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;0,9..144,900;1,9..144,500;1,9..144,600&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    :root {
        --paper: #F5F7EE;
        --paper-deep: #ECF0DF;
        --ink: #202B1E;
        --basil: #2F5233;
        --basil-deep: #1D3320;
        --basil-mid: #3C6B41;
        --turmeric: #E2932A;
        --turmeric-deep: #C97A1A;
        --clay: #C1502E;
        --line: #E1E4D4;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Whole-app background: soft accent gradients + a real food photograph,
       heavily blended with the paper tone so text stays fully readable */
    .stApp {
        background:
            radial-gradient(circle at 8% 10%, rgba(47,82,51,0.06) 0, transparent 38%),
            radial-gradient(circle at 94% 6%, rgba(226,147,42,0.08) 0, transparent 32%),
            radial-gradient(circle at 90% 92%, rgba(193,80,46,0.06) 0, transparent 36%),
            radial-gradient(circle at 4% 88%, rgba(47,82,51,0.06) 0, transparent 34%),
            linear-gradient(145deg, #F8FAF3 0%, #F2F5EA 52%, #F8F7EF 100%);
        background-attachment: fixed;
    }
    .bg-doodle {
        position: fixed;
        z-index: 0;
        opacity: 0.05;
        pointer-events: none;
    }
    .bg-doodle.bd-1 { top: 4%; left: 2%; width: 160px; animation: floatDriftSlow 14s ease-in-out infinite; }
    .bg-doodle.bd-2 { bottom: 6%; right: 3%; width: 190px; animation: floatDrift 16s ease-in-out infinite; }
    .bg-doodle.bd-3 { top: 45%; right: -40px; width: 130px; animation: floatDriftSlow 12s ease-in-out infinite 2s; }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1180px;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, .nx-card h3, .nx-card h4 {
        font-family: 'Fraunces', serif;
    }

    ::selection { background: var(--turmeric); color: white; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--paper-deep); }
    ::-webkit-scrollbar-thumb { background: #b9c4a4; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--basil-mid); }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes floatDrift {
        0%   { transform: translate(0,0) rotate(0deg); }
        50%  { transform: translate(-8px,-14px) rotate(6deg); }
        100% { transform: translate(0,0) rotate(0deg); }
    }
    @keyframes floatDriftSlow {
        0%   { transform: translate(0,0) rotate(0deg); }
        50%  { transform: translate(10px,10px) rotate(-8deg); }
        100% { transform: translate(0,0) rotate(0deg); }
    }
    @keyframes underlineGrow {
        from { width: 0; }
        to   { width: 60%; }
    }
    @keyframes quoteCycle {
        0%   { opacity: 0; transform: translateY(6px); }
        6%   { opacity: 1; transform: translateY(0); }
        24%  { opacity: 1; transform: translateY(0); }
        30%  { opacity: 0; transform: translateY(-6px); }
        100% { opacity: 0; transform: translateY(-6px); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
    }

    /* ---------------- Hero ---------------- */
    .hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(ellipse at 82% 16%, rgba(226,147,42,0.24), transparent 31%),
            linear-gradient(132deg, #172C1D 0%, #26472E 56%, #356440 100%);
        border-radius: 24px;
        padding: 3rem 2.8rem 2.6rem 2.8rem;
        margin-bottom: 2.4rem;
        box-shadow: 0 20px 45px rgba(29, 51, 32, 0.3);
        animation: fadeInUp 0.55s ease;
        isolation: isolate;
    }
    .hero .food-doodle {
        position: absolute;
        opacity: 0.16;
        z-index: 0;
        pointer-events: none;
    }
    .hero .food-doodle.leaf-1 { top: -18px; right: 6%; width: 130px; animation: floatDrift 7s ease-in-out infinite; }
    .hero .food-doodle.citrus-1 { bottom: -22px; right: 20%; width: 95px; animation: floatDriftSlow 9s ease-in-out infinite; }
    .hero .food-doodle.chili-1 { top: 14%; right: 1%; width: 75px; animation: floatDrift 8s ease-in-out infinite 1s; }
    .hero .food-doodle.carrot-1 { bottom: 4%; left: 2%; width: 85px; opacity: 0.13; animation: floatDriftSlow 10s ease-in-out infinite; }
    .hero .food-doodle.leaf-2 { top: 40%; left: -20px; width: 90px; opacity: 0.1; animation: floatDrift 11s ease-in-out infinite 0.5s; }
    .hero .food-doodle.berry-1 { top: 8%; left: 30%; width: 60px; opacity: 0.12; animation: floatDriftSlow 13s ease-in-out infinite 1.5s; }

    .hero .eyebrow {
        position: relative; z-index: 1;
        display: inline-flex; align-items: center; gap: 0.4rem;
        color: var(--turmeric);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    .hero h1 {
        position: relative; z-index: 1;
        font-family: 'Fraunces', serif;
        font-optical-sizing: auto;
        background: linear-gradient(100deg, #FBFBF3 30%, #F3D9A8 60%, #FBFBF3 90%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: clamp(2.3rem, 4.6vw, 3.3rem);
        font-weight: 700;
        font-style: italic;
        margin: 0 0 0.9rem 0;
        letter-spacing: -0.5px;
        line-height: 1.08;
    }
    .hero .quote-carousel {
        position: relative; z-index: 1;
        height: 1.7rem;
        margin: 0 0 1.1rem 0;
        max-width: 640px;
    }
    .hero .quote-carousel span {
        position: absolute;
        left: 0; top: 0;
        color: var(--turmeric);
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        font-size: 1.05rem;
        opacity: 0;
        animation: quoteCycle 16s ease-in-out infinite;
        white-space: nowrap;
    }
    .hero .quote-carousel span:nth-child(1) { animation-delay: 0s; }
    .hero .quote-carousel span:nth-child(2) { animation-delay: 4s; }
    .hero .quote-carousel span:nth-child(3) { animation-delay: 8s; }
    .hero .quote-carousel span:nth-child(4) { animation-delay: 12s; }
    .hero p.hero-desc {
        position: relative; z-index: 1;
        color: rgba(251,251,243,0.86);
        font-size: 1.02rem;
        max-width: 640px;
        margin: 0;
        line-height: 1.55;
    }
    .hero .badges {
        position: relative; z-index: 1;
        margin-top: 1.2rem;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(251,251,243,0.1);
        border: 1px solid rgba(251,251,243,0.22);
        color: #FBFBF3;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .hero .badge:hover {
        background: var(--turmeric);
        border-color: var(--turmeric);
        color: var(--basil-deep);
        transform: translateY(-2px) scale(1.05);
    }

    /* ---------------- Big section headers (per-tab) ---------------- */
    .section-hero {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 3px 12px rgba(32,43,30,0.05);
        animation: fadeInUp 0.4s ease;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        cursor: default;
    }
    .section-hero:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(32,43,30,0.1);
        border-color: var(--turmeric);
    }
    .section-hero-icon {
        flex-shrink: 0;
        font-size: 2.1rem;
        width: 62px;
        height: 62px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #EEF3E0, #E4EDD0);
        border-radius: 16px;
        transition: transform 0.25s cubic-bezier(.34,1.56,.64,1);
    }
    .section-hero:hover .section-hero-icon {
        transform: rotate(-8deg) scale(1.1);
        background: linear-gradient(135deg, var(--turmeric), var(--turmeric-deep));
    }
    .section-hero-text { min-width: 0; }
    .section-hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: clamp(1.5rem, 2.6vw, 2.05rem);
        color: var(--ink);
        line-height: 1.15;
        margin-bottom: 0.25rem;
        transition: color 0.2s ease, letter-spacing 0.2s ease;
    }
    .section-hero:hover .section-hero-title {
        color: var(--basil-deep);
        letter-spacing: 0.2px;
    }
    .section-hero .section-caption {
        margin: 0;
    }

    /* ---------------- Tabs ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: var(--paper-deep);
        padding: 8px;
        border-radius: 14px;
        flex-wrap: wrap;
        border: 1px solid var(--line);
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.1rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--basil-mid);
        transition: all 0.2s ease;
        position: relative;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(47,82,51,0.08);
        color: var(--basil-deep);
        transform: translateY(-1px) scale(1.02);
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--basil-deep) !important;
        box-shadow: 0 3px 10px rgba(29,51,32,0.12);
    }
    .stTabs [aria-selected="true"]::after {
        content: "";
        position: absolute;
        left: 50%;
        bottom: 6px;
        transform: translateX(-50%);
        height: 3px;
        width: 60%;
        background: var(--turmeric);
        border-radius: 999px;
        animation: underlineGrow 0.3s ease;
    }
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInUp 0.35s ease;
    }

    /* ---------------- Generic card ---------------- */
    .nx-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 0.7rem 0 1rem 0;
        box-shadow: 0 2px 10px rgba(32,43,30,0.05);
        animation: fadeInUp 0.4s ease;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .nx-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(32,43,30,0.09);
        border-color: #cddab4;
    }
    .nx-card h3, .nx-card h4 {
        margin: 0 0 0.4rem 0;
        color: var(--ink);
    }

    /* Card container for nutrient results */
    .food-title-card {
        background: white;
        border: 1px solid var(--line);
        border-left: 4px solid var(--turmeric);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 1.3rem 0;
        box-shadow: 0 2px 10px rgba(32,43,30,0.05);
        animation: fadeInUp 0.4s ease;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .food-title-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(32,43,30,0.08);
    }
    .food-title-card h3 {
        margin: 0;
        color: var(--ink);
    }
    .food-title-card .qty-tag {
        display: inline-block;
        background: #EEF3E0;
        color: var(--basil);
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border: 1px solid #d9e4c2;
    }

    /* ---------------- Metric cards: "nutrition facts panel" signature --------- */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        box-shadow: 0 2px 8px rgba(32,43,30,0.05);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        animation: fadeInUp 0.4s ease;
        border-top: 3px solid var(--ink);
        border-bottom: 3px solid var(--ink);
        border-left: 1px solid var(--line);
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 20px rgba(32,43,30,0.1);
        border-top-color: var(--turmeric);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 700;
        color: #5b6753;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Manrope', sans-serif;
    }
    div[data-testid="stMetricValue"] {
        color: var(--basil-deep);
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    /* ---------------- Buttons ---------------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--turmeric), var(--turmeric-deep));
        color: var(--basil-deep);
        border: none;
        border-radius: 10px;
        padding: 0.58rem 1.7rem;
        font-weight: 700;
        font-family: 'Manrope', sans-serif;
        box-shadow: 0 4px 12px rgba(226,147,42,0.32);
        transition: all 0.15s cubic-bezier(.34,1.56,.64,1);
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 18px rgba(226,147,42,0.42);
        color: var(--basil-deep);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.97);
    }

    /* Inputs get a gentle focus glow */
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        box-shadow: 0 0 0 3px rgba(226,147,42,0.25) !important;
        border-color: var(--turmeric) !important;
    }

    /* ---------------- Chat bubbles ---------------- */
    .chat-bubble-user {
        background: #EEF3E0;
        border-left: 3px solid var(--basil);
        padding: 0.8rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.3s ease;
        transition: transform 0.15s ease;
    }
    .chat-bubble-user:hover { transform: translateX(2px); }
    .chat-bubble-assistant {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 3px solid var(--turmeric);
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 0.3rem;
        box-shadow: 0 2px 6px rgba(32,43,30,0.04);
        animation: fadeIn 0.4s ease;
        white-space: pre-wrap;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .chat-bubble-assistant:hover {
        transform: translateX(2px);
        box-shadow: 0 4px 12px rgba(32,43,30,0.07);
    }
    .chat-sources {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #8a9480;
        margin-top: 0.35rem;
    }
    .section-caption {
        color: #5b6753;
        font-size: 0.93rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* ---------------- Dashboard polish ---------------- */
    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        color: var(--ink);
    }
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important;
    }
    [data-testid="stAlert"] p {
        color: var(--ink) !important;
    }
    .hero {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 28px;
        padding: clamp(2rem, 5vw, 3.6rem);
        margin-bottom: 1.4rem;
        box-shadow: 0 24px 55px rgba(29,51,32,0.2);
    }
    .hero .hero-desc { max-width: 700px; }
    .workspace-heading {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.7rem 0 0.85rem;
    }
    .workspace-heading h2 {
        color: var(--ink);
        font-family: 'Fraunces', serif;
        font-size: 1.55rem;
        margin: 0;
    }
    .workspace-heading p {
        color: #697562;
        font-size: 0.88rem;
        margin: 0.25rem 0 0;
    }
    .workspace-label {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--basil);
        background: rgba(47,82,51,0.08);
        border: 1px solid rgba(47,82,51,0.13);
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font: 700 0.72rem 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }
    .workspace-label::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #4A9B61;
        box-shadow: 0 0 0 3px rgba(74,155,97,0.13);
    }
    .workspace-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0 0 1.7rem;
    }
    .workspace-card {
        min-height: 142px;
        padding: 1.1rem 1.2rem;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(225,228,212,0.95);
        border-radius: 17px;
        box-shadow: 0 5px 18px rgba(32,43,30,0.045);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .workspace-card:hover {
        transform: translateY(-3px);
        border-color: #c8d5b6;
        box-shadow: 0 12px 25px rgba(32,43,30,0.09);
    }
    .workspace-card-icon {
        display: inline-flex;
        width: 37px;
        height: 37px;
        align-items: center;
        justify-content: center;
        background: #EEF3E0;
        border-radius: 12px;
        font-size: 1.15rem;
        margin-bottom: 0.6rem;
    }
    .workspace-card strong {
        display: block;
        color: var(--ink);
        font-size: 0.91rem;
        margin-bottom: 0.2rem;
    }
    .workspace-card span:last-child {
        display: block;
        color: #6A7564;
        font-size: 0.77rem;
        line-height: 1.45;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(232,237,221,0.88);
        border-color: rgba(216,223,203,0.9);
        padding: 0.4rem;
        border-radius: 15px;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding: 0 0.9rem;
        font-size: 0.86rem;
    }
    .section-hero {
        border-radius: 20px;
        padding: 1.25rem 1.45rem;
        box-shadow: 0 6px 20px rgba(32,43,30,0.055);
    }
    .nx-card, .food-title-card {
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(32,43,30,0.05);
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-top: 3px solid var(--basil-mid);
        border-bottom: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
    }
    .stTextInput input, .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div {
        border-radius: 11px;
    }
    .stButton > button, .stFormSubmitButton > button {
        min-height: 43px;
        border-radius: 11px;
    }
    @media (max-width: 900px) {
        .workspace-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { border-radius: 21px; padding: 1.8rem 1.35rem; }
        .hero .quote-carousel { height: 2.8rem; }
        .hero .quote-carousel span { white-space: normal; }
        .workspace-heading { align-items: start; flex-direction: column; }
        .workspace-grid { gap: 0.55rem; }
        .workspace-card { min-height: 135px; padding: 0.9rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
        .stTabs [data-baseweb="tab"] { padding: 0 0.65rem; font-size: 0.78rem; }
    }
    </style>
    """,
)

# ---------------------------------------------------------------------------
# Whole-page background doodles (fixed, very low opacity, sit behind content)
# ---------------------------------------------------------------------------
md_html(
    """
    <svg class="bg-doodle bd-1" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 5 C80 15 90 45 70 75 C55 95 25 95 15 75 C5 55 15 20 50 5 Z"
              fill="none" stroke="#2F5233" stroke-width="2"/>
    </svg>
    <svg class="bg-doodle bd-2" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 30 L35 130 L65 130 Z" fill="none" stroke="#C1502E" stroke-width="2"/>
        <path d="M50 30 C45 15 35 8 25 5 M50 30 C52 12 60 6 70 4" fill="none" stroke="#C1502E" stroke-width="1.6"/>
    </svg>
    <svg class="bg-doodle bd-3" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" fill="none" stroke="#E2932A" stroke-width="2"/>
        <g stroke="#E2932A" stroke-width="1.4">
            <line x1="50" y1="12" x2="50" y2="88"/>
            <line x1="16" y1="30" x2="84" y2="70"/>
            <line x1="16" y1="70" x2="84" y2="30"/>
        </g>
    </svg>
    """,
)

# ---------------------------------------------------------------------------
# HERO HEADER -- innovative title, rotating quote carousel, animated food
# doodles (hand-drawn inline SVG line art standing in for a background image)
# ---------------------------------------------------------------------------
md_html(
    """
    <div class="hero">
        <svg class="food-doodle leaf-1" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <path d="M50 5 C80 15 90 45 70 75 C55 95 25 95 15 75 C5 55 15 20 50 5 Z"
                  fill="none" stroke="#FBFBF3" stroke-width="2.5"/>
            <path d="M50 10 C50 40 50 65 20 78" fill="none" stroke="#FBFBF3" stroke-width="2"/>
        </svg>
        <svg class="food-doodle citrus-1" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="42" fill="none" stroke="#FBFBF3" stroke-width="2.5"/>
            <g stroke="#FBFBF3" stroke-width="1.6">
                <line x1="50" y1="10" x2="50" y2="90"/>
                <line x1="14" y1="28" x2="86" y2="72"/>
                <line x1="14" y1="72" x2="86" y2="28"/>
            </g>
        </svg>
        <svg class="food-doodle chili-1" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
            <path d="M30 10 C20 5 15 15 20 25 C40 35 55 60 50 95 C48 115 30 130 15 120"
                  fill="none" stroke="#FBFBF3" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <svg class="food-doodle carrot-1" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
            <path d="M50 30 L35 130 L65 130 Z" fill="none" stroke="#FBFBF3" stroke-width="2.5"/>
            <path d="M50 30 C45 15 35 8 25 5 M50 30 C52 12 60 6 70 4 M50 30 C50 12 50 4 50 2"
                  fill="none" stroke="#FBFBF3" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <svg class="food-doodle leaf-2" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <path d="M50 5 C80 15 90 45 70 75 C55 95 25 95 15 75 C5 55 15 20 50 5 Z"
                  fill="none" stroke="#FBFBF3" stroke-width="2"/>
        </svg>
        <svg class="food-doodle berry-1" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="55" r="30" fill="none" stroke="#FBFBF3" stroke-width="2"/>
            <path d="M50 25 C45 15 55 10 60 15" fill="none" stroke="#FBFBF3" stroke-width="2" stroke-linecap="round"/>
        </svg>

        <div class="eyebrow">🥗 &nbsp;AI Nutrition Dashboard</div>
        <h1>Decode Every Bite.</h1>
        <div class="quote-carousel">
            <span>"Food is information your body reads three times a day."</span>
            <span>"You can't out-train a diet you don't understand."</span>
            <span>"Small, informed choices compound into lasting health."</span>
            <span>"The best diet plan is the one built around real data — yours."</span>
        </div>
        <p class="hero-desc">Instant USDA nutrition facts, plus an AI assistant for personalized diet
        guidance, meal planning, and condition-specific advice — grounded entirely in your own
        knowledge base.</p>
        <div class="badges">
            <span class="badge">🐍 Python</span>
            <span class="badge">🎈 Streamlit</span>
            <span class="badge">🔗 LangChain</span>
            <span class="badge">📚 FAISS RAG</span>
            <span class="badge">🤖 Qwen · Groq</span>
            <span class="badge">🌐 USDA API</span>
        </div>
    </div>
    """,
)

# ---------------------------------------------------------------------------
# Dashboard overview -- summarizes the capabilities without implying that
# external services or the local knowledge base are currently connected.
# ---------------------------------------------------------------------------
md_html(
    """
    <div class="workspace-heading">
        <div>
            <h2>Your nutrition workspace</h2>
            <p>Evidence-aware tools for everyday food and wellness decisions.</p>
        </div>
        <div class="workspace-label">5 TOOLS · ONE WORKSPACE</div>
    </div>
    <div class="workspace-grid">
        <div class="workspace-card">
            <span class="workspace-card-icon">🔎</span>
            <strong>Food intelligence</strong>
            <span>Explore nutrition details from USDA FoodData Central.</span>
        </div>
        <div class="workspace-card">
            <span class="workspace-card-icon">🧠</span>
            <strong>Knowledge-grounded AI</strong>
            <span>Ask questions with answers grounded in your documents.</span>
        </div>
        <div class="workspace-card">
            <span class="workspace-card-icon">🥗</span>
            <strong>Personalized planning</strong>
            <span>Build diet and meal guidance around your preferences.</span>
        </div>
        <div class="workspace-card">
            <span class="workspace-card-icon">🩺</span>
            <strong>Condition-aware guidance</strong>
            <span>Explore general nutrition topics by health condition.</span>
        </div>
    </div>
    """,
)

# ---------------------------------------------------------------------------
# Shared RAG chain loader -- used by every tab except Nutrition Lookup.
# Cached so the vector store + LLM connection are only set up once.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base and connecting to Groq...")
def get_chain():
    from rag_chain import load_rag_chain
    return load_rag_chain()


def ask_rag(query: str):
    """
    Runs `query` through the shared RAG chain and returns
    {"answer": str, "sources": [str, ...]}. Raises the same exceptions
    as get_chain()/ask_question() so callers can show a friendly message.
    """
    from rag_chain import ask_question
    chain = get_chain()
    return ask_question(chain, query)


def render_rag_answer(answer: str, sources: list):
    st.markdown(f'<div class="chat-bubble-assistant">🤖 {answer}</div>', unsafe_allow_html=True)
    if sources:
        md_html(
            f'<div class="chat-sources">📄 Sources: {", ".join(sources)}</div>',
        )


def section_header(icon: str, title: str, caption: str):
    """Big, hoverable section title + caption card shown at the top of each tab."""
    md_html(
        f"""
        <div class="section-hero">
            <div class="section-hero-icon">{icon}</div>
            <div class="section-hero-text">
                <div class="section-hero-title">{title}</div>
                <p class="section-caption">{caption}</p>
            </div>
        </div>
        """,
    )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔍  Nutrition Lookup",
        "🤖  Ask AI Nutrition Assistant",
        "📊  Personalized Diet Planner",
        "🍽️  Meal Planner",
        "🩺  Disease Nutrition Advisor",
    ]
)

# -------------------------------------------------------------------
# TAB 1: Live nutrition lookup via USDA API, scaled to a quantity
# (the only tab that calls an external API)
# -------------------------------------------------------------------
with tab1:
    section_header(
        "🔍",
        "Nutrition Lookup",
        "Enter a food and how much of it you're having — nutrient values scale automatically.",
    )

    col_qty, col_unit, col_food = st.columns([1, 1, 2.5])
    with col_qty:
        quantity = st.number_input("Quantity", min_value=1, value=100, step=10)
    with col_unit:
        unit = st.selectbox("Unit", ["grams", "kg"], index=0)
    with col_food:
        query = st.text_input("Food name", placeholder="e.g. banana, chicken breast, brown rice")

    search_clicked = st.button("🔍 Search", key="search_btn")

    if search_clicked and query:
        try:
            with st.spinner("Searching USDA database..."):
                results = search_food(query)

            options = [
                f"{r['description']} ({r['brandOwner']})" if r["brandOwner"] else r["description"]
                for r in results
            ]
            st.session_state["last_results"] = results
            st.session_state["last_options"] = options
        except NutritionAPIError as e:
            st.error(str(e))
            st.session_state.pop("last_results", None)

    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        options = st.session_state["last_options"]

        choice_idx = st.selectbox(
            "Select the closest match:",
            range(len(options)),
            format_func=lambda i: options[i],
        )
        selected = results[choice_idx]

        try:
            with st.spinner("Fetching nutrient details..."):
                details = get_food_details(selected["fdcId"])

            # USDA nutrient amounts are reported per 100g reference amount.
            grams = quantity * 1000 if unit == "kg" else quantity
            scale = grams / 100.0

            qty_label = f"{quantity} {unit}"

            md_html(
                f"""
                <div class="food-title-card">
                    <h3>{details['description']}</h3>
                    <span class="qty-tag">Showing values for {qty_label}</span>
                </div>
                """,
            )

            if not details["nutrients"]:
                st.warning("No detailed nutrient data available for this item.")
            else:
                cols = st.columns(3)
                items = list(details["nutrients"].items())
                for i, (name, info) in enumerate(items):
                    scaled_amount = round(info["amount"] * scale, 2)
                    with cols[i % 3]:
                        st.metric(label=name, value=f"{scaled_amount} {info['unit']}")

        except NutritionAPIError as e:
            st.error(str(e))

# -------------------------------------------------------------------
# TAB 2: RAG-powered nutrition Q&A using a model via Groq's API
# -------------------------------------------------------------------
with tab2:
    section_header(
        "🤖",
        "Ask AI Nutrition Assistant",
        "Ask questions answered from your own nutrition knowledge base — grounded in the "
        "files you've added to <code>data/</code>, answered by a Groq-hosted model.",
    )

    try:
        get_chain()
        chain_ready = True
    except (FileNotFoundError, ValueError) as e:
        chain_ready = False
        st.error(str(e))
    except Exception as e:
        chain_ready = False
        st.error(
            "Could not connect to Groq. Double-check your GROQ_API_KEY in "
            f".env is correct and active.\n\nDetails: {e}"
        )

    if chain_ready:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        col_q, col_btn = st.columns([4, 1])
        with col_q:
            question = st.text_input(
                "Your nutrition question",
                placeholder="e.g. How much protein do I need per day for muscle gain?",
                key="rag_question",
                label_visibility="collapsed",
            )
        with col_btn:
            ask_clicked = st.button("Ask", key="ask_btn")

        if ask_clicked and question:
            with st.spinner("Thinking..."):
                try:
                    response = ask_rag(question)
                except APIError as e:
                    st.error(
                        "Groq rejected the request. Check that your API key is valid "
                        "and `GROQ_MODEL` names an available model. "
                        f"Details: {e}"
                    )
                else:
                    st.session_state.chat_history.append(
                        (question, response["answer"], response["sources"])
                    )

        for q, a, sources in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="chat-bubble-user">🧑 <b>You:</b> {q}</div>', unsafe_allow_html=True)
            render_rag_answer(a, sources)
            st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 3: Personalized Diet Planner
# Collects a profile, asks the RAG assistant to generate guidance --
# no calorie/macro math is done in this file.
# -------------------------------------------------------------------
with tab3:
    section_header(
        "📊",
        "Personalized Diet Planner",
        "Share your profile and goal — the AI assistant will generate personalized "
        "guidance grounded in your knowledge base.",
    )

    with st.form("diet_planner_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=100, value=28)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=230, value=165)
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)
        with c3:
            activity = st.selectbox(
                "Activity level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extremely active"],
                index=1,
            )
            goal = st.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain muscle / weight"])

        submitted = st.form_submit_button("📊 Get my personalized diet plan")

    if submitted:
        planner_query = (
            f"Suggest a personalized daily diet plan, including approximate calorie and "
            f"macronutrient guidance, for a {age}-year-old {gender.lower()} who is "
            f"{height_cm} cm tall, weighs {weight_kg} kg, has a '{activity.lower()}' "
            f"activity level, and whose goal is to '{goal.lower()}'."
        )
        try:
            with st.spinner("Generating your personalized plan..."):
                response = ask_rag(planner_query)
            st.session_state["diet_plan_response"] = response
        except (FileNotFoundError, ValueError) as e:
            st.session_state.pop("diet_plan_response", None)
            st.error(str(e))
        except Exception as e:
            st.session_state.pop("diet_plan_response", None)
            st.error(
                "Could not connect to Groq. Double-check your GROQ_API_KEY in "
                f".env is correct and active.\n\nDetails: {e}"
            )

    if "diet_plan_response" in st.session_state:
        st.markdown('<div class="nx-card"><h3>🎯 Your Personalized Guidance</h3></div>', unsafe_allow_html=True)
        resp = st.session_state["diet_plan_response"]
        render_rag_answer(resp["answer"], resp["sources"])
    else:
        st.info("Fill in the form above and click **Get my personalized diet plan** to get started.")

# -------------------------------------------------------------------
# TAB 4: Meal Planner
# Collects preferences, asks the RAG assistant to build a day's meals --
# no meal database or calorie math is hardcoded in this file.
# -------------------------------------------------------------------
with tab4:
    section_header(
        "🍽️",
        "Meal Planner",
        "Tell us your meal preferences — the AI assistant will build a full day's meal "
        "plan grounded in your knowledge base.",
    )

    mc1, mc2, mc3 = st.columns([1, 1, 1.4])
    with mc1:
        meals_per_day = st.selectbox("Meals per day", [3, 4, 5], index=0)
    with mc2:
        diet_pref = st.selectbox("Diet preference", ["Non-vegetarian", "Vegetarian", "Vegan"])
    with mc3:
        target_calories = st.text_input(
            "Target calories/day (optional)", placeholder="e.g. 2000"
        )

    generate_clicked = st.button("🍽️ Generate meal plan")

    if generate_clicked:
        calorie_clause = f", targeting approximately {target_calories} calories per day" if target_calories.strip() else ""
        meal_query = (
            f"Create a full day meal plan with {meals_per_day} meals for a "
            f"{diet_pref.lower()} diet{calorie_clause}. Suggest specific foods for each "
            f"meal and approximate calories per meal where possible."
        )
        try:
            with st.spinner("Generating your meal plan..."):
                response = ask_rag(meal_query)
            st.session_state["meal_plan_response"] = response
        except (FileNotFoundError, ValueError) as e:
            st.session_state.pop("meal_plan_response", None)
            st.error(str(e))
        except Exception as e:
            st.session_state.pop("meal_plan_response", None)
            st.error(
                "Could not connect to Groq. Double-check your GROQ_API_KEY in "
                f".env is correct and active.\n\nDetails: {e}"
            )

    if "meal_plan_response" in st.session_state:
        st.markdown('<div class="nx-card"><h3>📅 Your Day\'s Meal Plan</h3></div>', unsafe_allow_html=True)
        resp = st.session_state["meal_plan_response"]
        render_rag_answer(resp["answer"], resp["sources"])
    else:
        st.info("Set your preferences above and click **Generate meal plan** to see your day's meals.")

# -------------------------------------------------------------------
# TAB 5: Disease Nutrition Advisor
# Picks a condition (+ optional custom question), asks the RAG assistant
# -- no eat/avoid lists are hardcoded in this file.
# -------------------------------------------------------------------
with tab5:
    section_header(
        "🩺",
        "Disease Nutrition Advisor",
        "Select a condition for AI-generated dietary guidance, grounded in your knowledge "
        "base. This is educational information, not a substitute for advice from your "
        "doctor or dietitian.",
    )

    condition = st.selectbox(
        "Condition",
        [
            "Type 2 Diabetes",
            "Hypertension (High Blood Pressure)",
            "Heart Disease",
            "Obesity / Weight Management",
            "Chronic Kidney Disease (CKD)",
            "PCOS / PCOD",
        ],
    )
    custom_question = st.text_input(
        "Optional: ask something specific about this condition",
        placeholder="e.g. What breakfast options work well for this condition?",
    )

    advisor_clicked = st.button("🩺 Get nutrition guidance")

    if advisor_clicked:
        if custom_question.strip():
            disease_query = f"For someone with {condition}, {custom_question.strip()}"
        else:
            disease_query = (
                f"What foods should someone with {condition} eat and what should they "
                f"avoid? Give practical, general dietary tips."
            )
        try:
            with st.spinner("Fetching guidance..."):
                response = ask_rag(disease_query)
            st.session_state["disease_response"] = response
        except (FileNotFoundError, ValueError) as e:
            st.session_state.pop("disease_response", None)
            st.error(str(e))
        except Exception as e:
            st.session_state.pop("disease_response", None)
            st.error(
                "Could not connect to Groq. Double-check your GROQ_API_KEY in "
                f".env is correct and active.\n\nDetails: {e}"
            )

    if "disease_response" in st.session_state:
        st.markdown(f'<div class="nx-card"><h3>🩺 Guidance for {condition}</h3></div>', unsafe_allow_html=True)
        resp = st.session_state["disease_response"]
        render_rag_answer(resp["answer"], resp["sources"])
        st.caption(
            "⚠️ General educational guidance only — always confirm dietary changes with a "
            "qualified healthcare professional, especially for kidney disease or other "
            "complex conditions."
        )
    else:
        st.info("Choose a condition above and click **Get nutrition guidance**.")
