from __future__ import annotations

import json
import streamlit as st

from sentiment_analyzer import SentimentAnalyzer
from text_generator import LocalTextGenerator

st.set_page_config(
    page_title="AI Text Generator with Sentiment Analysis",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@st.cache_resource(show_spinner=True)
def load_generator() -> LocalTextGenerator:
    return LocalTextGenerator()


def inject_custom_css() -> None:
    gradient_background = """
        <style>
            body {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
                color: #f8fafc;
            }
            section.main > div {
                background: rgba(15, 23, 42, 0.85);
                border-radius: 16px;
                padding: 2.5rem;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
            }
            .stButton > button {
                background: linear-gradient(135deg, #6366f1, #0ea5e9);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.75rem 1.25rem;
                font-weight: 600;
                box-shadow: 0 8px 18px rgba(99, 102, 241, 0.35);
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #4f46e5, #0284c7);
            }
            .result-card {
                background: rgba(15, 23, 42, 0.6);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                backdrop-filter: blur(6px);
            }
            .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                border-radius: 999px;
                padding: 0.4rem 0.9rem;
                font-weight: 600;
                color: white;
                margin-bottom: 1rem;
            }
            .copy-button {
                background: transparent;
                border: 1px solid rgba(148, 163, 184, 0.4);
                color: #e2e8f0;
                padding: 0.5rem 1rem;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.2s ease;
                font-weight: 500;
            }
            .copy-button:hover {
                background: rgba(148, 163, 184, 0.15);
            }
        </style>
    """
    st.markdown(gradient_background, unsafe_allow_html=True)


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"


def render_sentiment_badge(sentiment: str, confidence: float) -> None:
    colors = {
        "positive": "background: linear-gradient(135deg, #16a34a, #22c55e);",
        "negative": "background: linear-gradient(135deg, #dc2626, #f87171);",
        "neutral": "background: linear-gradient(135deg, #64748b, #94a3b8);",
    }
    style = colors.get(sentiment, colors["neutral"])
    badge_html = f"""
        <div class="badge" style="{style}">
            <span>Detected sentiment:</span>
            <strong>{sentiment.title()} · {format_confidence(confidence)}</strong>
        </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


inject_custom_css()

with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption(
        "Runs 100% locally using RoBERTa for sentiment and FLAN-T5 for writing—no external APIs or tokens needed."
    )
    st.info(
        "First run downloads the models (~1 GB). Subsequent generations are much faster."
    )
    st.divider()
    st.subheader("About")
    st.write(
        "This app blends RoBERTa-based sentiment analysis with instruction-tuned FLAN-T5 text generation to craft tailored narratives entirely offline."
    )

st.title(" AI Text Generator with Sentiment Analysis")
prompt = st.text_area(
    "What should the AI write about?",
    height=220,
    placeholder="Describe the topic (a single word works too, but details help).",
)

col1, col2 = st.columns(2)

with col1:
    sentiment_choice = st.radio(
        "Target sentiment",
        options=("Auto-detect", "Positive", "Negative", "Neutral"),
        index=0,
    )

with col2:
    length_choice = st.select_slider(
        "Content length",
        options=["Short", "Medium", "Long"],
        value="Medium",
    )

if st.button("Generate Text", use_container_width=True):
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.error("Please enter a prompt so the model knows what to write about.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                analyzer = load_analyzer()
                analysis = analyzer.analyze(cleaned_prompt)
        except Exception as analysis_error:  
            st.error(f"Sentiment analysis failed: {analysis_error}")
        else:
            detected_sentiment = analysis.get("sentiment", "neutral")
            detected_confidence = analysis.get("confidence", 0.0)

            target_sentiment = (
                detected_sentiment
                if sentiment_choice.lower() == "auto-detect"
                else sentiment_choice.lower()
            )

            try:
                with st.spinner("Generating bespoke text locally..."):
                    generator = load_generator()
                    generated_text = generator.generate(
                        prompt=cleaned_prompt,
                        sentiment=target_sentiment,
                        length=length_choice.lower(),
                    )
            except Exception as generation_error:  
                st.error(f"Text generation failed: {generation_error}")
            else:
                with st.container():
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    render_sentiment_badge(detected_sentiment, detected_confidence)

                    tone_source = (
                        "auto-detected" if sentiment_choice.lower() == "auto-detect" else "manual selection"
                    )
                    st.caption(
                        f"Generation tone: {target_sentiment.title()} ({tone_source})"
                    )

                    st.markdown("### Generated Text")
                    st.write(generated_text)

                    word_count = len(generated_text.split())
                    st.caption(f"Word count: {word_count}")

                    text_for_js = json.dumps(generated_text)
                    copy_block = f"""
                        <button class="copy-button" onclick='navigator.clipboard.writeText({text_for_js}).then(() => {{
                            const el = this;
                            const original = el.innerText;
                            el.innerText = "Copied!";
                            setTimeout(() => {{ el.innerText = original; }}, 2000);
                        }})'>Copy to clipboard</button>
                    """
                    st.markdown(copy_block, unsafe_allow_html=True)

                    st.download_button(
                        label="Download as .txt",
                        data=generated_text,
                        file_name="ai_text_generator_output.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)
