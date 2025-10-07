#  AI Text Generator with Sentiment Analysis

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B.svg)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-FLAN--T5--Large-green.svg)

A modern Streamlit application that blends RoBERTa-based sentiment analysis with local FLAN-T5 instruction-tuned generation to craft on-tone narratives without any external APIs.

## Features

- **Sentiment analysis** powered by `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Instruction-based text generation** powered locally by `google/flan-t5-large`
- **Length presets** (Short, Medium, Long) with automatic token controls
- **Modern UI** featuring gradient styling, inline badges, clipboard + download actions
- **Offline-friendly** workflow that never sends prompts to external services

## Project Structure

```
.
├── app.py                  # Streamlit interface combining analysis and generation
├── sentiment_analyzer.py   # RoBERTa-based sentiment analysis helper
├── text_generator.py       # Local FLAN-T5 generator with tone/length controls
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation and usage guide
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-text-generator-sentiment.git
   cd ai-text-generator-sentiment
   ```
2. **Create & activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Wait for the initial model downloads (about ~1 GB on first launch) and watch the sidebar for status tips.
3. Provide a prompt (anything from a single word to a detailed brief), pick a tone or auto-detect, choose a length, and press **Generate Text**.
4. Copy or download the generated narrative with a single click.

### Example Prompts

- "Describe the long-term environmental and economic impacts of transitioning coastal cities to tidal energy, focusing on infrastructure, job creation, and resilience."
- "Craft a product announcement for an upcoming augmented reality headset designed for collaborative workplace training in manufacturing plants."

## Technical Approach

- **Sentiment Analysis with RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest` offers transformer-level contextual understanding, outperforming lexicon approaches like VADER on colloquial and nuanced text thanks to subword tokenization and pretrained language modeling.
- **Text Generation with FLAN-T5 (local)**: `google/flan-t5-large` offers a high-quality 780M parameter model that excels at instruction-following. Running it locally avoids API quotas and keeps data on-device.
- **Instruction Prompting**: Each call builds a structured set of natural-language requirements that spell out tone, length, topical focus, and coherence, keeping the FLAN-T5 output tightly aligned with expectations.

## Challenges Faced & Solutions

- **Balancing quality vs. footprint**: Earlier experiments with smaller local models produced off-topic prose. Upgrading to `flan-t5-large` delivers significantly stronger instruction following and coherence for more demanding generation tasks.
- **Sentiment alignment**: Prompt engineering plus explicit tone instructions ensure the generated text mirrors the detected (or selected) sentiment.
- **Model warm-up & caching**: The first pass can take a while while weights download; Streamlit spinners and resource caching keep subsequent runs responsive.

## Deployment (Streamlit Cloud)

1. Fork or upload this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and create a new app pointing to `app.py`.
3. Because the model downloads are sizable, consider adding a `requirements.txt` pre-install step and using Streamlit's persistent caching so the weights remain between restarts (Streamlit Cloud retains `.cache` between runs).
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically and bootstraps the app.

## Future Enhancements

- Support multiple generator backends (Llama 3, Claude Infrastructure, etc.) with user selection
- Offer stylistic presets (story, report, marketing copy) layered atop sentiment control
- Persist generation history with export to CSV/Markdown
- Add multilingual sentiment detection and generation pathways

Enjoy crafting sentiment-aligned narratives that resonate with any audience! 🚀

