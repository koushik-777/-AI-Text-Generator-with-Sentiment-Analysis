"""Text generation utilities using local Transformer models."""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LocalTextGenerator:
    """Generate text locally using a fine-tuned instruction model."""

    MODEL_ID = "google/flan-t5-large"

    SENTIMENT_INSTRUCTIONS = {
        "positive": "optimistic, enthusiastic, uplifting tone focusing on benefits and hopeful outcomes",
        "negative": "critical, concerned, pessimistic tone emphasizing problems and challenges",
        "neutral": "objective, balanced, factual tone without emotional bias",
    }

    LENGTH_GUIDES = {
        "short": "50-75 words (2-3 sentences)",
        "medium": "150-200 words (1 large paragraph)",
        "long": "300-400 words (2-3 paragraphs)",
    }

    MAX_NEW_TOKENS = {
        "short": 120,
        "medium": 300,
        "long": 512,
    }

    def __init__(self, *, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID)
        self._model.to(self.device)
        self._model.eval()

    def _build_instruction_prompt(self, user_prompt: str, sentiment: str, length: str) -> str:
        sentiment_key = sentiment.lower()
        length_key = length.lower()

        if sentiment_key not in self.SENTIMENT_INSTRUCTIONS:
            raise ValueError(f"Unsupported sentiment option: {sentiment}")
        if length_key not in self.LENGTH_GUIDES:
            raise ValueError(f"Unsupported length option: {length}")

        tone_instruction = self.SENTIMENT_INSTRUCTIONS[sentiment_key]
        length_instruction = self.LENGTH_GUIDES[length_key]

        return (
            "You are a skilled writer. "
            "Write a response that strictly follows the user's topic and instructions.\n\n"
            f"Topic: {user_prompt}\n\n"
            "Requirements:\n"
            f"- Tone: {tone_instruction}\n"
            f"- Length: {length_instruction}\n"
            "- Stay strictly on topic\n"
            "- Be coherent, well structured, and easy to follow\n"
            "- Avoid repetition\n\n"
            "Write the full response now."
        )

    def generate(self, prompt: str, sentiment: str, length: str) -> str:
        """Generate text for the given prompt using the configured model."""

        instruction_prompt = self._build_instruction_prompt(prompt, sentiment, length)
        inputs = self._tokenizer(
            instruction_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        max_new_tokens = self.MAX_NEW_TOKENS[length.lower()]

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1,
            )

        decoded = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return decoded.strip()
