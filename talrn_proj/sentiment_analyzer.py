"""Sentiment analysis utilities using a RoBERTa model from Hugging Face."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class SentimentResult:
    """Structured result of a sentiment analysis operation."""

    sentiment: str
    confidence: float

    def as_dict(self) -> Dict[str, float]:
        return {"sentiment": self.sentiment, "confidence": round(self.confidence, 4)}


class SentimentAnalyzer:
    """Analyze sentiment using the `cardiffnlp/twitter-roberta-base-sentiment-latest` model."""

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.eval()

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze the given text and return the dominant sentiment and confidence score."""
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits[0]
            probabilities = torch.nn.functional.softmax(logits, dim=0)

        top_index = int(probabilities.argmax().item())
        sentiment = self.LABEL_MAPPING.get(top_index, "neutral")
        confidence = float(probabilities[top_index].item())

        return SentimentResult(sentiment=sentiment, confidence=confidence).as_dict()
