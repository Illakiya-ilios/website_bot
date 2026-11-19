"""
Relevance reranking for retrieved chunks.

A small cross-encoder (ms-marco-MiniLM-L-6-v2, about 90 MB, CPU) scores each
(query, chunk) pair and keeps the best ones. It runs locally because the AWS
account isn't authorised for Bedrock's Rerank API. It uses the transformers
and torch packages that llm-guard already installs.

RERANKER_ENABLED=false in .env turns it off (chunks are then kept in
retrieval order).
"""
import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("ilios-api.reranker")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

_tokenizer = _model = None
if ENABLED:
    logger.info("Loading reranker %s (first run downloads it)...", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()


def rerank(query, documents, top_n):
    """Return the top_n documents most relevant to the query."""
    if not documents or _model is None:
        return documents[:top_n]
    with torch.no_grad():
        encoded = _tokenizer(
            [query] * len(documents),
            [d.page_content for d in documents],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        scores = _model(**encoded).logits.squeeze(-1).tolist()
    ranked = sorted(zip(scores, range(len(documents))), key=lambda p: -p[0])
    return [documents[i] for _, i in ranked[:top_n]]
