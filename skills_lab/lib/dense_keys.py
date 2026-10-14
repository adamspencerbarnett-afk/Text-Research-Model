"""Dense question keys: a pretrained sentence encoder as the memory's second key (the standard
retriever of current RAG systems), in the same slot as the K1 learned key.

    mem.attach_learned(DenseKeys(device="cuda", weight=1.0))

Retrieval then ranks the top-50 literal candidates together with the top-50 dense candidates by
    literal similarity + weight x cosine similarity of the encoder vectors,
and returns the literal similarity as the score, so the memory gate and the evaluation groups keep
their meaning. The encoder is all-MiniLM-L6-v2 (22M parameters, Apache 2.0, trained contrastively on
about a billion sentence pairs): it imports outside knowledge into retrieval only, never into the
answering network, and every result that uses it says so.
"""
from __future__ import annotations

import torch

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseKeys:
    def __init__(self, device: str = "cpu", weight: float = 1.0, model: str = MODEL):
        from sentence_transformers import SentenceTransformer
        self.dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = SentenceTransformer(model, device=self.dev)
        self.name, self.weight, self.lit_std, self.key_std = model, weight, 1.0, 1.0
        self.vectors = torch.zeros(0, self.model.get_embedding_dimension(), device=self.dev)

    @torch.no_grad()
    def embed(self, questions: list[bytes], batch: int = 256) -> torch.Tensor:
        if not questions:
            return torch.zeros(0, self.vectors.shape[1], device=self.dev)
        text = [q.decode("utf-8", "replace") for q in questions]
        return self.model.encode(text, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True,
                                 show_progress_bar=False).to(self.dev)
