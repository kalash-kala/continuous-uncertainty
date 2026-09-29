"""Question-token selection strategies (a) stopwords, (b) POS, (c) strategy_d.

Selectors map LLaVA tokenizer pieces to spaCy words, then pick which token
positions to use as queries for per-token attention diagnostics.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np


_SPACY = None  # cached spaCy nlp


def load_spacy(model_name: str = "en_core_web_sm"):
    """Lazy-load spaCy model. Raises with a helpful message if missing."""
    global _SPACY
    if _SPACY is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spacy is required for token selection. "
                "Install with `pip install spacy==3.7.2` and "
                "`python -m spacy download en_core_web_sm`."
            ) from e
        try:
            _SPACY = spacy.load(model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            ) from e
    return _SPACY


@dataclass
class SelectedToken:
    position: int                    # row index in the [T, P] per-token attention matrix
    text: str                        # cleaned LLaVA piece
    spacy_text: Optional[str] = None  # corresponding spaCy word text
    pos: Optional[str] = None         # POS of the spaCy word (or None)
    is_stop: bool = False
    score: Optional[float] = None     # ranking score (strategy_d only)

    def to_dict(self):
        return {
            "position": int(self.position),
            "text": self.text,
            "spacy_text": self.spacy_text,
            "pos": self.pos,
            "is_stop": bool(self.is_stop),
            "score": (float(self.score) if self.score is not None else None),
        }


def clean_llava_token(tok: str) -> str:
    """Strip SentencePiece prefix (▁) and BPE markers; return plain text."""
    if tok is None:
        return ""
    t = tok.replace("▁", " ").replace("Ġ", " ")
    t = t.replace("</s>", "").replace("<s>", "").replace("<0x0A>", "\n")
    return t.strip()


def _is_piece_meaningful(t: str) -> bool:
    if not t:
        return False
    if all(not c.isalnum() for c in t):
        return False
    return True


def align_llava_tokens_to_spacy(question: str, llava_tokens: Sequence[str]):
    """Best-effort alignment of LLaVA pieces to spaCy words by walking text.

    Returns a list (one entry per LLaVA token) of dicts:
        {"text": cleaned_piece, "spacy_idx": int|None, "spacy_text": str|None,
         "pos": str|None, "is_stop": bool}

    Method: clean each LLaVA piece, scan the question text starting at a moving
    cursor for that substring. Map character span -> spaCy token via doc.char_span.
    Falls back gracefully when no match is found.
    """
    nlp = load_spacy()
    doc = nlp(question)

    # Build a list of spaCy tokens with character spans.
    spacy_spans = [(t.idx, t.idx + len(t.text), t) for t in doc]

    cursor = 0
    aligned = []
    q_lower = question.lower()
    for raw in llava_tokens:
        cleaned = clean_llava_token(raw)
        if not _is_piece_meaningful(cleaned):
            aligned.append({"text": cleaned, "spacy_idx": None, "spacy_text": None,
                            "pos": None, "is_stop": False})
            continue
        # Search starting at cursor (case-insensitive).
        search_str = cleaned.lower()
        hit = q_lower.find(search_str, cursor)
        if hit < 0:
            # Try from the beginning (handles odd resegmentation)
            hit = q_lower.find(search_str)
        if hit < 0:
            aligned.append({"text": cleaned, "spacy_idx": None, "spacy_text": None,
                            "pos": None, "is_stop": False})
            continue
        end = hit + len(search_str)
        cursor = end
        # Find covering spaCy token (any whose span overlaps [hit, end))
        spacy_tok = None
        for s, e, tok in spacy_spans:
            if s <= hit < e or hit <= s < end:
                spacy_tok = tok
                break
        if spacy_tok is None:
            aligned.append({"text": cleaned, "spacy_idx": None, "spacy_text": None,
                            "pos": None, "is_stop": False})
        else:
            aligned.append({
                "text": cleaned,
                "spacy_idx": int(spacy_tok.i),
                "spacy_text": spacy_tok.text,
                "pos": spacy_tok.pos_,
                "is_stop": bool(spacy_tok.is_stop or spacy_tok.is_punct),
            })
    return aligned


def _build_initial(positions, llava_tokens, aligned):
    """Build SelectedToken list pre-filter (one entry per position)."""
    out = []
    for pos in positions:
        if pos >= len(aligned):
            continue
        a = aligned[pos]
        out.append(SelectedToken(
            position=pos,
            text=a["text"] or clean_llava_token(llava_tokens[pos]),
            spacy_text=a["spacy_text"],
            pos=a["pos"],
            is_stop=a["is_stop"],
        ))
    return out


def _filter_stopwords(tokens: List[SelectedToken]) -> List[SelectedToken]:
    return [t for t in tokens if (not t.is_stop) and t.text and any(c.isalnum() for c in t.text)]


def _filter_pos(tokens: List[SelectedToken], keep_pos: Sequence[str]) -> List[SelectedToken]:
    keep = set(keep_pos)
    return [t for t in tokens if t.pos in keep]


def _compute_strategy_d_scores(tokens: List[SelectedToken],
                               per_frame_matrices: Sequence[np.ndarray]) -> List[SelectedToken]:
    """Score = rank-sum across:
        +image_attention_mass (mean across frames)
        +temporal_variance (mean across patches of per-patch variance across frames)
        -attention_entropy  (mean across frames; we negate so 'low' is best)

    per_frame_matrices: list of np.ndarray, each shape [T, P] (head-averaged, single layer
        or averaged-across-layers). We expect P>0 and T>=max(token.position)+1.
    """
    if not tokens or not per_frame_matrices:
        return tokens

    # Stack into [F, T, P]
    F = len(per_frame_matrices)
    stack = np.stack(per_frame_matrices, axis=0).astype(np.float64)  # [F, T, P]
    eps = 1e-12

    pos_idx = np.array([t.position for t in tokens], dtype=np.int64)
    sub = stack[:, pos_idx, :]  # [F, K, P]

    # image-mass: sum over patches, then mean across frames -> [K]
    mass = sub.sum(axis=2).mean(axis=0)

    # temporal variance: variance over frames per patch, then mean over patches -> [K]
    tv = sub.var(axis=0).mean(axis=1)

    # entropy: normalize per-frame distribution then compute entropy, mean across frames -> [K]
    sub_n = sub / (sub.sum(axis=2, keepdims=True) + eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(sub_n * np.log(sub_n + eps)).sum(axis=2)  # [F, K]
    ent_mean = ent.mean(axis=0)

    # Rank-sum: higher mass, higher tv, lower entropy => higher score.
    def _rank(arr, descending=True):
        order = np.argsort(-arr if descending else arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr))
        return ranks
    r1 = _rank(mass, descending=True)
    r2 = _rank(tv, descending=True)
    r3 = _rank(ent_mean, descending=False)  # low entropy => rank 0
    combined = r1 + r2 + r3  # lower is better

    for i, t in enumerate(tokens):
        t.score = float(-combined[i])  # higher score = better
    return tokens


def select_tokens(llava_tokens: Sequence[str],
                  question: str,
                  strategy: str = "strategy_d",
                  top_k: int = 5,
                  pos_tags_to_keep: Sequence[str] = ("NOUN", "PROPN", "VERB", "ADJ", "ADV"),
                  per_frame_matrices: Optional[Sequence[np.ndarray]] = None
                  ) -> List[SelectedToken]:
    """Run the configured selection strategy.

    strategy: "stopwords" | "pos" | "strategy_d"
    top_k: max number of tokens to return (after filtering)
    pos_tags_to_keep: used by strategy='pos' and as a pre-filter for 'strategy_d'
    per_frame_matrices: required for strategy_d (list of [T, P] arrays across frames)
    """
    positions = list(range(len(llava_tokens)))
    aligned = align_llava_tokens_to_spacy(question, llava_tokens)
    base = _build_initial(positions, llava_tokens, aligned)
    # Always drop stopwords/punct/non-alnum as a baseline
    base = _filter_stopwords(base)

    if strategy == "stopwords":
        return base[:top_k]
    if strategy == "pos":
        return _filter_pos(base, pos_tags_to_keep)[:top_k]
    if strategy == "strategy_d":
        # Prefer to score among content-ish words to avoid wasting budget on noise.
        candidates = _filter_pos(base, pos_tags_to_keep) or base
        if per_frame_matrices is None or len(per_frame_matrices) == 0:
            return candidates[:top_k]
        scored = _compute_strategy_d_scores(candidates, per_frame_matrices)
        scored.sort(key=lambda t: (t.score if t.score is not None else -np.inf), reverse=True)
        return scored[:top_k]
    raise ValueError(f"Unknown selection strategy: {strategy}")
