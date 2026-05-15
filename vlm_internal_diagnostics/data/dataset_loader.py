"""Dataset loader producing per-sequence groups of per-frame records."""
from collections import defaultdict
from .manifest_utils import load_manifest_rows, expand_row_to_frames


class SequenceDataset:
    """Iterable of sequences, each a list of per-frame record dicts."""

    def __init__(self, manifest_path, image_root, max_sequences=None,
                 categories=None, limit_frames_per_sequence=None):
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.max_sequences = max_sequences
        self.categories = set(categories) if categories else None
        self.limit_frames_per_sequence = limit_frames_per_sequence
        self._sequences = None

    def _load(self):
        seqs = []
        for row in load_manifest_rows(self.manifest_path):
            cat = row.get("primary_category", "unknown")
            if self.categories is not None and cat not in self.categories:
                continue
            records = expand_row_to_frames(row, self.image_root)
            if self.limit_frames_per_sequence:
                records = records[: self.limit_frames_per_sequence]
            if records:
                seqs.append(records)
            if self.max_sequences is not None and len(seqs) >= self.max_sequences:
                break
        return seqs

    def __iter__(self):
        if self._sequences is None:
            self._sequences = self._load()
        return iter(self._sequences)

    def __len__(self):
        if self._sequences is None:
            self._sequences = self._load()
        return len(self._sequences)
