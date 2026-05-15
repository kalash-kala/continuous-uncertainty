"""Utilities for parsing the CSV manifest into per-frame records."""
import ast
import os
import re
import csv
from pathlib import Path


def _parse_list_field(value):
    """Parse a list-like CSV field. Handles `[1, 2, 3]` and `[yes yes]` (no comma)."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            return list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            inner = s[1:-1].strip()
            if not inner:
                return []
            parts = re.split(r"[,\s]+", inner)
            out = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    out.append(int(p))
                except ValueError:
                    out.append(p.strip("'\""))
            return out
    try:
        return [int(s)]
    except ValueError:
        return [s]


def parse_gt_answer(value):
    """Normalize the gt_answer column to a single 'yes'/'no' string."""
    if value is None:
        return None
    s = str(value).strip()
    if s.startswith("["):
        items = _parse_list_field(s)
        if items:
            return str(items[0]).strip().lower()
        return None
    return s.strip().lower()


def sequence_id_from_video_path(video_path):
    """`images/NExTVideo/0089/3066966990.mp4` -> `NExTVideo_0089_3066966990`."""
    s = str(video_path).strip()
    if s.startswith("images/"):
        s = s[len("images/"):]
    if s.endswith(".mp4"):
        s = s[:-4]
    return s.replace("/", "_")


def load_manifest_rows(manifest_path):
    """Yield dict rows from the CSV manifest."""
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def expand_row_to_frames(row, image_root):
    """Turn one CSV row (sequence) into 10 per-frame record dicts."""
    sequence_id = sequence_id_from_video_path(row["video_path"])
    question = row["question"]
    frames = _parse_list_field(row["frames"])
    center_frames = _parse_list_field(row["center_frames"])
    gt_answer = parse_gt_answer(row["gt_answer"])
    category = row.get("primary_category", "unknown")

    if not center_frames:
        max_ambiguity_frame = frames[len(frames) // 2] if frames else None
    else:
        max_ambiguity_frame = center_frames[0]

    try:
        max_ambiguity_index = frames.index(max_ambiguity_frame)
    except ValueError:
        max_ambiguity_index = len(frames) // 2

    records = []
    for i, frame_number in enumerate(frames):
        try:
            fn = int(frame_number)
            fname = f"frame_{fn:04d}.png"
        except (ValueError, TypeError):
            fname = f"frame_{frame_number}.png"
        image_path = os.path.join(image_root, sequence_id, fname)
        records.append({
            "sequence_id": sequence_id,
            "frame_index": i,
            "frame_number": frame_number,
            "image_path": image_path,
            "question": question,
            "category": category,
            "max_ambiguity_frame": max_ambiguity_frame,
            "max_ambiguity_index": max_ambiguity_index,
            "relative_ambiguity_distance": abs(i - max_ambiguity_index),
            "ground_truth": gt_answer,
            "video_path": row["video_path"],
        })
    return records
