"""Yes/no margin, binary entropy, boundary error, answer flip rate."""
import math
import numpy as np


def logsumexp(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    m = arr.max()
    return float(m + math.log(np.exp(arr - m).sum()))


def yes_no_margin(logit_yes, logit_no):
    return float(logit_yes - logit_no)


def binary_entropy_from_logits(logit_yes, logit_no, eps=1e-12):
    m = max(logit_yes, logit_no)
    p_yes = math.exp(logit_yes - m) / (math.exp(logit_yes - m) + math.exp(logit_no - m))
    p_no = 1.0 - p_yes
    p_yes = min(max(p_yes, eps), 1 - eps)
    p_no = 1.0 - p_yes
    return float(-p_yes * math.log(p_yes) - p_no * math.log(p_no)), float(p_yes)


def boundary_index(margins):
    margins = np.asarray(margins, dtype=np.float64)
    return int(np.argmin(np.abs(margins)))


def boundary_error(margins, max_ambiguity_index):
    return int(abs(boundary_index(margins) - int(max_ambiguity_index)))


def answer_flip_rate(parsed_answers):
    n = len(parsed_answers)
    if n < 2:
        return 0.0
    flips = sum(1 for i in range(n - 1) if parsed_answers[i] != parsed_answers[i + 1])
    return float(flips / (n - 1))
