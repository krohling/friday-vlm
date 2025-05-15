# test_length_grouping.py
import math
import random

import pytest
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Adjust this import to wherever you placed the implementation
# ─────────────────────────────────────────────────────────────────────────────
from friday.train.sampling import (
    split_to_even_chunks,
    get_length_grouped_indices,
    get_modality_length_grouped_indices,
)

# Utilities
# ---------------------------------------------------------------------------


def _tot_length(indices, lengths):
    """Helper: sum of token lengths for a list of indices."""
    return sum(lengths[i] for i in indices)


def _is_permutation(result, n):
    """True if result is a permutation of range(n)."""
    return sorted(result) == list(range(n))


# ─────────────────────────────────────────────────────────────────────────────
# 1. split_to_even_chunks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num_chunks", [2, 3, 5])
def test_indices_preserved(num_chunks):
    """Every original index must appear exactly once."""
    indices = list(range(17))
    lengths = [i + 1 for i in indices]
    chunks = split_to_even_chunks(indices, lengths, num_chunks)

    flat = [i for c in chunks for i in c]
    assert _is_permutation(flat, len(indices))

def test_uneven_not_divisible_path():
    """When not divisible the function falls back to simple slicing."""
    indices = list(range(10))  # 10 % 3 != 0
    lengths = [1] * 10
    chunks = split_to_even_chunks(indices, lengths, num_chunks=3)

    expected = [indices[i::3] for i in range(3)]
    assert chunks == expected


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_length_grouped_indices
# ─────────────────────────────────────────────────────────────────────────────
def test_single_gpu_trivial():
    """world_size=1 should just yield a length-sorted megabatch list."""
    lengths = [9, 1, 8, 2, 7, 3]
    out = get_length_grouped_indices(lengths, batch_size=2, world_size=1,
                                        generator=torch.Generator().manual_seed(0))

    # Inside each megabatch (size=batch_size) lengths should descend
    for i in range(0, len(out), 2):
        a, b = lengths[out[i]], lengths[out[i + 1]]
        assert a >= b

    assert _is_permutation(out, len(lengths))

def test_multi_gpu_balancing():
    lengths = [1, 50, 2, 45, 3, 40, 4, 35]
    bs, ws = 2, 2
    idx = get_length_grouped_indices(
        lengths, bs, ws, generator=torch.Generator().manual_seed(42)
    )

    mb_size = bs * ws
    megabatches = [idx[i : i + mb_size] for i in range(0, len(idx), mb_size)]
    for mb in megabatches:
        # contiguous slices, not step-slicing
        chunks = [mb[g * bs : (g + 1) * bs] for g in range(ws)]
        totals = [_tot_length(c, lengths) for c in chunks]
        assert max(totals) - min(totals) <= max(lengths)


def test_determinism_with_seed():
    """Same manual seed ⇒ identical ordering."""
    lengths = [random.randint(1, 100) for _ in range(37)]
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)

    a = get_length_grouped_indices(lengths, 4, 3, generator=g1)
    b = get_length_grouped_indices(lengths, 4, 3, generator=g2)
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_modality_length_grouped_indices
# ─────────────────────────────────────────────────────────────────────────────
def test_all_same_modality_delegates():
    lengths = [5, 10, 3, 7, 2]

    gen = torch.Generator().manual_seed(0)
    a = get_length_grouped_indices(lengths, 2, 1, generator=gen)
    
    gen.manual_seed(0) # reset state
    b = get_modality_length_grouped_indices(lengths, 2, 1, generator=gen)
    assert a == b

def test_mixed_modalities_permutation():
    """Mixed +/- list returns a valid permutation and keeps counts."""
    pos = [10, 12, 8, 14]
    neg = [-5, -7, -6]        # text-only ⇒ stored as negative
    lengths = pos + neg

    out = get_modality_length_grouped_indices(
        lengths, batch_size=2, world_size=2,
        generator=torch.Generator().manual_seed(99)
    )

    # 1) permutation property
    assert _is_permutation(out, len(lengths))

    # 2) counts preserved
    mm_count = sum(lengths[i] > 0 for i in out)
    lang_count = sum(lengths[i] < 0 for i in out)
    assert mm_count == len(pos)
    assert lang_count == len(neg)

def test_error_on_zero_length():
    with pytest.raises(AssertionError, match="zero length"):
        get_modality_length_grouped_indices(
            [5, 0, -7], batch_size=1, world_size=1
        )
