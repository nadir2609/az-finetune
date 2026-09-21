import numpy as np
import torch
from monotonic_align import maximum_path, mask_from_lens


def brute_force_best(value):
    """Exhaustive monotonic alignment: each mel frame y maps to one text x,
    x starts at 0, ends at X-1, and advances by 0 or 1 per frame."""
    X, Y = value.shape
    best, best_path = -1e18, None

    def rec(y, x, score, path):
        nonlocal best, best_path
        if y == Y:
            if x == X - 1 and score > best:
                best, best_path = score, list(path)
            return
        for nx in (x, x + 1):
            if nx < X and (Y - y - 1) >= (X - 1 - nx):
                path.append(nx)
                rec(y + 1, nx, score + value[nx, y], path)
                path.pop()

    rec(1, 0, value[0, 0], [0])
    return best, best_path


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        X, Y = rng.integers(2, 5), rng.integers(5, 9)
        v = rng.normal(size=(X, Y)).astype(np.float32)
        path = maximum_path(torch.from_numpy(v)[None], torch.ones(1, X, Y))[0].numpy()
        _, bp = brute_force_best(v)
        expected = np.zeros((X, Y), dtype=np.float32)
        for y, x in enumerate(bp):
            expected[x, y] = 1
        np.testing.assert_array_equal(path, expected)


def test_mask_limits_path_and_shape():
    sim = torch.randn(2, 6, 20)
    mask = mask_from_lens(sim, torch.tensor([6, 3]), torch.tensor([20, 9]))
    path = maximum_path(sim, mask)
    assert path.shape == sim.shape
    assert path[1, 3:, :].sum() == 0 and path[1, :, 9:].sum() == 0
    assert torch.all(path[0].sum(0) == 1)           # each frame -> exactly one token
    assert torch.all(path[1, :, :9].sum(0) == 1)
