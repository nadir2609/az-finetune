"""Numba port of resemble-ai/monotonic_align core.pyx (no C compiler needed)."""
import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def _maximum_path_each(path, value, t_x, t_y, max_neg_val):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@numba.njit(nogil=True, cache=True, parallel=True)
def _maximum_path_batch(paths, values, t_xs, t_ys, max_neg_val):
    for i in numba.prange(values.shape[0]):
        _maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)


def maximum_path_c(paths, values, t_xs, t_ys, max_neg_val=-1e9):
    _maximum_path_batch(paths, values, np.asarray(t_xs, dtype=np.int32),
                        np.asarray(t_ys, dtype=np.int32), np.float32(max_neg_val))
