import numpy as np
import torch

from monotonic_align.core import maximum_path_c


def mask_from_len(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    index = torch.arange(max_len).to(lens).view(1, -1)
    return index < lens.unsqueeze(1)


def mask_from_lens(similarity, symbol_lens, mel_lens):
    _, S, T = similarity.size()
    mask_ST = mask_from_len(symbol_lens, S).unsqueeze(2) * mask_from_len(mel_lens, T).unsqueeze(1)
    return mask_ST.to(similarity)


def maximum_path(value, mask=None):
    if mask is None:
        mask = torch.ones_like(value)
    device, dtype = value.device, value.dtype
    value = np.ascontiguousarray((value * mask).data.cpu().numpy().astype(np.float32))
    mask = mask.data.cpu().numpy()
    path = np.zeros_like(value).astype(np.int32)
    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
