# evaluator.py
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from core import CHAR_DENSITY

EVAL_WEIGHTS = {
    "ssim": 0.6,
    "edge": 0.3,
    "noise": 0.25,
    "density": 0.05
}

def rasterize_char_grid_to_lum(char_rows, target_shape=None):
    if isinstance(char_rows, list) and len(char_rows) and isinstance(char_rows[0], str):
        rows = [list(r) for r in char_rows]
    else:
        rows = char_rows
    h = len(rows)
    w = len(rows[0]) if h else 0
    arr = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            ch = rows[y][x] if x < len(rows[y]) else ' '
            arr[y, x] = CHAR_DENSITY.get(ch, 0.0) * 255.0
    if target_shape is not None:
        th, tw = target_shape
        out = np.zeros((th, tw), dtype=np.float32)
        mh = min(th, h)
        mw = min(tw, w)
        out[:mh, :mw] = arr[:mh, :mw]
        return out
    return arr

def edge_overlap_score(orig_mag, rendered_lum):
    rx = cv2.Sobel(rendered_lum.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    ry = cv2.Sobel(rendered_lum.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    rend_mag = np.sqrt(rx**2 + ry**2)
    o = orig_mag / (orig_mag.max() + 1e-9) if orig_mag.max() > 0 else orig_mag
    r = rend_mag / (rend_mag.max() + 1e-9) if rend_mag.max() > 0 else rend_mag
    t_o = np.percentile(o, 75) if o.size else 0.0
    t_r = np.percentile(r, 75) if r.size else 0.0
    o_bin = (o >= t_o).astype(np.float32)
    r_bin = (r >= t_r).astype(np.float32)
    inter = np.sum(o_bin * r_bin)
    union = np.sum(o_bin) + np.sum(r_bin) - inter
    return float(inter / union) if union > 0 else 0.0

def local_noise_penalty(char_grid):
    rows = len(char_grid)
    cols = len(char_grid[0]) if rows else 0
    occ = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            ch = char_grid[r][c] if c < len(char_grid[r]) else ' '
            occ[r, c] = 1.0 if (ch != ' ' and ch != '') else 0.0
    kernel = np.ones((3,3), dtype=np.float32)
    local_sum = cv2.filter2D(occ, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    local_mean = local_sum / 9.0
    local_abs_diff = cv2.filter2D(np.abs(occ - local_mean), -1, kernel, borderType=cv2.BORDER_CONSTANT) / 9.0
    penalty = float(np.mean(local_abs_diff))
    singletons = 0
    for r in range(rows):
        for c in range(cols):
            if occ[r,c] == 1.0:
                r0 = max(0, r-1); r1 = min(rows, r+2)
                c0 = max(0, c-1); c1 = min(cols, c+2)
                if np.sum(occ[r0:r1, c0:c1]) <= 1.0:
                    singletons += 1
    singleton_pen = singletons / max(1, rows*cols)
    return min(1.0, penalty*1.5 + singleton_pen*2.0)

def density_penalty(char_grid):
    rows = len(char_grid)
    cols = len(char_grid[0]) if rows else 0
    total = rows * cols
    if total == 0:
        return 0.0
    non_space = sum(1 for r in range(rows) for c in range(cols) if char_grid[r][c] != ' ')
    density = non_space / total
    if density <= 0.6:
        return 0.0
    return (density - 0.6) / 0.4

def evaluate_candidate_grid(orig_lum, mag, char_rows, weights=None):
    weights = weights or EVAL_WEIGHTS
    ascii_lum = rasterize_char_grid_to_lum(char_rows, target_shape=orig_lum.shape)
    try:
        ssim_val = ssim(orig_lum, ascii_lum, data_range=255)
    except Exception:
        corr = np.corrcoef(orig_lum.flatten(), ascii_lum.flatten())[0,1]
        ssim_val = float(np.clip((corr + 1)/2, 0.0, 1.0))
    edge_score = edge_overlap_score(mag, ascii_lum)
    noise_pen = local_noise_penalty([list(r) for r in char_rows])
    dens_pen = density_penalty([list(r) for r in char_rows])
    score = (weights['ssim'] * ssim_val + weights['edge'] * edge_score) - (weights['noise'] * noise_pen + weights['density'] * dens_pen)
    score = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    return {
        "score": score,
        "ssim": float(ssim_val),
        "edge": float(edge_score),
        "noise_penalty": float(noise_pen),
        "density_penalty": float(dens_pen),
        "ascii_lum": ascii_lum
    }

def evaluate_similarity(orig_img, ascii_art, char_density_map=CHAR_DENSITY):
    orig_lum = (0.2126*orig_img[:,:,0] + 0.7152*orig_img[:,:,1] + 0.0722*orig_img[:,:,2])
    ascii_img = np.zeros_like(orig_lum)
    lines = ascii_art.splitlines()
    for y, line in enumerate(lines[:ascii_img.shape[0]]):
        for x, ch in enumerate(line[:ascii_img.shape[1]]):
            ascii_img[y,x] = char_density_map.get(ch,0)*255
    score = ssim(orig_lum, ascii_img, data_range=255)
    return score
