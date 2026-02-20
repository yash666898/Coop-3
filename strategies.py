# strategies.py
import numpy as np
from core import floyd_steinberg_dither, orientation_char, best_char_for_luminance

def strategy_edge_focused(img, lum, mag, orient, params=None):
    params = params or {}
    h, w = lum.shape
    pct = params.get('edge_pct', 85)
    sparsity = params.get('sparsity', 0.6)
    thresh = np.percentile(mag, pct)
    coords = np.argwhere(mag >= thresh)
    coords = sorted(coords, key=lambda c: -mag[c[0], c[1]])
    keep_n = max(1, int(len(coords) * sparsity))
    keep = set((int(c[0]), int(c[1])) for c in coords[:keep_n])
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            if (y,x) in keep:
                row.append(orientation_char(orient[y,x]))
            else:
                row.append(' ')
        rows.append("".join(row))
    return rows

def strategy_shading_focused(img, lum, mag, orient, params=None):
    params = params or {}
    h, w = lum.shape
    dithered = floyd_steinberg_dither(lum)
    shade_charset = [' ', '░', '▒', '▓', '█']
    rows = []
    for y in range(h):
        row_chars = []
        for x in range(w):
            v = dithered[y,x] / 255.0
            idx = int(np.clip(v * (len(shade_charset)-1), 0, len(shade_charset)-1))
            row_chars.append(shade_charset[idx])
        rows.append("".join(row_chars))
    return rows

def strategy_hybrid(img, lum, mag, orient, params=None):
    edge_rows = strategy_edge_focused(img, lum, mag, orient, params or {})
    shade_rows = strategy_shading_focused(img, lum, mag, orient, params or {})
    merged = []
    for er, sr in zip(edge_rows, shade_rows):
        row = ''.join(e if e != ' ' else s for e, s in zip(er, sr))
        merged.append(row)
    return merged

def strategy_high_detail(img, lum, mag, orient, params=None):
    base = strategy_hybrid(img, lum, mag, orient, params)
    h = len(base)
    w = len(base[0]) if h else 0
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            ch = base[y][x]
            if ch == ' ':
                y0 = max(0, y-1); y1 = min(h, y+2)
                x0 = max(0, x-1); x1 = min(w, x+2)
                local = lum[y0:y1, x0:x1]
                if local.size and np.var(local) > 200:
                    row.append('.' if np.mean(local) > 128 else ',')
                else:
                    row.append(' ')
            else:
                row.append(ch)
        out.append("".join(row))
    return out

def strategy_simplified(img, lum, mag, orient, params=None):
    params = params or {}
    h, w = lum.shape
    thresh = np.percentile(mag, params.get('edge_pct', 90))
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            if mag[y,x] >= thresh:
                row.append('#')
            else:
                local = lum[max(0,y-2):y+3, max(0,x-2):x+3]
                if local.size and np.mean(local) < 100:
                    row.append('█')
                else:
                    row.append(' ')
        rows.append("".join(row))
    return rows

STRATEGIES = {
    "edge": strategy_edge_focused,
    "shading": strategy_shading_focused,
    "hybrid": strategy_hybrid,
    "high_detail": strategy_high_detail,
    "simplified": strategy_simplified
}
