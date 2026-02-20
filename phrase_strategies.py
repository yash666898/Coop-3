# phrase_strategies.py
import random 
import numpy as np 
from core import floyd_steinberg_dither, CHAR_DENSITY 
from strategies import STRATEGIES # to register later if desired

def normalize_phrase(phrase):
    seen = set()
    chars = []
    for ch in phrase:
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
    return chars

def phrase_char_density_map(phrase_chars):
    dens = {}
    for ch in phrase_chars:
        if ch in CHAR_DENSITY:
            dens[ch] = CHAR_DENSITY[ch]
        else:
            if ch.isspace():
                dens[ch] = 0.0
            elif ch.isalpha():
                dens[ch] = 0.6
            elif ch.isdigit():
                dens[ch] = 0.5
            else:
                dens[ch] = 0.45
    return dens

def pick_char_for_tone(dens_map, target_density):
    return min(dens_map.keys(), key=lambda c: abs(dens_map[c] - target_density))

def tile_phrase_row(phrase, width, offset=0):
    if not phrase:
        return " " * width
    out = []
    p = phrase
    L = len(p)
    for i in range(width):
        out.append(p[(i + offset) % L])
    return "".join(out)

def strategy_phrase_shading(img, lum, mag, orient, phrase, params=None):
    params = params or {}
    mode = params.get('mode', 'sampled')
    h, w = lum.shape
    phrase_chars = normalize_phrase(phrase)
    dens_map = phrase_char_density_map(phrase_chars)
    dithered = floyd_steinberg_dither(lum)
    rows = []
    if mode == 'tiled':
        for y in range(h):
            base = tile_phrase_row(phrase, w, offset=y)
            row_chars = list(base)
            for x in range(w):
                target = dithered[y,x] / 255.0
                ch = row_chars[x]
                if ch not in dens_map:
                    row_chars[x] = pick_char_for_tone(dens_map, target)
                else:
                    if abs(dens_map[ch] - target) > 0.35:
                        row_chars[x] = pick_char_for_tone(dens_map, target)
            rows.append("".join(row_chars))
    else:
        for y in range(h):
            row_chars = []
            for x in range(w):
                target = dithered[y,x] / 255.0
                ch = pick_char_for_tone(dens_map, target)
                row_chars.append(ch)
            rows.append("".join(row_chars))
    return rows

def strategy_phrase_edge(img, lum, mag, orient, phrase, params=None):
    params = params or {}
    pct = params.get('edge_pct', 85)
    sparsity = params.get('sparsity', 0.6)
    h, w = lum.shape
    phrase_chars = normalize_phrase(phrase)
    dens_map = phrase_char_density_map(phrase_chars)
    thresh = np.percentile(mag, pct)
    coords = np.argwhere(mag >= thresh)
    coords = sorted(coords, key=lambda c: -mag[c[0], c[1]])
    keep_n = max(1, int(len(coords) * sparsity))
    keep = set((int(c[0]), int(c[1])) for c in coords[:keep_n])
    line_glyphs = [c for c in phrase_chars if c in "-_|/\\—–"]
    rows = []
    for y in range(h):
        row_chars = []
        for x in range(w):
            if (y,x) in keep:
                if line_glyphs:
                    ang = orient[y,x]
                    if -0.25 < ang < 0.25:
                        prefer = '-' if '-' in line_glyphs else line_glyphs[0]
                    elif 1.2 < ang < 1.8 or -1.8 < ang < -1.2:
                        prefer = '|' if '|' in line_glyphs else line_glyphs[0]
                    elif ang >= 0.25 and ang < 1.2:
                        prefer = '/' if '/' in line_glyphs else line_glyphs[0]
                    elif ang <= -0.25 and ang > -1.2:
                        prefer = '\\' if '\\' in line_glyphs else line_glyphs[0]
                    else:
                        prefer = line_glyphs[0]
                    ch = prefer
                else:
                    ch = max(phrase_chars, key=lambda c: dens_map.get(c, 0.5))
            else:
                ch = min(phrase_chars, key=lambda c: dens_map.get(c, 0.5))
            row_chars.append(ch)
        rows.append("".join(row_chars))
    return rows

def strategy_phrase_hybrid(img, lum, mag, orient, phrase, params=None):
    edge_rows = strategy_phrase_edge(img, lum, mag, orient, phrase, params)
    shade_rows = strategy_phrase_shading(img, lum, mag, orient, phrase, params)
    merged = []
    for er, sr in zip(edge_rows, shade_rows):
        row = ''.join(e if e != ' ' else s for e, s in zip(er, sr))
        merged.append(row)
    return merged

def strategy_phrase_readable(img, lum, mag, orient, phrase, params=None):
    params = params or {}
    importance_pct = params.get('importance_pct', 80)
    h, w = lum.shape
    thresh = np.percentile(mag, importance_pct)
    rows = []
    for y in range(h):
        base = tile_phrase_row(phrase, w, offset=y)
        row_chars = list(base)
        for x in range(w):
            if mag[y,x] < thresh:
                if random.random() < 0.6:
                    row_chars[x] = ' '
        rows.append("".join(row_chars))
    return rows

# Register into STRATEGIES if available
try:
    STRATEGIES['phrase_shading'] = lambda img, lum, mag, orient, params=None: strategy_phrase_shading(img, lum, mag, orient, params.get('phrase','') if params else '')
    STRATEGIES['phrase_edge'] = lambda img, lum, mag, orient, params=None: strategy_phrase_edge(img, lum, mag, orient, params.get('phrase','') if params else '')
    STRATEGIES['phrase_hybrid'] = lambda img, lum, mag, orient, params=None: strategy_phrase_hybrid(img, lum, mag, orient, params.get('phrase','') if params else '')
    STRATEGIES['phrase_readable'] = lambda img, lum, mag, orient, params=None: strategy_phrase_readable(img, lum, mag, orient, params.get('phrase','') if params else '')
except Exception:
    pass
