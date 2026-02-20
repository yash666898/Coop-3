# orchestrator.py
from core import luminance_map, edge_map, floyd_steinberg_dither, ansi_color
from strategies import STRATEGIES as BASE_STRATEGIES
from evaluator import evaluate_candidate_grid

# Combine base strategies and phrase strategies (phrase_strategies registers into BASE_STRATEGIES if imported)
STRATEGIES = BASE_STRATEGIES

def build_analysis(img):
    lum = luminance_map(img)
    mag, orient = edge_map(lum)
    dithered = floyd_steinberg_dither(lum)
    return {"lum": lum, "mag": mag, "orient": orient, "dither": dithered}

def generate_and_select_best(img, strategies=None, weights=None, adaptive=True):
    if strategies is None:
        strategies = list(STRATEGIES.keys())
    analysis = build_analysis(img)
    candidates = {}
    for name in strategies:
        func = STRATEGIES[name]
        # allow passing params via a dict in future; for now call with default params
        rows = func(img, analysis['lum'], analysis['mag'], analysis['orient'], params={})
        metrics = evaluate_candidate_grid(analysis['lum'], analysis['mag'], rows, weights=weights)
        candidates[name] = {"rows": rows, "metrics": metrics}
    best_name = max(candidates.keys(), key=lambda n: candidates[n]['metrics']['score'])
    best = candidates[best_name]
    if adaptive and best['metrics']['score'] < 0.6:
        for sname in ['edge', 'hybrid', 'high_detail']:
            if sname not in STRATEGIES:
                continue
            for pct in [80, 85, 90]:
                params = {'edge_pct': pct, 'sparsity': 0.5}
                rows = STRATEGIES[sname](img, analysis['lum'], analysis['mag'], analysis['orient'], params=params)
                metrics = evaluate_candidate_grid(analysis['lum'], analysis['mag'], rows, weights=weights)
                if metrics['score'] > best['metrics']['score']:
                    best = {"rows": rows, "metrics": metrics}
                    candidates[f"{sname}_pct{pct}"] = {"rows": rows, "metrics": metrics}
    return best, candidates

def rows_to_ansi(img, rows, ansi_color_func=ansi_color):
    h, w, _ = img.shape
    out_lines = []
    for y in range(min(h, len(rows))):
        row = rows[y]
        line = []
        for x in range(min(w, len(row))):
            ch = row[x]
            r,g,b = img[y,x]
            line.append(ansi_color_func(int(r), int(g), int(b), ch))
        out_lines.append("".join(line))
    return "\n".join(out_lines)
