# li.py
import sys
from core import load_and_resize_image, ansi_color, luminance_map, edge_map, select_char
from orchestrator import generate_and_select_best, rows_to_ansi
from evaluator import evaluate_similarity
from strategies import STRATEGIES as STRATEGY_REGISTRY
import phrase_strategies  # ensures phrase strategies are registered

def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image path: ").strip()
    img = load_and_resize_image(img_path)

    while True:
        choice = input(
            "\nChoose rendering mode:\n"
            "1. Grayscale ASCII (with dithering)\n"
            "2. Color ASCII (ANSI TrueColor)\n"
            "3. Multi-layer Blending\n"
            "4. Evaluate Perceptual Similarity\n"
            "5. Exit\n"
            "6. Auto-select best perceptual rendering\n"
            "7. Phrase-driven rendering (use a phrase)\n"
            "Enter choice [1-7]: "
        ).strip()

        if choice == "1":
            from strategies import strategy_shading_focused
            rows = strategy_shading_focused(img, luminance_map(img), *edge_map(luminance_map(img)))
            print("\n".join(rows))
        elif choice == "2":
            lum = luminance_map(img)
            mag, orient = edge_map(lum)
            h, w, _ = img.shape
            out_lines = []
            for y in range(h):
                line = []
                for x in range(w):
                    region = 'shading'
                    ch = select_char(region, lum[y,x], orient[y,x])
                    r,g,b = img[y,x]
                    line.append(ansi_color(int(r), int(g), int(b), ch))
                out_lines.append("".join(line))
            print("\n".join(out_lines))
        elif choice == "3":
            from strategies import strategy_hybrid
            rows = strategy_hybrid(img, luminance_map(img), *edge_map(luminance_map(img)))
            print("\n".join(rows))
        elif choice == "4":
            from strategies import strategy_shading_focused
            rows = strategy_shading_focused(img, luminance_map(img), *edge_map(luminance_map(img)))
            score = evaluate_similarity(img, "\n".join(rows))
            print(f"SSIM similarity score: {score:.3f}")
        elif choice == "6":
            best, all_cands = generate_and_select_best(img)
            print("\n--- Best candidate metrics ---")
            print(f"Score: {best['metrics']['score']:.3f}")
            print(f"SSIM: {best['metrics']['ssim']:.3f}")
            print(f"Edge overlap: {best['metrics']['edge']:.3f}")
            print(f"Noise penalty: {best['metrics']['noise_penalty']:.3f}")
            print(f"Density penalty: {best['metrics']['density_penalty']:.3f}")
            ansi = rows_to_ansi(img, best['rows'])
            print(ansi)
            print("\nOther candidates summary:")
            for name, c in all_cands.items():
                print(f"{name}: score={c['metrics']['score']:.3f}, ssim={c['metrics']['ssim']:.3f}, edge={c['metrics']['edge']:.3f}")
        elif choice == "7":
            phrase = input("Enter phrase to render with: ").rstrip("\n")
            if not phrase:
                print("Empty phrase; aborting.")
                continue
            # use phrase strategies if registered
            phrase_strats = [k for k in STRATEGY_REGISTRY.keys() if k.startswith('phrase')]
            if not phrase_strats:
                # fallback: call functions directly from phrase_strategies
                from phrase_strategies import strategy_phrase_shading, strategy_phrase_edge, strategy_phrase_hybrid, strategy_phrase_readable
                phrase_strats = ['phrase_shading','phrase_edge','phrase_hybrid','phrase_readable']
                funcs = {
                    'phrase_shading': lambda: strategy_phrase_shading(img, luminance_map(img), *edge_map(luminance_map(img)), phrase),
                    'phrase_edge': lambda: strategy_phrase_edge(img, luminance_map(img), *edge_map(luminance_map(img)), phrase),
                    'phrase_hybrid': lambda: strategy_phrase_hybrid(img, luminance_map(img), *edge_map(luminance_map(img)), phrase),
                    'phrase_readable': lambda: strategy_phrase_readable(img, luminance_map(img), *edge_map(luminance_map(img)), phrase)
                }
                candidates = {}
                analysis_lum = luminance_map(img)
                mag, _ = edge_map(analysis_lum)
                for name in phrase_strats:
                    rows = funcs[name]()
                    from evaluator import evaluate_candidate_grid
                    metrics = evaluate_candidate_grid(analysis_lum, mag, rows)
                    candidates[name] = {'rows': rows, 'metrics': metrics}
            else:
                # call registered strategies via orchestrator
                from orchestrator import build_analysis
                analysis = build_analysis(img)
                candidates = {}
                for name in phrase_strats:
                    func = STRATEGY_REGISTRY[name]
                    rows = func(img, analysis['lum'], analysis['mag'], analysis['orient'], params={'phrase': phrase})
                    from evaluator import evaluate_candidate_grid
                    metrics = evaluate_candidate_grid(analysis['lum'], analysis['mag'], rows)
                    candidates[name] = {'rows': rows, 'metrics': metrics}
            best_name = max(candidates.keys(), key=lambda n: candidates[n]['metrics']['score'])
            best = candidates[best_name]
            print(f"Best phrase strategy: {best_name} (score={best['metrics']['score']:.3f})")
            print(rows_to_ansi(img, best['rows']))
        elif choice == "5":
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()

