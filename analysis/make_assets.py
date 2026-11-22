"""analysis/make_assets.py: talk assets (COVERAGE section 6): figures F1 (sq4096), F3, F4, F6,
headline_numbers.csv and facts_sheet.txt. Run from the repo root: python3 analysis/make_assets.py
Reads results/cpu/ (including rep/) and results/gpu/; writes only to analysis/out/.
Needs pandas and matplotlib.
"""
import os

from data import aggregate, load_runs
from figures import OUT, fig_f1, fig_f3, fig_f4, fig_f6
from headline import build


def main():
    runs = load_runs()
    agg = aggregate(runs)
    fig_f1(agg)
    fig_f3(agg)
    fig_f4(agg)
    fig_f6(agg)
    numbers, facts = build(agg, runs)
    os.makedirs(OUT, exist_ok=True)
    numbers.to_csv(os.path.join(OUT, "headline_numbers.csv"), index=False)
    with open(os.path.join(OUT, "facts_sheet.txt"), "w") as f:
        f.write(facts)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
