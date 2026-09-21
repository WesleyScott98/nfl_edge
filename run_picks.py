"""Full-slate picks from FanDuel via The Odds API.

  export ODDS_API_KEY=...
  python run_picks.py --week 3                                  # every game
  python run_picks.py --week 3 --games GB@CHI NYG@LA --out "Puka Nacua" --questionable "Davante Adams=0.85"
  python run_picks.py --week 3 --teaser-price -130 --csv week3.csv
"""
import argparse

import picks
from game import Model

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--week", type=int, required=True)
ap.add_argument("--games", nargs="*", help="AWAY@HOME")
ap.add_argument("--out", nargs="*", default=[])
ap.add_argument("--questionable", nargs="*", default=[])
ap.add_argument("--teaser-price", type=int, default=-120)
ap.add_argument("--csv")
a = ap.parse_args()
m = Model(a.season)
games = [tuple(g.split("@")) for g in a.games] if a.games else None
q = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.questionable}
out = picks.run_slate(m, a.week, games=games, out_names=a.out, questionable=q, teaser_price=a.teaser_price)
picks.show(out)
if a.csv:
    out["all_priced"].drop(columns=["leg"]).to_csv(a.csv, index=False)
