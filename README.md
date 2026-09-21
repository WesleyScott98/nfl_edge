# nfl_edge — calibrated NFL prop & SGP model

Simulates every game 20,000 times from free nflverse data, prices props and same-game parlays
with their real correlation, compares to FanDuel, and only flags bets where the model still
beats break-even after deferring 65% to the market.

## Setup (local)
```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key        # optional; free key at the-odds-api.com (500 credits/mo)
python run_game.py --away NYG --home LA --week 2 --spread 6.5 --total 47.5 --out "Puka Nacua"
python run_game.py --away NYG --home LA --week 2 --props --csv nyg_la.csv   # needs API key
python run_game.py --away GB --home CHI --week 3 --questionable "D.J. Moore=0.8" --wind 18 --temp 38
```

## Setup (Databricks Free Edition)
1. Workspace -> Create -> Git folder (or upload this folder as workspace files).
2. Open `databricks_notebook.py` (it imports as a notebook).
3. Put your Odds API key in a secret scope (`databricks secrets create-scope nfl`, key `odds_api_key`)
   or paste it into the widget. Run all cells.
4. Optional: schedule the notebook as a Job (e.g. Sunday 11:00 ET, after inactives ~90 min pre-kick).

## What the model accounts for
- **This season vs last:** 2026 plays count fully; prior seasons fade as games are played
  (`PRIOR_SEASON_K`). Player usage is recency-weighted (half-life 4 games).
- **Roster status:** before every simulation, players whose latest official weekly roster status is
  injured reserve, commissioner exempt, suspended, released, retired or practice squad — or who
  are now on a different team — are removed automatically (e.g. Josh Jacobs, exempt list, 2026).
- **Injuries:** Out/Doubtful from the official report (Doubtful players played 0 of 104 times in
  2024-25) are removed and their volume redistributed. **Questionable** players are simulated as
  playing 56% of the time (measured, 2024-25) — override with news, e.g. `questionable={"Puka Nacua": 0.85}`.
  Their own props are priced *conditional on playing* (books void if they sit); teammates absorb
  their volume in the scenarios where they sit. Injury-shortened games are dropped from usage;
  teammate absences are adjusted for (with/without).
- **Defensive injuries:** not automatic yet — use `def_adjust={"NYG": {"def_ypt": 1.06}}` when a
  defense is missing starters (1.06 = allows 6% more yards per target).
- **Weather** (outdoor games): rain/snow, cold (<32F) and wind shift pass rate, catch rate and
  yards per target. Scoring is *not* adjusted — the closing total already prices weather.
  Observed weather for past games; Open-Meteo kickoff forecast for upcoming games
  (retractable roofs treated as closed). Override with `weather={...}` or CLI `--wind --temp --precip`.

| weather effect (2021-25, controlling for total) | size | t-stat |
|---|---|---|
| rain/snow: pass rate | -1.8 pts | -2.3 |
| rain/snow: catch rate | -2.6 pts | -3.6 |
| rain/snow: yards per target | -0.53 | -3.6 |
| wind: yards per target | -0.19 per 10 mph | -3.1 |
| cold <32F: pass rate | -3.0 pts | -3.3 |

## Defensive analysis (matchup report) — tested, used as context, not in the engine
`Model.matchup_report(away, home, week)` shows, for each defense: opponent-adjusted pass/run EPA
and league rank, man rate, two-high shell rate (Cover 2/4/6 vs Cover 0/1/3), blitz and pressure
rate, and defense-vs-position (share of targets and yards/target allowed to WR/TE/RB vs league).

All three were built into the simulator (`defense.py`, switches `USE_OPP_ADJ`, `USE_DVP`,
`USE_SHELL`, `DEF_STRENGTH`) and tested on 2025 (both halves, paired bootstrap by player-game):

| feature set | change in Brier | 95% CI |
|---|---|---|
| all three, full strength | +0.00015 (worse) | [-0.00008, +0.00034] |
| all three, half strength | +0.00001 | [-0.00010, +0.00013] |
| opp-adjusted + shells, half | -0.00002 | [-0.00009, +0.00006] |

None beats noise, so they are **off** in the probability engine. Once the model has the spread,
total, player usage and basic defensive efficiency, these add no measurable predictive value for
player props (consistent with defense-vs-position being famously noisy). Re-test mid-season.

## QB passing yards — why it's excluded from best_bets()
When the starter plays the whole game, projected passing yards are unbiased (-0.7 yds). The
+17 yd bias came entirely from the 12% of starts where the QB left early (75 yds actual vs 224
projected). Books settle QB props if he plays a snap, so this is real risk. Modelling it
explicitly (`P_QB_EXIT`) fixed the bias late in 2025 but not early, and didn't improve Brier
(calibration already absorbs it), so it's off by default. QB yardage is hard mainly because of
*whether he finishes*, which is why it stays out of the bet finder.

## Injury-shortened / snap-limited games
A regular who plays < 60% of his usual snaps (hurt mid-game, or on a pitch count returning from
injury) used to have that game dropped, which discarded real evidence (e.g. 9 targets on 52% of
snaps). Now the game is kept, shares are scaled up by (usual/actual snaps)^0.5 and the game is
down-weighted. Tested on 2025: better overall (-0.00009 Brier, not significant alone) and clearly
better for the ~28% of player-games it changes (-0.00053). Full scaling (^1.0) was worse —
snap-limited players are used on high-leverage passing downs, so per-snap rates overstate.

## Out-of-season test (2024 — never used for tuning or calibration)
| market | skill 2024 (unseen) | skill 2025 |
|---|---|---|
| receptions | 33% | 30% |
| receiving yards | 31% | 27% |
| rushing yards | 31% | 31% |
| anytime TD | 11% | 12% |
| QB passing yards | 2% | 9% |

Skill carries across seasons. Calibration drifts 2–4 pts season to season (2024 overs hit a bit
more often than 2025's calibration expected), so production calibration is fit on 2024+2025, and
the bet finder requires a 3-pt edge after shrinking toward the market.

## Bet tracking (`tracking.py`) — the only real proof of an edge
Log every bet you place, add FanDuel's price at kickoff, and grade from nflverse after games:
```python
import tracking as T
bid = T.log_bet(2026, 3, "GB@CHI", [dict(player="D.J. Moore", stat="rec_yds", side="over", line=59.5)],
                price=-115, stake=1, p_final=0.56)
T.set_close(bid, -135)   # closing price
T.grade(); T.report()    # record, ROI + 95% CI, expected vs actual profit, avg CLV, share beating close
```
Positive average closing-line value (CLV) over 50–100 bets is the fastest honest signal of an
edge; ROI needs hundreds of bets before the confidence interval means anything. Parlays with a
voided/pushed leg are marked `win*` for a manual payout check (FanDuel reprices them).
On Databricks, point `NFL_EDGE_BETS` at a Unity Catalog Volume so the log persists.

## Full-slate pick engine (`picks.py`, `run_picks.py`)
One run prices every FanDuel market for every game and returns: best singles by category
(Moneyline/Spread, Totals, Player props, Anytime TD), same-game parlays, cross-game parlays and teasers.

| market | how it's priced | what the tests said |
|---|---|---|
| main spread / total | anchored on FanDuel's own line | a stats model LOST to closing lines (47.5-48.6% ATS, 50-52% O/U, 2024-25), so no independent opinions; these only move with injury news you feed in |
| moneyline, alt spreads, alt totals, team totals | real historical margins around the spread (6,232 games, 2003-25) | beat a bell curve on ML and alt ±3/±7 out of sample; catches prices inconsistent with key numbers (15% of games land on 3, 9% on 7) |
| teasers | 6-pt legs from the same margin distribution | only +1.5..+2.5 underdogs still clear breakeven (76.5% in 2021-25 vs 73.9% needed at -120); favorite legs fell to 69.1% |
| player props, ATD, alt ladders | calibrated simulation | skill 27-33% (props), 11% (TD) in 2024 and 2025 |
| SGPs | joint probability from the simulation (correlation included) | shown with fair odds and the minimum FanDuel price to bet; FanDuel's SGP quote isn't in the API, so compare in the app |
| cross-game parlays | product of legs that are +EV on their own, one per game | FanDuel pays the straight product, so EV is exact given the leg probabilities |

A bet is flagged only if, after deferring 65% to the market, it still beats break-even by 3+ points.
Most slates will produce only a handful of flagged singles — that is the point.

## Weekly workflow
1. Update lines: pass live FanDuel spread/total (or use `--props` to pull everything).
2. After inactives, rerun with `--out` for anyone ruled out, and `snap_override` for pitch counts.
3. Read `best_bets()` — anything left there beat break-even *after* shrinking toward the market.
4. For SGPs, `edge.price_parlay(sim, legs, book_american=+550)` gives the correlated joint
   probability, fair odds, and EV vs FanDuel's quoted price.

## Files
| file | what it does |
|---|---|
| `config.py` | every modelling assumption; tuned values marked |
| `data.py` | cached nflverse loaders: pbp, snaps, injuries, schedules, FTN, NGS coverage |
| `features.py` | team pace/pass rate/EPA/funnel/man-zone/blitz; player shares with shrinkage, with/without teammate adjustment, prior-team usage |
| `sim.py` | correlated Monte Carlo game simulator |
| `game.py` | `Model` — one call to build features + simulate a game |
| `odds.py` | FanDuel via The Odds API; odds math; manual line entry |
| `edge.py` | calibrated prob vs market -> EV, fractional Kelly; SGP pricing |
| `calibrate.py` + `calibration.json` | Platt recalibration per stat |
| `defense.py` | opponent-adjusted ratings, defense-vs-position, coverage shells (report + optional engine) |
| `picks.py`, `run_picks.py` | full-slate pick engine across all markets |
| `markets.py` | key-number margin distribution, game-line pricing, teasers |
| `tracking.py` | bet log, auto-grading, ROI / CLV reporting |
| `backtest.py`, `tune.py` | leak-free backtest, calibration tables, parameter tuning |
| `weather_study.py` | regression behind the weather coefficients |

## Validation (2025 season, features built only from data before each week)
Tuned on weeks 4–11, scored on held-out weeks 12–18. Brier skill = improvement over always
predicting the base rate at that line.

| market | Brier skill | notes |
|---|---|---|
| receptions | 30% | reliable (final holdout calibrated Brier, all markets: 0.1180) |
| receiving yards | 27% | reliable |
| rushing yards | 31% | slight over-projection (~4 yds) |
| anytime TD | 12% | TDs are noisy for everyone |
| QB passing yards | 9% | see section below — **excluded from best_bets()** |

Calibrated probabilities land within ~2–3 points of actual hit rates in every 10% bucket.
Weather adjustments improved held-out Brier overall (0.1192 -> 0.1187) and in the 26 bad-weather
games (0.1158 -> 0.1146); QB passing-yard bias in those games fell from +35 to +15 yds.

## Known limitations (read before betting)
- **Players returning from injury**: partially fixed (see above). If a player is on a known snap
  limit this week, pass `snap_override={"Name": 0.6}`.
- **New arrivals**: use previous-team role at half weight; teams with many new pieces are noisier.
- **Coverage scheme**: NGS man/zone data is published with a lag (currently through 2025), so
  2026 scheme effects lean on last season. Coordinator changes aren't modelled.
- **Odds API integration is untested against the live API** (written to its documented format);
  check the first run's output carefully. Player names are matched on full name.
- Weather lowers pass rate but doesn't yet lower rushing efficiency in slop (rush yds slightly
  over-projected in bad weather). Forecast fetching is untested from here (sandbox can't reach
  Open-Meteo) — it runs on Databricks/local; otherwise pass weather manually.
- Early-season numbers lean heavily on last season (see `PRIOR_SEASON_K`).
- No model reliably beats closing lines; the market-shrinkage and Kelly cap are there on purpose.
  Re-run `backtest.py` / `tune.py` each off-season and after any change.
