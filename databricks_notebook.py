# Databricks notebook source
# MAGIC %md
# MAGIC # nfl_edge — weekly run
# MAGIC Simulates a game, prices FanDuel props, ranks +EV bets. See README for how it works and its limits.

# COMMAND ----------
# MAGIC %pip install nflreadpy polars pyarrow scipy requests

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os, sys
sys.path.insert(0, os.getcwd())                 # notebook sits in the repo folder
os.environ["NFL_EDGE_CACHE"] = "/tmp/nfl_edge_cache"
dbutils.widgets.text("away", "NYG"); dbutils.widgets.text("home", "LA")
dbutils.widgets.text("week", "2"); dbutils.widgets.text("season", "2026")
dbutils.widgets.text("spread_home", ""); dbutils.widgets.text("total", "")
dbutils.widgets.text("out", "", "Ruled out (comma-separated full names)")
dbutils.widgets.text("questionable", "", "Questionable: Name=P,Name=P (blank = injury report)")
dbutils.widgets.text("odds_api_key", "")
key = dbutils.widgets.get("odds_api_key")
if not key:
    try:
        key = dbutils.secrets.get("nfl", "odds_api_key")
    except Exception:
        key = ""
if key:
    os.environ["ODDS_API_KEY"] = key

# COMMAND ----------
import importlib, config; importlib.reload(config)
import game, edge, odds
from run_game import TEAM_NAMES
away, home = dbutils.widgets.get("away"), dbutils.widgets.get("home")
week, season = int(dbutils.widgets.get("week")), int(dbutils.widgets.get("season"))
sp, tot = dbutils.widgets.get("spread_home"), dbutils.widgets.get("total")
outs = [x.strip() for x in dbutils.widgets.get("out").split(",") if x.strip()]
m = game.Model(season)
display(m.matchup_report(away, home, week).reset_index())

# COMMAND ----------
q = {kv.split("=")[0].strip(): float(kv.split("=")[1]) for kv in dbutils.widgets.get("questionable").split(",") if "=" in kv}
# weather="auto": observed for past games, Open-Meteo kickoff forecast for upcoming outdoor games
sim = m.simulate(away, home, week, float(sp) if sp else None, float(tot) if tot else None,
                 out_names=outs, questionable=q, weather="auto")
print("Weather:", sim.weather, "| Questionable (P plays):", sim.questionable)
proj = sim.summary()
display(proj[(proj.targets_mean > 1.5) | (proj.carries_mean > 3) | (proj.pass_yds_mean > 0)].round(2))

# COMMAND ----------
# MAGIC %md ## FanDuel props -> +EV bets (needs Odds API key)

# COMMAND ----------
if key:
    ev = odds.game_lines()
    ev["h"], ev["a"] = ev["home"].map(TEAM_NAMES), ev["away"].map(TEAM_NAMES)
    eid = ev[(ev.h == home) & (ev.a == away)]["event_id"].iloc[0]
    priced = edge.price_props(sim, odds.player_props(eid))
    display(edge.best_bets(priced))
    spark.createDataFrame(priced).write.mode("append").saveAsTable("nfl_edge_priced_props")  # history for tracking CLV
else:
    print("No API key: use odds.manual_lines([...]) to enter FanDuel numbers by hand.")

# COMMAND ----------
# MAGIC %md ## Price a same-game parlay

# COMMAND ----------
legs = [dict(player="Davante Adams", stat="rec_yds", line=49.5),
        dict(player="Matthew Stafford", stat="pass_yds", line=224.5),
        dict(player="Tyler Higbee", stat="receptions", line=2.5)]
edge.price_parlay(sim, legs, book_american=None)   # put FanDuel's SGP price here to get EV

# COMMAND ----------
# MAGIC %md ## Full slate: best picks across every market (ML, spreads, totals, alt lines, team totals, props, TDs, SGPs, parlays, teasers)

# COMMAND ----------
import picks
slate = picks.run_slate(m, week, out_names=outs, questionable=q, teaser_price=-120)   # needs the Odds API key
picks.show(slate)
display(slate["best_singles"].drop(columns=["leg"]))
