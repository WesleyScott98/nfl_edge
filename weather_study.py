import sys; sys.path.insert(0, '/home/claude/nfl_edge')
import numpy as np, pandas as pd, data
seasons = [2021, 2022, 2023, 2024, 2025]
pbp = data.pbp(seasons)
pbp = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & (pbp['two_point_attempt'] != 1)].copy()
pbp['pass_play'] = ((pbp['pass_attempt'] == 1) | (pbp['sack'] == 1)).astype(float)
pbp['neutral'] = pbp['wp'].between(0.2, 0.8) & (pbp['down'] <= 3) & (pbp['half_seconds_remaining'] > 120)
pbp['pyds'] = pbp['yards_gained'] * pbp['pass_play']
pbp['ryds'] = pbp['yards_gained'] * (1 - pbp['pass_play'])
pbp['tgt'] = ((pbp['pass_attempt'] == 1) & (pbp['sack'] != 1)).astype(float)
g = pbp.groupby(['season', 'game_id', 'posteam', 'defteam']).agg(
    plays=('pass_play', 'size'), pp=('pass_play', 'sum'), pyds=('pyds', 'sum'), ryds=('ryds', 'sum'),
    tgt=('tgt', 'sum'), cmp=('complete_pass', 'sum'),
    npass=('pass_play', lambda s: s[pbp.loc[s.index, 'neutral']].sum()),
    nplays=('neutral', 'sum')).reset_index()
sch = pd.concat([data.schedules(s) for s in seasons])[['game_id', 'roof', 'temp', 'wind', 'total_line', 'spread_line', 'home_team']]
wtext = pbp.groupby('game_id')['weather'].first().rename('weather')
g = g.merge(sch, on='game_id').merge(wtext, on='game_id', how='left')
g['outdoor'] = g['roof'].isin(['outdoors', 'open'])
g['wind_x'] = np.where(g['outdoor'], g['wind'].fillna(g['wind'].median()), 0.0)
g['cold'] = (g['outdoor'] & (g['temp'] < 32)).astype(float)
wtxt = g['weather'].fillna('').str.lower()
g['precip'] = (g['outdoor'] & wtxt.str.contains('rain|snow|shower|drizzle|sleet')).astype(float)
g['team_spread'] = np.where(g['posteam'] == g['home_team'], g['spread_line'], -g['spread_line'])
g['pass_rate'] = g['pp'] / g['plays']
g['npass_rate'] = g['npass'] / g['nplays'].replace(0, np.nan)
g['pass_share'] = g['pyds'] / (g['pyds'] + g['ryds']).replace(0, np.nan)
g['cmp_rate'] = g['cmp'] / g['tgt'].replace(0, np.nan)
g['ypt'] = g['pyds'] / g['tgt'].replace(0, np.nan)
# remove team-season offense and defense means (two-way fixed effects, one pass)
def resid(col):
    x = g[col] - g.groupby(['season', 'posteam'])[col].transform('mean')
    return x - x.groupby([g['season'], g['defteam']]).transform('mean')
X = np.column_stack([np.ones(len(g)), g['wind_x'], g['cold'], g['precip'], g['total_line'] - 45, g['team_spread']])
names = ['const', 'wind (per mph)', 'cold <32F', 'rain/snow', 'total (per pt)', 'team spread']
out = {}
for col in ['npass_rate', 'pass_share', 'cmp_rate', 'ypt']:
    y = resid(col).values; m = ~np.isnan(y) & ~np.isnan(X).any(1)
    b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    e = y[m] - X[m] @ b; s2 = e @ e / (m.sum() - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X[m].T @ X[m])))
    out[col] = dict(zip(names, zip(b, se)))
    print(f'\n{col}  (n={m.sum()})'); [print(f'   {n:16s} {bb:+.4f}  (se {ss:.4f})  t={bb/ss:+.1f}') for n, bb, ss in zip(names, b, se) if n != 'const']
print('\nwindy outdoor games (>=15 mph):', int((g['wind_x'] >= 15).sum() / 2), 'of', int(len(g) / 2))
# questionable -> played?
inj = pd.concat([data.injuries(s) for s in [2024, 2025]])
inj = inj[inj['position'].isin(['QB', 'RB', 'WR', 'TE']) & inj['report_status'].isin(['Questionable', 'Doubtful', 'Out'])]
sn = data.snaps([2024, 2025]); sn = sn[sn['offense_snaps'] > 0]
played = set(zip(sn['season'], sn['week'], sn['gsis_id']))
inj['played'] = [(s, w, i) in played for s, w, i in zip(inj['season'], inj['week'], inj['gsis_id'])]
print('\nP(played | status), skill positions 2024-25:'); print(inj.groupby('report_status')['played'].agg(['mean', 'size']).round(3))
