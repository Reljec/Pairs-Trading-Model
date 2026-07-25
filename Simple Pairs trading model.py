import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from hmmlearn.hmm import GaussianHMM
from itertools import combinations
from datetime import datetime, timedelta

starting_balance = 250000
max_position_pct = 0.15
history_days = 365
training_split = 0.75
corr_min = 0.75
corr_max = 0.95
coint_pvalue = 0.1
zscore_window = 15
entry_z = 1.75
exit_z = 0.5
stop_z = 3.5
cost_per_leg_pct = 0.10
short_borrow_annual_pct = 1.00
mc_simulations = 1000
mc_days = 30
mc_seed = 42
benchmark_ticker = "^GSPC"

tickers = ['LLOY.L','BARC.L','NWG.L','STAN.L','HSBA.L','LGEN.L','AV.L','PRU.L',
           'BP.L','SHEL.L','SSE.L','NG.L','SVT.L','UU.L',
           'AZN.L','GSK.L','ULVR.L','DGE.L','IMB.L','BATS.L',
           'RIO.L','AAL.L','ANTO.L','TSCO.L','SBRY.L','MKS.L',
           'VOD.L','BT-A.L','BDEV.L','PSN.L','TW.L','BKG.L',
           'BA.L','RR.L','AUTO.L','EXPN.L','REL.L','LSEG.L',
           'SGRO.L','LAND.L','BLND.L','IAG.L','EZJ.L','IHG.L']

end_date = datetime.today()
start_date = end_date - timedelta(days=history_days)

raw_data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
prices = raw_data['Close']
prices = prices.dropna(axis=1, how='all')
prices = prices.dropna(thresh=int(len(prices.columns)*0.5))

print(prices.shape)

split_point = int(len(prices) * training_split)
train_prices = prices.iloc[:split_point]
live_prices = prices.iloc[split_point:]

returns = np.log(train_prices / train_prices.shift(1)).dropna()
corr_matrix = returns.corr()

good_pairs = []
ticker_list = corr_matrix.columns.tolist()

for i in range(len(ticker_list)):
    for j in range(i + 1, len(ticker_list)):
        c = corr_matrix.iloc[i, j]
        if c >= corr_min and c <= corr_max:
            good_pairs.append((ticker_list[i], ticker_list[j], c))

print("pairs after correlation filter:", len(good_pairs))

final_pairs = []

for pair in good_pairs:
    a = pair[0]
    b = pair[1]
    corr_val = pair[2]

    pair_data = pd.concat([train_prices[a], train_prices[b]], axis=1).dropna()
    pair_data.columns = ['a', 'b']

    if len(pair_data) < 60:
        continue

    score, pvalue, crit = coint(pair_data['a'], pair_data['b'])

    if pvalue < coint_pvalue:
        x = add_constant(pair_data['b'])
        model = OLS(pair_data['a'], x).fit()
        hedge_ratio = model.params['b']

        final_pairs.append({
            'a': a,
            'b': b,
            'corr': corr_val,
            'pvalue': pvalue,
            'hedge_ratio': hedge_ratio
        })

print("cointegrated pairs:", len(final_pairs))

bench_raw = yf.download(benchmark_ticker, period='1y', auto_adjust=True, progress=False)
bench_prices = bench_raw['Close'].squeeze()

all_trades = []
all_daily_pnl = []

for pair_num in range(len(final_pairs)):

    pair = final_pairs[pair_num]
    a = pair['a']
    b = pair['b']

    df = pd.concat([live_prices[a], live_prices[b]], axis=1).dropna()
    df.columns = ['price_a', 'price_b']

    if len(df) < zscore_window + 2:
        continue

    hedge_ratios = []
    for i in range(len(df)):
        if i < zscore_window - 1:
            hedge_ratios.append(pair['hedge_ratio'])
            continue
        window_a = df['price_a'].iloc[i - zscore_window + 1: i + 1]
        window_b = df['price_b'].iloc[i - zscore_window + 1: i + 1]
        x = add_constant(window_b)
        model = OLS(window_a, x).fit()
        hedge_ratios.append(model.params.iloc[1])

    df['hedge_ratio'] = hedge_ratios
    df['spread'] = df['price_a'] - df['hedge_ratio'] * df['price_b']
    df['spread_mean'] = df['spread'].rolling(zscore_window).mean()
    df['spread_std'] = df['spread'].rolling(zscore_window).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df = df.dropna()

    position = 0
    positions = []
    for z in df['zscore']:
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        elif position == 1:
            if z >= exit_z or z > stop_z:
                position = 0
        elif position == -1:
            if z <= exit_z or z < -stop_z:
                position = 0
        positions.append(position)

    df['position'] = positions

    trade_size = starting_balance * max_position_pct
    cost_per_leg = cost_per_leg_pct / 100
    borrow_daily = short_borrow_annual_pct / 100 / 252

    entry_pos = 0
    entry_spread = None
    entry_date = None

    for i in range(len(df)):
        pos_today = df['position'].iloc[i]

        if entry_pos == 0 and pos_today != 0:
            entry_pos = pos_today
            entry_spread = df['spread'].iloc[i]
            entry_date = df.index[i]

        elif entry_pos != 0 and pos_today == 0:
            exit_spread = df['spread'].iloc[i]
            exit_date = df.index[i]

            spread_change = entry_pos * (exit_spread - entry_spread)
            if abs(entry_spread) > 0:
                pct_return = spread_change / abs(entry_spread)
                gross_pnl = pct_return * trade_size
            else:
                gross_pnl = 0

            holding_days = (exit_date - entry_date).days
            legs_cost = 4 * cost_per_leg * trade_size
            if entry_pos == -1:
                borrow_cost = borrow_daily * holding_days * trade_size
            else:
                borrow_cost = 0
            total_cost = legs_cost + borrow_cost
            net_pnl = gross_pnl - total_cost

            if entry_pos == 1:
                direction = 'Long Spread'
            else:
                direction = 'Short Spread'

            all_trades.append({
                'pair': a + " / " + b,
                'direction': direction,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'holding_days': holding_days,
                'gross_pnl': round(gross_pnl, 2),
                'cost': round(total_cost, 2),
                'pnl': round(net_pnl, 2)
            })

            all_daily_pnl.append((exit_date, net_pnl))
            entry_pos = 0

    trade_count = 0
    for t in all_trades:
        if t['pair'] == a + " / " + b:
            trade_count += 1
    print(a, b, "->", trade_count, "trades")

trade_df = pd.DataFrame(all_trades)

pnl_series = pd.Series(dtype=float)
for date, pnl in all_daily_pnl:
    if date in pnl_series.index:
        pnl_series[date] = pnl_series[date] + pnl
    else:
        pnl_series[date] = pnl

pnl_series = pnl_series.sort_index()
equity_curve = starting_balance + pnl_series.cumsum()
daily_returns = pnl_series / starting_balance

total_pnl = pnl_series.sum()
total_return = total_pnl / starting_balance
n_days = max(len(daily_returns), 1)
ann_return = (1 + total_return) ** (252 / n_days) - 1

if daily_returns.std() > 0:
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
else:
    sharpe = 0

rolling_peak = equity_curve.cummax()
drawdown = (equity_curve - rolling_peak) / rolling_peak
max_drawdown = drawdown.min()

n_trades = len(trade_df)
if n_trades > 0:
    n_wins = (trade_df['pnl'] > 0).sum()
    win_rate = n_wins / n_trades
else:
    win_rate = 0

bench_returns = bench_prices.pct_change().dropna()
bench_returns = bench_returns[bench_returns.index.isin(daily_returns.index)]
bench_total_return = (1 + bench_returns).prod() - 1

print("total pnl:", total_pnl)
print("total return:", total_return)
print("sharpe:", sharpe)
print("max drawdown:", max_drawdown)
print("win rate:", win_rate)
print("benchmark return:", bench_total_return)

np.random.seed(mc_seed)

if len(daily_returns) >= 10:
    returns_array = daily_returns.values
else:
    returns_array = np.zeros(30)

X = returns_array.reshape(-1, 1)

hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=300, random_state=mc_seed)
hmm_model.fit(X)

hidden_states = hmm_model.predict(X)

vols = [np.sqrt(hmm_model.covars_[0][0][0]), np.sqrt(hmm_model.covars_[1][0][0])]
order = np.argsort(vols)

means_sorted = hmm_model.means_[order]
covars_sorted = hmm_model.covars_[order]
transmat_sorted = hmm_model.transmat_[np.ix_(order, order)]

last_state = hidden_states[-1]
start_regime = int(np.where(order == last_state)[0][0])

sim_paths = np.zeros((mc_simulations, mc_days))

for sim in range(mc_simulations):
    current_regime = start_regime
    cum_return = 0
    for day in range(mc_days):
        current_regime = int(np.random.choice(2, p=transmat_sorted[current_regime]))
        mu = means_sorted[current_regime][0]
        sigma = np.sqrt(covars_sorted[current_regime][0][0])
        r = np.random.normal(mu, sigma)
        cum_return = (1 + cum_return) * (1 + r) - 1
        sim_paths[sim, day] = cum_return

final_returns = sim_paths[:, -1]
p5 = np.percentile(final_returns, 5)
p50 = np.percentile(final_returns, 50)
p95 = np.percentile(final_returns, 95)
pct_profitable = (final_returns > 0).mean()

print("monte carlo p5:", p5)
print("monte carlo p50:", p50)
print("monte carlo p95:", p95)
print("pct profitable:", pct_profitable)

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("")
print("PAIRS FOUND")
if len(final_pairs) > 0:
    pairs_df = pd.DataFrame(final_pairs)
    pairs_df['corr'] = pairs_df['corr'].round(4)
    pairs_df['pvalue'] = pairs_df['pvalue'].round(5)
    pairs_df['hedge_ratio'] = pairs_df['hedge_ratio'].round(4)
    print(pairs_df.to_string(index=False))
else:
    print("no pairs passed the filters")

print("")
print("TRADE LOG")
if not trade_df.empty:
    trade_df_display = trade_df.copy()
    trade_df_display['entry_date'] = trade_df_display['entry_date'].dt.strftime('%Y-%m-%d')
    trade_df_display['exit_date'] = trade_df_display['exit_date'].dt.strftime('%Y-%m-%d')
    print(trade_df_display.to_string(index=False))
else:
    print("no trades were generated")

print("")
print("PERFORMANCE SUMMARY")
print("total pnl:", round(total_pnl, 2))
print("total return:", round(total_return * 100, 2), "%")
print("annualised return:", round(ann_return * 100, 2), "%")
print("sharpe ratio:", round(sharpe, 3))
print("max drawdown:", round(max_drawdown * 100, 2), "%")
print("win rate:", round(win_rate * 100, 2), "%")
print("number of trades:", n_trades)
print("sp500 return:", round(bench_total_return * 100, 2), "%")

print("")
print("MONTE CARLO RESULTS")
print("5th percentile:", round(p5 * 100, 2), "%")
print("median:", round(p50 * 100, 2), "%")
print("95th percentile:", round(p95 * 100, 2), "%")
print("percent profitable paths:", round(pct_profitable * 100, 2), "%")

print("")
print("done")