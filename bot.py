#!/usr/bin/env python3
"""
SOLUSDC 1-Minute RSI(28)/EMA(13) Crossover Backtest
=====================================================

Strategy
--------
- Indicator: RSI(28) and a 13-period EMA of that RSI line.
- Entry (LONG ONLY): when RSI(28) crosses ABOVE its 13 EMA -> BUY at the
  close of the signal candle (next-bar-open execution is also supported,
  see EXECUTION_MODE below).
- Stop Loss: low of the candle immediately BEFORE the signal/trigger candle.
- Take Profit: entry + 2.2 * (entry - stop_loss)   [Risk:Reward = 1 : 2.2]
- Costs: 0.06% total round-trip cost (entry + exit combined), applied as a
  simple percentage drag on the trade's gross return.
- Position sizing: FULL ACCOUNT, COMPOUNDING (spot-style, no leverage,
  no shorting, one position open at a time). Every trade risks 100% of
  current equity (you chose this explicitly to see compounding behavior --
  see the warning printed at runtime).
- Account starts at $100.

Data
----
Binance public REST API, no API key required:
  GET https://api.binance.com/api/v3/klines
Symbol: SOLUSDC, Interval: 1m, Range: last 365 days (auto, paginated).
Data is cached to a local CSV (solusdc_1m_1y.csv) so re-runs don't
re-download ~525,600 candles every time.

Dependencies
------------
This script installs any missing dependencies itself on first run
(pandas, numpy, requests). No manual `pip install` needed.

Usage
-----
    python3 solusdc_rsi_ema_backtest.py

Optional flags:
    --refresh           Force re-download of data (ignore cache)
    --execution close   Enter at signal-candle CLOSE (default)
    --execution next    Enter at NEXT candle OPEN (more realistic, avoids lookahead)
"""

import sys
import subprocess
import importlib
import argparse
import os
import time
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 0. SELF-INSTALLING DEPENDENCIES
# ---------------------------------------------------------------------------
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "requests": "requests",
}

def ensure_dependencies():
    missing = []
    for import_name, pip_name in REQUIRED.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[setup] Installing missing dependencies: {', '.join(missing)} ...")
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        # --break-system-packages is needed on some externally-managed Python
        # installs (PEP 668, e.g. recent Debian/Ubuntu). Try normal install
        # first, fall back if it fails.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            cmd_alt = cmd + ["--break-system-packages"]
            result2 = subprocess.run(cmd_alt, capture_output=True, text=True)
            if result2.returncode != 0:
                print("[setup] ERROR installing dependencies:")
                print(result.stderr)
                print(result2.stderr)
                sys.exit(1)
        print("[setup] Dependencies installed.")

ensure_dependencies()

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
SYMBOL = "SOLUSDC"
INTERVAL = "3m"
DAYS_BACK = 365
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solusdc_1m_1y.csv")

RSI_LEN = 28
EMA_LEN = 13
RR_MULTIPLE = 2.2          # target = entry + RR_MULTIPLE * risk
ROUND_TRIP_COST_PCT = 0.06 / 100.0   # 0.06% total (entry+exit combined)
STARTING_EQUITY = 100.0
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max candles per request


# ---------------------------------------------------------------------------
# 2. DATA DOWNLOAD (Binance public API, paginated, cached)
# ---------------------------------------------------------------------------
def download_binance_klines(symbol=SYMBOL, interval=INTERVAL, days_back=DAYS_BACK):
    """
    Downloads `days_back` days of historical klines from Binance's public
    REST API (no API key needed) and returns a clean OHLCV DataFrame.
    Paginates in chunks of MAX_LIMIT candles, respecting rate limits.
    """
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    all_rows = []
    cur_start = start_time
    session = requests.Session()
    request_count = 0

    print(f"[data] Downloading {days_back}d of {interval} {symbol} klines from Binance...")

    while cur_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "limit": MAX_LIMIT,
        }
        for attempt in range(5):
            try:
                resp = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
                if resp.status_code == 200:
                    break
                elif resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"[data] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[data] HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"[data] Request error: {e}, retrying...")
                time.sleep(3)
        else:
            raise RuntimeError("Failed to fetch klines after multiple retries.")

        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        cur_start = last_open_time + 1  # next page starts right after last candle
        request_count += 1

        if request_count % 20 == 0:
            pct = min(100.0, (cur_start - start_time) / (end_time - start_time) * 100)
            print(f"[data] ...{len(all_rows):,} candles fetched ({pct:.1f}%)")

        # Be polite to the API (well under the 1200 req/min weight limit)
        time.sleep(0.05)

        if len(batch) < MAX_LIMIT:
            # short batch usually means we've caught up to "now"
            if cur_start >= end_time:
                break

    print(f"[data] Download complete: {len(all_rows):,} candles.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


def load_data(refresh=False):
    if (not refresh) and os.path.exists(CACHE_FILE):
        print(f"[data] Loading cached data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=["open_time", "close_time"])
        age_days = (datetime.now(timezone.utc) - df["open_time"].max().tz_convert("UTC")).total_seconds() / 86400
        print(f"[data] Cached data spans {df['open_time'].min()} -> {df['open_time'].max()} "
              f"({len(df):,} rows, newest candle is {age_days:.1f} days old)")
        return df

    df = download_binance_klines()
    df.to_csv(CACHE_FILE, index=False)
    print(f"[data] Saved cache to {CACHE_FILE}")
    return df


# ---------------------------------------------------------------------------
# 3. INDICATORS
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)          # no losses -> RSI 100
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat -> 50
    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], RSI_LEN)
    df["rsi_ema"] = df["rsi"].ewm(span=EMA_LEN, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# 4. BACKTEST ENGINE
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, execution_mode: str = "close"):
    """
    execution_mode:
      'close' -> enter at the close of the signal candle (the candle where
                 the crossover is confirmed). Simpler, but technically
                 assumes you can transact exactly at the closing price.
      'next'  -> enter at the OPEN of the candle AFTER the signal candle.
                 More realistic (no lookahead), since the signal candle's
                 close isn't known until it closes.

    Rules (long only):
      - Signal: rsi crosses above rsi_ema on the signal candle (rsi[i-1] <= ema[i-1]
        and rsi[i] > ema[i]).
      - Entry price: signal candle close ('close' mode) or next candle open ('next' mode).
      - Stop loss: low of the candle immediately BEFORE the signal candle (index i-1).
      - Take profit: entry + 2.2 * (entry - stop_loss).
      - Only one position open at a time (no pyramiding/overlapping trades).
      - Exit check uses high/low of subsequent candles; if both TP and SL are
        touched within the same candle, SL is assumed hit first (conservative).
      - Round trip cost (0.06%) deducted from gross trade return.
      - Position size = 100% of current equity (full compounding, spot-style,
        no leverage).
    """
    n = len(df)
    rsi = df["rsi"].values
    ema = df["rsi_ema"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    openp = df["open"].values
    times = df["open_time"].values

    trades = []
    equity = STARTING_EQUITY
    equity_curve = [(times[0], equity)]

    in_position = False
    i = max(RSI_LEN, EMA_LEN) + 2  # warmup

    while i < n - 1:
        if not in_position:
            # crossover check: rsi was <= ema, now > ema
            if (not np.isnan(rsi[i - 1])) and (not np.isnan(ema[i - 1])) and \
               (not np.isnan(rsi[i])) and (not np.isnan(ema[i])):
                crossed_up = (rsi[i - 1] <= ema[i - 1]) and (rsi[i] > ema[i])
            else:
                crossed_up = False

            if crossed_up:
                signal_idx = i
                sl_price = low[signal_idx - 1]  # previous (trigger-1) candle low

                if execution_mode == "close":
                    entry_idx = signal_idx
                    entry_price = close[signal_idx]
                else:  # 'next'
                    entry_idx = signal_idx + 1
                    if entry_idx >= n:
                        break
                    entry_price = openp[entry_idx]

                risk = entry_price - sl_price
                if risk <= 0:
                    # invalid setup (SL above/at entry) -> skip this signal
                    i += 1
                    continue

                tp_price = entry_price + RR_MULTIPLE * risk

                # walk forward from the bar after entry to find exit
                exit_idx = None
                exit_price = None
                exit_reason = None
                j = entry_idx + 1
                while j < n:
                    hit_sl = low[j] <= sl_price
                    hit_tp = high[j] >= tp_price
                    if hit_sl and hit_tp:
                        # conservative assumption: SL hit first
                        exit_idx, exit_price, exit_reason = j, sl_price, "SL"
                        break
                    elif hit_sl:
                        exit_idx, exit_price, exit_reason = j, sl_price, "SL"
                        break
                    elif hit_tp:
                        exit_idx, exit_price, exit_reason = j, tp_price, "TP"
                        break
                    j += 1

                if exit_idx is None:
                    # ran out of data before resolving -> close at last available price
                    exit_idx = n - 1
                    exit_price = close[exit_idx]
                    exit_reason = "EOD_FORCE_CLOSE"

                gross_ret = (exit_price - entry_price) / entry_price
                net_ret = gross_ret - ROUND_TRIP_COST_PCT  # full round-trip cost on the trade

                pnl_dollars = equity * net_ret
                equity_before = equity
                equity = equity + pnl_dollars

                trades.append({
                    "signal_time": times[signal_idx],
                    "entry_time": times[entry_idx],
                    "exit_time": times[exit_idx],
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "risk_pct": risk / entry_price,
                    "gross_return_pct": gross_ret,
                    "net_return_pct": net_ret,
                    "equity_before": equity_before,
                    "equity_after": equity,
                    "pnl_dollars": pnl_dollars,
                    "bars_held": exit_idx - entry_idx,
                })
                equity_curve.append((times[exit_idx], equity))

                i = exit_idx + 1  # no overlapping trades
                continue
        i += 1

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
    return trades_df, equity_df


# ---------------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------------
def max_drawdown(equity_series: pd.Series):
    running_max = equity_series.cummax()
    dd = (equity_series - running_max) / running_max
    return dd.min(), dd.idxmin()


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, starting_equity: float):
    if trades_df.empty:
        print("No trades were generated. Try a longer period or check the data.")
        return {}

    n_trades = len(trades_df)
    wins = trades_df[trades_df["net_return_pct"] > 0]
    losses = trades_df[trades_df["net_return_pct"] <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades * 100

    final_equity = equity_df["equity"].iloc[-1]
    total_pnl_dollars = final_equity - starting_equity
    total_return_pct = (final_equity / starting_equity - 1) * 100

    avg_win_pct = wins["net_return_pct"].mean() * 100 if n_wins > 0 else 0.0
    avg_loss_pct = losses["net_return_pct"].mean() * 100 if n_losses > 0 else 0.0
    avg_win_dollars = wins["pnl_dollars"].mean() if n_wins > 0 else 0.0
    avg_loss_dollars = losses["pnl_dollars"].mean() if n_losses > 0 else 0.0

    gross_profit = wins["pnl_dollars"].sum() if n_wins > 0 else 0.0
    gross_loss = -losses["pnl_dollars"].sum() if n_losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Expectancy (EV) per trade, in % return terms and in $ terms
    ev_pct = (win_rate / 100 * avg_win_pct) + ((1 - win_rate / 100) * avg_loss_pct)
    ev_dollars = trades_df["pnl_dollars"].mean()

    # average R multiple actually achieved (using each trade's own risk)
    trades_df = trades_df.copy()
    trades_df["r_multiple"] = trades_df["net_return_pct"] / trades_df["risk_pct"].replace(0, np.nan)
    avg_r = trades_df["r_multiple"].mean()
    expectancy_r = (win_rate / 100) * RR_MULTIPLE - (1 - win_rate / 100) * 1  # theoretical R-based EV

    largest_win = trades_df["pnl_dollars"].max()
    largest_loss = trades_df["pnl_dollars"].min()

    max_dd_pct, max_dd_time_idx = max_drawdown(equity_df["equity"])
    max_dd_time = equity_df.loc[max_dd_time_idx, "time"] if not equity_df.empty else None

    # Consecutive wins/losses
    streak = (trades_df["net_return_pct"] > 0).astype(int)
    max_consec_win = max_consec_loss = cur_win = cur_loss = 0
    for v in streak:
        if v == 1:
            cur_win += 1; cur_loss = 0
            max_consec_win = max(max_consec_win, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_consec_loss = max(max_consec_loss, cur_loss)

    avg_bars_held = trades_df["bars_held"].mean()

    # Sharpe-like ratio on per-trade returns (not annualized to a clean period
    # since trade frequency is irregular; shown as per-trade Sharpe)
    ret_std = trades_df["net_return_pct"].std()
    sharpe_per_trade = trades_df["net_return_pct"].mean() / ret_std if ret_std > 0 else np.nan

    # CAGR based on actual elapsed time of the data
    start_time = equity_df["time"].iloc[0]
    end_time = equity_df["time"].iloc[-1]
    elapsed_days = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() / 86400
    elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
    cagr = ((final_equity / starting_equity) ** (1 / elapsed_years) - 1) * 100 if elapsed_years and elapsed_years > 0 else np.nan

    metrics = {
        "Total trades": n_trades,
        "Winning trades": n_wins,
        "Losing trades": n_losses,
        "Win rate (%)": win_rate,
        "Starting equity ($)": starting_equity,
        "Final equity ($)": final_equity,
        "Total P&L ($)": total_pnl_dollars,
        "Total return (%)": total_return_pct,
        "CAGR (%) [approx, compounding]": cagr,
        "Avg win (%)": avg_win_pct,
        "Avg loss (%)": avg_loss_pct,
        "Avg win ($, at time of trade)": avg_win_dollars,
        "Avg loss ($, at time of trade)": avg_loss_dollars,
        "Largest win ($)": largest_win,
        "Largest loss ($)": largest_loss,
        "Gross profit ($, sum of winning trades)": gross_profit,
        "Gross loss ($, sum of losing trades)": gross_loss,
        "Profit factor": profit_factor,
        "Expectancy / EV per trade (%)": ev_pct,
        "Expectancy / EV per trade ($, at time of trade)": ev_dollars,
        "Theoretical R-based expectancy (R per trade)": expectancy_r,
        "Avg realized R-multiple per trade": avg_r,
        "Max drawdown (%)": max_dd_pct * 100,
        "Max drawdown date": max_dd_time,
        "Max consecutive wins": max_consec_win,
        "Max consecutive losses": max_consec_loss,
        "Avg bars held (minutes)": avg_bars_held,
        "Per-trade Sharpe (mean/std of trade returns)": sharpe_per_trade,
        "Exit reason breakdown": trades_df["exit_reason"].value_counts().to_dict(),
    }
    return metrics


def print_report(metrics: dict):
    print("\n" + "=" * 70)
    print(" BACKTEST RESULTS: SOLUSDC 1m | RSI(28)/EMA(13) crossover | Long-only")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<55}: {v:,.4f}")
        else:
            print(f"{k:<55}: {v}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SOLUSDC RSI/EMA crossover backtest")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data, ignore cache")
    parser.add_argument("--execution", choices=["close", "next"], default="close",
                         help="Entry execution mode: 'close' of signal candle or 'next' candle open")
    args = parser.parse_args()

    print("\n*** IMPORTANT: position sizing = 100% of current equity per trade ***")
    print("*** (full compounding, no leverage, spot-style, one trade at a time) ***")
    print("*** This is what you asked for, but note it means a single bad trade")
    print("*** can wipe out a large fraction of the account. Review results")
    print("*** carefully before considering this for real capital. ***\n")

    df = load_data(refresh=args.refresh)
    df = add_indicators(df)

    trades_df, equity_df = run_backtest(df, execution_mode=args.execution)

    if trades_df.empty:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades_df, equity_df, STARTING_EQUITY)
    print_report(metrics)

    out_trades = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.csv")
    trades_df.to_csv(out_trades, index=False)
    print(f"\nFull trade log saved to: {out_trades}")

    out_equity = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_curve.csv")
    equity_df.to_csv(out_equity, index=False)
    print(f"Equity curve saved to: {out_equity}")


if __name__ == "__main__":
    main()
