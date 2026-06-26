#!/usr/bin/env python3
"""
SOL 9EMA / 9SMA Crossover — 2-Year Backtest
=============================================

Strategy under test (BUY only):
    Entry  -> 9 EMA crosses ABOVE 9 SMA (on a CLOSED candle)
              Enter at the close of the crossing candle.
    SL     -> low of the crossing ("trigger") candle
    Target -> entry + 2 * (entry - SL)      (1:2 risk:reward)
    Exit   -> whichever of SL / Target is hit first, checked candle by
              candle going forward. Opposite (bearish) crossovers are
              IGNORED as an exit reason, per spec -- a trade only closes
              on SL or Target.
    Sizing -> account starts at $1000. Each trade goes ALL-IN:
                  quantity = balance / entry_price
              0.06% round-trip fee is charged on notional value
              (applied at entry and modeled in the exit pnl).
    Only one position open at a time. While in a trade, new BUY signals
    are ignored (we don't pyramid).

Runs independently on 3m, 5m and 15m and prints/exports a metrics
report for each, plus an equity curve CSV.

------------------------------------------------------------------
DATA: This script pulls 2 years of historical klines directly from
Binance's public REST API (no key needed). It paginates in chunks of
1000 candles using startTime/endTime, respecting Binance's public rate
limits with a small delay between calls.

Run it somewhere with normal internet access to api.binance.com
(your machine, a VPS, Railway, etc).
------------------------------------------------------------------
"""

import sys
import subprocess
import importlib


# ---------------------------------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = ["requests", "pandas", "numpy"]


def ensure_dependencies():
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[setup] '{pkg}' not found, installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg]
            )
            print(f"[setup] '{pkg}' installed.")


ensure_dependencies()

import time
import math
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
SYMBOL = "SOLUSDT"
TIMEFRAMES = ["3m", "5m", "15m"]

EMA_FAST_PERIOD = 9
SMA_PERIOD = 9
EMA_TREND_PERIOD = 13   # informational only, not used as a filter here
RSI_PERIOD = 28         # informational only, not used as a filter here

LOOKBACK_DAYS = 730       # ~2 years
STARTING_BALANCE = 1000.0
ROUNDTRIP_FEE_PCT = 0.0006   # 0.06% round trip (entry+exit combined)
RISK_REWARD = 2.0            # target = entry + RR * (entry - sl)

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000

OUTPUT_DIR = "."  # change if you want CSVs elsewhere


# ---------------------------------------------------------------------------
# 2. DATA FETCH (paginated, 2 years)
# ---------------------------------------------------------------------------
def fetch_full_history(symbol, interval, lookback_days):
    """Paginate Binance klines backwards from now to `lookback_days` ago.
    Returns a pandas DataFrame, oldest -> newest, with closed candles only
    (the very last in-progress candle, if any, is dropped)."""

    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
    )

    all_rows = []
    cursor = start_time
    session = requests.Session()

    print(f"[data] Fetching {symbol} {interval} from {datetime.fromtimestamp(start_time/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(end_time/1000, tz=timezone.utc).date()} ...")

    while cursor < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": MAX_LIMIT,
        }
        resp = session.get(BINANCE_BASE_URL + KLINES_ENDPOINT, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text[:300]}")
        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]

        # Advance cursor past the last candle we got. Binance kline open
        # times are evenly spaced, so +1ms is enough to not refetch the
        # same candle, but we also break if we've reached "now".
        cursor = last_open_time + 1

        if len(batch) < MAX_LIMIT:
            break

        time.sleep(0.25)  # be polite to the public rate limit

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Drop duplicate candles that can occur at pagination boundaries
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)

    # Drop the last candle if it's still forming (close_time in the future)
    now_utc = datetime.now(timezone.utc)
    if df.iloc[-1]["close_time"].to_pydatetime() > now_utc:
        df = df.iloc[:-1].reset_index(drop=True)

    print(f"[data] Got {len(df)} closed candles for {interval}.")
    return df


# ---------------------------------------------------------------------------
# 3. INDICATORS (vectorized with pandas, same conventions as the live bot:
#    SMA-seeded EMA, Wilder's RSI)
# ---------------------------------------------------------------------------
def add_indicators(df):
    df = df.copy()

    closes = df["close"]

    # SMA-seeded EMA, matching the live bot's behaviour exactly.
    def sma_seeded_ema(series, period):
        values = series.to_numpy()
        out = np.full(len(values), np.nan)
        if len(values) < period:
            return pd.Series(out, index=series.index)
        k = 2 / (period + 1)
        seed = values[:period].mean()
        out[period - 1] = seed
        prev = seed
        for i in range(period, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
        return pd.Series(out, index=series.index)

    df["ema9"] = sma_seeded_ema(closes, EMA_FAST_PERIOD)
    df["sma9"] = closes.rolling(SMA_PERIOD).mean()
    df["ema13"] = sma_seeded_ema(closes, EMA_TREND_PERIOD)

    # Wilder's RSI(28)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi28"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, "rsi28"] = 100.0

    # Crossover relationship + detect actual cross events
    df["relationship"] = np.where(df["ema9"] > df["sma9"], "above", "below")
    df["prev_relationship"] = df["relationship"].shift(1)
    df["bullish_cross"] = (df["prev_relationship"] == "below") & (df["relationship"] == "above")

    return df


# ---------------------------------------------------------------------------
# 4. BACKTEST ENGINE
# ---------------------------------------------------------------------------
def run_backtest(df, interval):
    """
    Walk forward candle by candle. On a bullish cross (closed candle),
    if not already in a position, open a BUY trade:
        entry = close of trigger candle
        sl    = low of trigger candle
        target = entry + RISK_REWARD * (entry - sl)
    Then scan forward from the NEXT candle to find whichever of
    sl/target is touched first (using candle high/low, not just close,
    for realism). If a candle's range touches both sl and target in the
    same candle, we conservatively assume SL was hit first (worst case).

    Position sizing: all-in. quantity = balance / entry.
    Fee: 0.06% round-trip charged on notional (entry_notional + exit_notional) * fee/2 each side,
         net effect modeled as: pnl_after_fee = pnl_gross - (entry_notional + exit_notional) * (ROUNDTRIP_FEE_PCT/2)... 
    Simplified equivalent (used below): fee_cost = entry_notional * ROUNDTRIP_FEE_PCT (whole round-trip
         charged once against entry notional, which is the standard simplification of round-trip cost).
    """
    trades = []
    balance = STARTING_BALANCE
    equity_curve = [{"time": df.iloc[0]["open_time"], "balance": balance}]

    in_position = False
    entry_idx = None
    entry_price = sl_price = target_price = None
    quantity = None

    n = len(df)
    min_valid_idx = max(EMA_TREND_PERIOD, RSI_PERIOD + 1, SMA_PERIOD) + 1

    for i in range(min_valid_idx, n):
        row = df.iloc[i]

        if not in_position:
            if bool(row["bullish_cross"]):
                entry_price = float(row["close"])
                sl_price = float(row["low"])
                risk = entry_price - sl_price

                if risk <= 0:
                    # Degenerate case (shouldn't really happen: entry should
                    # be >= candle low). Skip this signal.
                    continue

                target_price = entry_price + RISK_REWARD * risk
                quantity = balance / entry_price
                entry_idx = i
                in_position = True
        else:
            # Check if THIS candle's range hits SL or Target.
            # Conservative assumption: if both are touched in the same
            # candle, SL is assumed hit first.
            hit_sl = row["low"] <= sl_price
            hit_target = row["high"] >= target_price

            exit_price = None
            exit_reason = None

            if hit_sl:
                exit_price = sl_price
                exit_reason = "SL"
            elif hit_target:
                exit_price = target_price
                exit_reason = "TARGET"

            if exit_price is not None:
                entry_notional = quantity * entry_price
                exit_notional = quantity * exit_price
                gross_pnl = exit_notional - entry_notional
                fee_cost = entry_notional * ROUNDTRIP_FEE_PCT  # round-trip fee on notional
                net_pnl = gross_pnl - fee_cost

                balance += net_pnl

                r_multiple = (exit_price - entry_price) / (entry_price - sl_price)

                trades.append({
                    "entry_time": df.iloc[entry_idx]["open_time"],
                    "exit_time": row["open_time"],
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "target_price": target_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "quantity": quantity,
                    "gross_pnl": gross_pnl,
                    "fee_cost": fee_cost,
                    "net_pnl": net_pnl,
                    "r_multiple": r_multiple,
                    "balance_after": balance,
                    "bars_held": i - entry_idx,
                })

                equity_curve.append({"time": row["open_time"], "balance": balance})

                in_position = False
                entry_idx = None
                entry_price = sl_price = target_price = quantity = None

                if balance <= 0:
                    print(f"[{interval}] Account blown to <= 0 at {row['open_time']}, stopping backtest.")
                    break

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    return trades_df, equity_df, balance


# ---------------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------------
def compute_metrics(trades_df, equity_df, starting_balance, final_balance, interval, df_full):
    if trades_df.empty:
        return {
            "timeframe": interval,
            "total_trades": 0,
            "note": "No trades were generated in this period.",
        }

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    total_trades = len(trades_df)
    win_rate = len(wins) / total_trades * 100

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()  # positive number
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_r = trades_df["r_multiple"].mean()
    avg_win_r = wins["r_multiple"].mean() if not wins.empty else 0
    avg_loss_r = losses["r_multiple"].mean() if not losses.empty else 0

    total_return_pct = (final_balance / starting_balance - 1) * 100

    # Max drawdown on the equity curve
    eq = equity_df["balance"].to_numpy()
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd_pct = drawdowns.min() * 100 if len(drawdowns) else 0.0

    # Consecutive wins/losses
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    cur_streak_type = None
    for pnl in trades_df["net_pnl"]:
        is_win = pnl > 0
        if cur_streak_type == is_win:
            streak += 1
        else:
            streak = 1
            cur_streak_type = is_win
        if is_win:
            max_win_streak = max(max_win_streak, streak)
        else:
            max_loss_streak = max(max_loss_streak, streak)

    avg_bars_held = trades_df["bars_held"].mean()

    total_fees = trades_df["fee_cost"].sum()

    period_days = (df_full.iloc[-1]["open_time"] - df_full.iloc[0]["open_time"]).total_seconds() / 86400
    trades_per_day = total_trades / period_days if period_days > 0 else 0

    # CAGR-ish: annualized return based on actual period length
    years = period_days / 365.25
    if years > 0 and final_balance > 0:
        annualized_return_pct = ((final_balance / starting_balance) ** (1 / years) - 1) * 100
    else:
        annualized_return_pct = float("nan")

    return {
        "timeframe": interval,
        "period_start": df_full.iloc[0]["open_time"],
        "period_end": df_full.iloc[-1]["open_time"],
        "period_days": round(period_days, 1),
        "total_trades": total_trades,
        "trades_per_day": round(trades_per_day, 2),
        "win_rate_pct": round(win_rate, 2),
        "wins": len(wins),
        "losses": len(losses),
        "avg_r_multiple": round(avg_r, 3),
        "avg_win_r_multiple": round(avg_win_r, 3),
        "avg_loss_r_multiple": round(avg_loss_r, 3),
        "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else "inf",
        "starting_balance": starting_balance,
        "final_balance": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return_pct, 2) if not math.isnan(annualized_return_pct) else "N/A",
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_bars_held": round(avg_bars_held, 1),
        "total_fees_paid": round(total_fees, 2),
    }


def print_metrics_report(metrics):
    tf = metrics["timeframe"]
    print("\n" + "=" * 60)
    print(f" RESULTS — {SYMBOL} [{tf}]  9EMA/9SMA Crossover BUY Strategy")
    print("=" * 60)

    if metrics.get("total_trades", 0) == 0:
        print(metrics.get("note", "No trades."))
        return

    print(f" Period             : {metrics['period_start'].date()} -> {metrics['period_end'].date()} "
          f"({metrics['period_days']} days)")
    print(f" Total trades       : {metrics['total_trades']}  ({metrics['trades_per_day']}/day)")
    print(f" Win rate           : {metrics['win_rate_pct']}%  ({metrics['wins']}W / {metrics['losses']}L)")
    print(f" Avg R multiple     : {metrics['avg_r_multiple']}  (avg win {metrics['avg_win_r_multiple']}R, "
          f"avg loss {metrics['avg_loss_r_multiple']}R)")
    print(f" Profit factor      : {metrics['profit_factor']}")
    print(f" Starting balance   : ${metrics['starting_balance']:.2f}")
    print(f" Final balance      : ${metrics['final_balance']:.2f}")
    print(f" Total return       : {metrics['total_return_pct']}%")
    print(f" Annualized return  : {metrics['annualized_return_pct']}%")
    print(f" Max drawdown       : {metrics['max_drawdown_pct']}%")
    print(f" Max win streak     : {metrics['max_win_streak']}")
    print(f" Max loss streak    : {metrics['max_loss_streak']}")
    print(f" Avg bars held      : {metrics['avg_bars_held']}")
    print(f" Total fees paid    : ${metrics['total_fees_paid']:.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    all_metrics = []

    for interval in TIMEFRAMES:
        try:
            raw_df = fetch_full_history(SYMBOL, interval, LOOKBACK_DAYS)
        except Exception as e:
            print(f"[{interval}] ERROR fetching data: {e}")
            continue

        df = add_indicators(raw_df)
        trades_df, equity_df, final_balance = run_backtest(df, interval)
        metrics = compute_metrics(trades_df, equity_df, STARTING_BALANCE, final_balance, interval, df)
        all_metrics.append(metrics)
        print_metrics_report(metrics)

        # Export per-timeframe CSVs
        if not trades_df.empty:
            trades_path = f"{OUTPUT_DIR}/{SYMBOL}_{interval}_trades.csv"
            equity_path = f"{OUTPUT_DIR}/{SYMBOL}_{interval}_equity_curve.csv"
            trades_df.to_csv(trades_path, index=False)
            equity_df.to_csv(equity_path, index=False)
            print(f"[{interval}] Trades log  -> {trades_path}")
            print(f"[{interval}] Equity curve -> {equity_path}")

    # Combined summary table across timeframes
    print("\n" + "#" * 60)
    print(" SUMMARY ACROSS TIMEFRAMES")
    print("#" * 60)
    summary_df = pd.DataFrame(all_metrics)
    if not summary_df.empty:
        cols = ["timeframe", "total_trades", "win_rate_pct", "profit_factor",
                "total_return_pct", "annualized_return_pct", "max_drawdown_pct"]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))
        summary_df.to_csv(f"{OUTPUT_DIR}/{SYMBOL}_backtest_summary.csv", index=False)
        print(f"\nSummary saved -> {OUTPUT_DIR}/{SYMBOL}_backtest_summary.csv")


if __name__ == "__main__":
    main()
