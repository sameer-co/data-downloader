#!/usr/bin/env python3
"""
Binance Historical Data Downloader with Telegram Notifications
- Downloads 6 years of OHLCV data for timeframes: 1m, 3m, 5m, 15m, 1h
- Respects Binance rate limits
- Sends each completed CSV file to Telegram
- Self-installs all dependencies
"""

import subprocess
import sys
import os

# ─────────────────────────────────────────────
# SELF-INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
REQUIRED = ["requests", "pandas", "python-telegram-bot"]

def install_dependencies():
    print("📦 Checking and installing dependencies...")
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
            print(f"  ✅ {pkg} already installed")
        except ImportError:
            print(f"  ⬇️  Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✅ {pkg} installed")

install_dependencies()

# ─────────────────────────────────────────────
# IMPORTS (after install)
# ─────────────────────────────────────────────
import time
import math
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import telegram

# ─────────────────────────────────────────────
# ⚙️  USER CONFIGURATION — edit these
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
TELEGRAM_CHAT_ID   = "1950462171"

SYMBOL      = "BTCUSDT"                       # Binance trading pair
TIMEFRAMES  = ["5m", "15m", "1h"] # order to download
OUTPUT_DIR  = Path("binance_data")            # folder for CSV files
YEARS_BACK  = 6                               # how many years of history

# Binance REST limits
# Spot: 1200 weight/min; each klines call = 2 weight
# Safe: ~500 calls/min → sleep ~0.12s between calls
# We use 0.2s to be conservative
REQUEST_DELAY_S     = 0.2   # seconds between API calls
MAX_BARS_PER_CALL   = 1000  # Binance max klines limit
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com"

TF_MS = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
}


# ─────────────────────────────────────────────
# BINANCE HELPERS
# ─────────────────────────────────────────────

def get_server_time() -> int:
    """Return Binance server time in ms."""
    r = requests.get(f"{BINANCE_BASE}/api/v3/time", timeout=10)
    r.raise_for_status()
    return r.json()["serverTime"]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """
    Fetch up to MAX_BARS_PER_CALL klines from Binance.
    Returns list of raw kline rows.
    Handles 429/418 rate-limit responses with back-off.
    """
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     MAX_BARS_PER_CALL,
    }
    backoff = 5
    while True:
        try:
            r = requests.get(
                f"{BINANCE_BASE}/api/v3/klines",
                params=params,
                timeout=20,
            )
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", backoff))
                log.warning(f"⚠️  Rate-limited (429). Sleeping {retry_after}s …")
                time.sleep(retry_after)
                backoff = min(backoff * 2, 120)
                continue
            if r.status_code == 418:
                log.warning(f"🚫 IP banned (418). Sleeping {backoff}s …")
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            log.error(f"Request error: {exc}. Retrying in {backoff}s …")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


def klines_to_df(raw: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume",
              "quote_asset_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    return df[["open_time", "open", "high", "low", "close", "volume",
               "quote_asset_volume", "num_trades", "close_time"]]


def download_timeframe(symbol: str, interval: str, years: int) -> Path:
    """
    Download `years` of klines for the given interval.
    Saves to OUTPUT_DIR/<symbol>_<interval>_6y.csv
    Returns path to the CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{symbol}_{interval}_{years}y.csv"

    now_ms   = get_server_time()
    tf_ms    = TF_MS[interval]
    start_ms = now_ms - years * 365 * 24 * 3600 * 1000

    total_bars = (now_ms - start_ms) // tf_ms
    calls_needed = math.ceil(total_bars / MAX_BARS_PER_CALL)

    log.info(
        f"📥 [{interval}] Downloading ~{total_bars:,} bars "
        f"({calls_needed} API calls) …"
    )

    all_frames = []
    current_ms = start_ms
    fetched    = 0

    while current_ms < now_ms:
        chunk_end = min(current_ms + tf_ms * MAX_BARS_PER_CALL - 1, now_ms)
        raw = fetch_klines(symbol, interval, current_ms, chunk_end)
        if not raw:
            break

        df = klines_to_df(raw)
        all_frames.append(df)
        fetched += len(df)

        last_close_ms = int(df["close_time"].iloc[-1].timestamp() * 1000)
        current_ms = last_close_ms + 1

        pct = min(100, (fetched / total_bars) * 100)
        log.info(f"  [{interval}] {fetched:,}/{total_bars:,} bars  ({pct:.1f}%)")

        time.sleep(REQUEST_DELAY_S)

    result = pd.concat(all_frames, ignore_index=True).drop_duplicates("open_time")
    result.to_csv(out_path, index=False)
    log.info(f"  ✅ [{interval}] Saved {len(result):,} rows → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# TELEGRAM HELPERS
# ─────────────────────────────────────────────

async def send_file_telegram(bot: telegram.Bot, chat_id: str, path: Path, caption: str):
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"📤 Sending {path.name} ({size_mb:.2f} MB) to Telegram …")
    with open(path, "rb") as fh:
        await bot.send_document(
            chat_id=chat_id,
            document=fh,
            filename=path.name,
            caption=caption,
            read_timeout=120,
            write_timeout=120,
        )
    log.info(f"  ✅ Sent {path.name}")


async def send_message_telegram(bot: telegram.Bot, chat_id: str, text: str):
    await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

    # Announce start
    await send_message_telegram(
        bot, TELEGRAM_CHAT_ID,
        f"🚀 <b>Binance Downloader Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"Timeframes: {', '.join(TIMEFRAMES)}\n"
        f"History: {YEARS_BACK} years\n"
        f"Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    overall_start = time.time()

    for tf in TIMEFRAMES:
        tf_start = time.time()
        log.info(f"\n{'='*50}")
        log.info(f"Starting timeframe: {tf}")
        log.info(f"{'='*50}")

        csv_path = download_timeframe(SYMBOL, tf, YEARS_BACK)

        elapsed = time.time() - tf_start
        rows    = sum(1 for _ in open(csv_path)) - 1  # subtract header
        size_mb = csv_path.stat().st_size / 1_048_576

        caption = (
            f"✅ <b>{SYMBOL} — {tf} ({YEARS_BACK}y)</b>\n"
            f"Rows: {rows:,}\n"
            f"Size: {size_mb:.2f} MB\n"
            f"Duration: {elapsed/60:.1f} min"
        )

        await send_file_telegram(bot, TELEGRAM_CHAT_ID, csv_path, caption)

    total_elapsed = time.time() - overall_start
    await send_message_telegram(
        bot, TELEGRAM_CHAT_ID,
        f"🎉 <b>All downloads complete!</b>\n"
        f"Total time: {total_elapsed/60:.1f} min\n"
        f"Files saved in: <code>{OUTPUT_DIR.resolve()}</code>"
    )
    log.info(f"\n🎉 All done in {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
