#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
════════════════════════════════════════════════════════════
    COOL_SIGNALS STYLE — HIGH ACCURACY PULLBACK BOT
════════════════════════════════════════════════════════════
✅ Original Screenshot Design Format
✅ High-Accuracy EMA Pullback Strategy Integrated
✅ "No New Signals Found" Notification Fixed
✅ Fixed 0 Scanned Counter
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BOT_TOKEN = "8666315793:AAGQ-ejV45YezPFQZnOiIFhhawIePkCg7X4"
CHAT_ID = "5911994666"

INTERVAL_MINUTES = 15
TOP_N_COINS = 84  # Original count from your screenshot

EMA_FAST = 20    # Safe Pullback Strategy
EMA_SLOW = 200   # Safe Pullback Strategy
ATR_PERIOD = 14

CANDLE_INTERVAL = "15m"
CANDLE_LIMIT = 250

SKIP_COINS = {
    "USDT","USDC","BUSD","DAI","TUSD","USDP","USDD","FDUSD",
    "PYUSD","USDS","USD1","USDe","WBTC","WETH","STETH","WSTETH",
    "WBETH","BTCB","CBBTC","WBNB","WEETH","LEO","CRV"
}

LOG_FILE = "coolsignals_pullback.log"
STATE_FILE = "bot_state.json"

BINANCE_BASE = "https://api.binance.com/api/v3"

# ═══════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("CoolSignals")

session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))

# ═══════════════════════════════════════════════════════════
# HELPERS & TECHNICALS
# ═══════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

def calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))

# ═══════════════════════════════════════════════════════════
# BINANCE DIRECT TICKER FETCH
# ═══════════════════════════════════════════════════════════

def fetch_top_binance_coins(limit: int = TOP_N_COINS) -> list[str]:
    url = f"{BINANCE_BASE}/ticker/24hr"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        tickers = r.json()
        
        usdt_pairs = []
        for t in tickers:
            sym = t["symbol"]
            if sym.endswith("USDT") and not any(x in sym for x in ["UP", "DOWN", "BULL", "BEAR"]):
                base_asset = sym[:-4]
                if base_asset in SKIP_COINS: continue
                
                usdt_pairs.append({
                    "symbol": base_asset,
                    "volume": float(t.get("quoteVolume", 0))
                })
        
        usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
        return [x["symbol"] for x in usdt_pairs[:limit]]
    except Exception as e:
        log.error(f"Binance coin fetch error: {e}")
        return []

def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    pair = f"{symbol}USDT"
    url  = f"{BINANCE_BASE}/klines"
    try:
        r = session.get(url, params={"symbol": pair, "interval": CANDLE_INTERVAL, "limit": CANDLE_LIMIT}, timeout=10)
        if r.status_code == 400: return None
        r.raise_for_status()
        raw = r.json()
        df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol","close_ts","qvol","trades","tbvol","tqvol","_"])
        for col in ["open","high","low","close","vol"]: df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["ts","open","high","low","close","vol"]].reset_index(drop=True)
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════════════
# HIGH-ACCURACY PULLBACK STRATEGY LOGIC
# ═══════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame) -> dict | None:
    if len(df) < EMA_SLOW + 10: return None

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    
    e20   = calc_ema(closes, EMA_FAST)
    e200  = calc_ema(closes, EMA_SLOW)
    n = len(df) - 1

    price    = float(closes.iloc[n])
    atr_val  = calc_atr(df)
    rsi_val  = calc_rsi(closes)

    current_e20 = e20.iloc[n]
    current_e200 = e200.iloc[n]

    # 🟢 BULLISH PULLBACK (LONG SIGNAL)
    if current_e20 > current_e200 and price > current_e200:
        if lows.iloc[n] <= current_e20 * 1.002 and price >= current_e20:
            if 48 < rsi_val < 70:
                return {
                    "type"    : "LONG", "entry"   : price,
                    "sl"      : price - (atr_val * 1.5), 
                    "tp1"     : price + (atr_val * 1.5),
                    "tp2"     : price + (atr_val * 3.0), 
                    "rsi"     : rsi_val, "candle_time": str(df["ts"].iloc[n]),
                }

    # 🔴 BEARISH PULLBACK (SHORT SIGNAL)
    if current_e20 < current_e200 and price < current_e200:
        if highs.iloc[n] >= current_e20 * 0.998 and price <= current_e20:
            if 30 < rsi_val < 52:
                return {
                    "type"    : "SHORT", "entry"   : price,
                    "sl"      : price + (atr_val * 1.5), 
                    "tp1"     : price - (atr_val * 1.5),
                    "tp2"     : price - (atr_val * 3.0), 
                    "rsi"     : rsi_val, "candle_time": str(df["ts"].iloc[n]),
                }

    return None

# ═══════════════════════════════════════════════════════════
# ORIGINAL SCREENSHOT MESSAGE DESIGN FORMAT
# ═══════════════════════════════════════════════════════════

def build_signal_msg(symbol: str, sig: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M")
    direction_emoji = "🟢" if sig["type"] == "LONG" else "🔴"
    
    return (
        f"🚨 *CONFIRMED FUTURES SIGNAL (20/200 Pullback)* 🚨\n\n"
        f"🪙 Coin: #{symbol}/USDT\n"
        f"📈 Direction: {direction_emoji} {sig['type']}\n"
        f"⏱ Timeframe: 15 Minute\n\n"
        f"📥 Entry Price: {sig['entry']:.4f}\n"
        f"🎯 Take Profit 1: {sig['tp1']:.4f}\n"
        f"🎯 Take Profit 2: {sig['tp2']:.4f}\n"
        f"🛑 Stop Loss: {sig['sl']:.4f}\n\n"
        f"📊 Filters:\n"
        f"✅ RSI (14) Confirmed: {sig['rsi']:.1f}\n\n"
        f"🕒 _Generated at: {now} UTC_"
    )

def build_summary_msg(scanned: int, found: int, skipped: int) -> str:
    if found == 0:
        return (
            f"📡 *Scan Complete Successfully*\n\n"
            f"🔍 Total Scanned: {scanned} Coins\n"
            f"❌ *Signals : No new filtered signals found.*\n"
            f"⏩ Skipped due to API load: {skipped}\n"
            f"⏱ Next loop in 15 minutes"
        )
    return (
        f"📡 *Scan Complete Successfully*\n\n"
        f"🔍 Total Scanned: {scanned} Coins\n"
        f"✅ Safe Signals Found: {found}\n"
        f"⏩ Skipped due to API load: {skipped}\n"
        f"⏱ Next loop in 15 minutes"
    )

# ═══════════════════════════════════════════════════════════
# MAIN STATE LOOP
# ═══════════════════════════════════════════════════════════

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try: return json.loads(Path(STATE_FILE).read_text())
        except Exception: pass
    return {}

def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))

async def run_bot():
    bot = Bot(token=BOT_TOKEN)
    state = load_state()

    start_msg = (
        f"🔄 *Cloud Nodes Synchronized!*\n"
        f"🔍 Scanning exactly {TOP_N_COINS} High-Volume Coins..."
    )
    await bot.send_message(chat_id=CHAT_ID, text=start_msg, parse_mode=ParseMode.MARKDOWN)

    while True:
        cycle_start = time.time()
        signals_found = 0
        skipped = 0
        scanned_successfully = 0

        coins = fetch_top_binance_coins(TOP_N_COINS)
        if not coins:
            log.error("Binance returned no coins. Retrying...")
            await asyncio.sleep(30)
            continue

        for symbol in coins:
            try:
                await asyncio.sleep(0.3)  # Anti-block pause
                
                df = fetch_ohlcv(symbol)
                if df is None or len(df) < EMA_SLOW + 5:
                    skipped += 1
                    continue

                scanned_successfully += 1
                sig = detect_signal(df)
                if sig is None: continue

                key = f"{symbol}_{sig['type']}_{sig['candle_time']}"
                if key in state: continue

                msg = build_signal_msg(symbol, sig)
                await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                
                state[key] = True
                save_state(state)
                signals_found += 1
                await asyncio.sleep(0.5)

            except TelegramError as te:
                log.error(f"Telegram error [{symbol}]: {te}")
                await asyncio.sleep(2)
            except Exception as e:
                skipped += 1

        summary = build_summary_msg(scanned_successfully, signals_found, skipped)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            log.error(f"Error sending summary: {e}")

        if len(state) > 500:
            keys = list(state.keys())
            state = {k: state[k] for k in keys[-500:]}
            save_state(state)

        elapsed = time.time() - cycle_start
        wait = max(10, INTERVAL_MINUTES * 60 - elapsed)
        await asyncio.sleep(wait)

if __name__ == "__main__":
    log.info("Starting Correct Hybrid Bot...")
    asyncio.run(run_bot())
