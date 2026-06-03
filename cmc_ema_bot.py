#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
════════════════════════════════════════════════════════════
    CMC EMA PULLBACK SIGNAL BOT — FINAL COMPLETE VERSION
════════════════════════════════════════════════════════════
✅ Added Trend & Momentum Pullback Strategy (High Probability)
✅ Added "No New Signals Found" Notification
✅ Fixed Missing ParseMode Import
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

BOT_TOKEN = os.getenv("BOT_TOKEN", "8666315793:AAGQ-ejV45YezPFQZnOiIFhhawIePkCg7X4")
CHAT_ID = str(os.getenv("CHAT_ID", "5911994666"))
CMC_API_KEY = os.getenv("CMC_API_KEY", "725ae1359e2b4f95b90cd2b398886c25")

INTERVAL_MINUTES = 15
TOP_N_COINS = 100
MIN_VOLUME_USD = 5_000_000

EMA_FAST = 20
EMA_SLOW = 200
ATR_PERIOD = 14

CANDLE_INTERVAL = "15m"
CANDLE_LIMIT = 250

CUSTOM_COINS = []
SKIP_COINS = {
    "USDT","USDC","BUSD","DAI","TUSD","USDP","USDD","FDUSD",
    "PYUSD","USDS","USD1","USDe","WBTC","WETH","STETH","WSTETH",
    "WBETH","BTCB","CBBTC","WBNB","WEETH","LEO","CRV"
}

LOG_FILE = "cmc_ema_bot.log"
STATE_FILE = "bot_state.json"

CMC_BASE = "https://pro-api.coinmarketcap.com/v1"
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
log = logging.getLogger("EMABot")

# Session setup
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))

# ═══════════════════════════════════════════════════════════
# HELPERS & TECHNICALS
# ═══════════════════════════════════════════════════════════

def fp(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if   abs(v) < 0.000001: return f"{v:.10f}"
    elif abs(v) < 0.0001  : return f"{v:.8f}"
    elif abs(v) < 0.01    : return f"{v:.6f}"
    elif abs(v) < 1       : return f"{v:.5f}"
    elif abs(v) < 100     : return f"{v:.3f}"
    else                  : return f"{v:,.2f}"

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

def signal_strength(ema20_val, ema200_val, closes: pd.Series) -> float:
    gap = abs(ema20_val - ema200_val) / ema200_val * 100
    mom = abs((closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] * 100)
    return round(min(99.1, 74 + gap * 12 + mom * 1.8), 1)

# ═══════════════════════════════════════════════════════════
# CMC & BINANCE API FETCHERS
# ═══════════════════════════════════════════════════════════

def fetch_top_coins(limit: int = TOP_N_COINS) -> list[str]:
    url = f"{CMC_BASE}/cryptocurrency/listings/latest"
    params = {"start": 1, "limit": limit, "sort": "market_cap", "cryptocurrency_type": "coins", "convert": "USD"}
    try:
        r = session.get(url, headers={"X-CMC_PRO_API_KEY": CMC_API_KEY, "Accept": "application/json"}, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        symbols = []
        for coin in data.get("data", []):
            sym = coin["symbol"]
            vol = coin.get("quote", {}).get("USD", {}).get("volume_24h", 0) or 0
            if sym in SKIP_COINS or vol < MIN_VOLUME_USD: continue
            symbols.append(sym)
        return symbols
    except Exception as e:
        log.error(f"CMC fetch error: {e}")
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
        log.debug(f"[{symbol}] OHLCV error: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# NEW HIGH-FREQUENCY PULLBACK SIGNAL DETECTION
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
    strength = signal_strength(e20.iloc[n], e200.iloc[n], closes)

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
                    "tp3"     : price + (atr_val * 5.0),
                    "ema20"   : float(current_e20), "ema200"  : float(current_e200),
                    "atr"     : atr_val, "rsi"     : rsi_val, "strength": strength,
                    "candle_time": str(df["ts"].iloc[n]),
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
                    "tp3"     : price - (atr_val * 5.0),
                    "ema20"   : float(current_e20), "ema200"  : float(current_e200),
                    "atr"     : atr_val, "rsi"     : rsi_val, "strength": strength,
                    "candle_time": str(df["ts"].iloc[n]),
                }

    return None

# ═══════════════════════════════════════════════════════════
# TELEGRAM MESSAGE BUILDERS
# ═══════════════════════════════════════════════════════════

def build_signal_msg(symbol: str, sig: dict) -> str:
    now      = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    is_long  = sig["type"] == "LONG"
    emoji    = "🟢" if is_long else "🔴"
    direct   = "📈 *LONG  — BUY (Pullback)*" if is_long else "📉 *SHORT — SELL (Pullback)*"
    desc     = "Price bounced off support at *EMA 20*" if is_long else "Price rejected at resistance from *EMA 20*"

    filled = round(sig["strength"] / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    rsi_label = f"{sig['rsi']:.1f} ✅ Confirmed"

    return (
        f"{emoji} *CMC FILTERED SIGNAL — {symbol}/USDT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{direct}\n"
        f"ℹ️ {desc}\n"
        f"🕐 *Time     :* `{now}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry    :* `$ {fp(sig['entry'])}`\n"
        f"🛑 *Stop Loss:* `$ {fp(sig['sl'])}`\n"
        f"🎯 *TP 1     :* `$ {fp(sig['tp1'])}`\n"
        f"🎯 *TP 2     :* `$ {fp(sig['tp2'])}`\n"
        f"🎯 *TP 3     :* `$ {fp(sig['tp3'])}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *RSI (14) :* `{rsi_label}`\n"
        f"💪 *Strength :* `{bar}` {sig['strength']}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Educational purpose only. Follow risk management._"
    )

def build_summary_msg(scanned: int, found: int, skipped: int) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    if found == 0:
        return (
            f"📡 *Scan Complete — {now}*\n"
            f"──────────────────────────\n"
            f"🔍 Scanned : `{scanned}` coins\n"
            f"❌ *Signals : No new filtered signals found.*\n"
            f"⏭ Skipped : `{skipped}`\n"
            f"⏱ Next scan in `{INTERVAL_MINUTES}` minutes..."
        )
    return (
        f"📡 *Scan Complete — {now}*\n"
        f"──────────────────────────\n"
        f"🔍 Scanned : `{scanned}` coins\n"
        f"✅ Signals : `{found}` highly filtered pullbacks found\n"
        f"⏭ Skipped : `{skipped}`\n"
        f"⏱ Next scan in `{INTERVAL_MINUTES}` minutes..."
    )

# ═══════════════════════════════════════════════════════════
# STATE & MAIN LOOP
# ═══════════════════════════════════════════════════════════

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try: return json.loads(Path(STATE_FILE).read_text())
        except Exception: pass
    return {}

def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))

async def run_bot():
    bot   = Bot(token=BOT_TOKEN)
    state = load_state()

    start_msg = (
        f"🤖 *CMC Filtered Signal Bot — Started!*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *Source    :* CoinMarketCap Top {TOP_N_COINS}\n"
        f"⏱ *Interval  :* Every `{INTERVAL_MINUTES}` minutes\n"
        f"📐 *Strategy  :* EMA Pullback + RSI (Trend Continuous)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Scanning started..."
    )
    await bot.send_message(chat_id=CHAT_ID, text=start_msg, parse_mode=ParseMode.MARKDOWN)

    while True:
        cycle_start   = time.time()
        signals_found = 0
        skipped       = 0

        coins = CUSTOM_COINS if CUSTOM_COINS else fetch_top_coins(TOP_N_COINS)
        if not coins:
            log.error("No coins fetched. Retrying in 5 mins...")
            await asyncio.sleep(300)
            continue

        for symbol in coins:
            try:
                df = fetch_ohlcv(symbol)
                if df is None or len(df) < EMA_SLOW + 5:
                    skipped += 1
                    continue

                sig = detect_signal(df)
                if sig is None: continue

                key = f"{symbol}_{sig['type']}_{sig['candle_time']}"
                if key in state: continue

                msg = build_signal_msg(symbol, sig)
                await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                
                state[key] = True
                save_state(state)
                signals_found += 1
                await asyncio.sleep(0.6)

            except TelegramError as te:
                log.error(f"Telegram error [{symbol}]: {te}")
                await asyncio.sleep(2)
            except Exception as e:
                skipped += 1

        scanned = len(coins) - skipped
        summary = build_summary_msg(scanned, signals_found, skipped)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            log.error(f"Error sending summary: {e}")

        if len(state) > 500:
            keys = list(state.keys())
            state = {k: state[k] for k in keys[-500:]}
            save_state(state)

        elapsed = time.time() - cycle_start
        wait    = max(10, INTERVAL_MINUTES * 60 - elapsed)
        log.info(f"Cycle done. Signals found: {signals_found}. Waiting {wait:.0f}s.")
        await asyncio.sleep(wait)

if __name__ == "__main__":
    log.info("Starting Bot...")
    asyncio.run(run_bot())
