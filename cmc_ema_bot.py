#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
════════════════════════════════════════════════════════════
    EMA TRIPLE SCALPER BOT — TOP 100 COINS DEPLOYMENT VERSION
════════════════════════════════════════════════════════════
✅ STRATEGY: EMA 9 / 21 / 50 Golden Triple Scalper
✅ TIMEFRAME: 5-Minute (Quick & Responsive Signals)
✅ SCAN VOLUME: Top 100 High-Volume Crypto Pairs
✅ CRASH-PROOF: Dynamic MultiIndex column flattening handled
"""

import asyncio
import json
import logging
import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

# yfinance warnings and extra logs bypass
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BOT_TOKEN = os.getenv("BOT_TOKEN", "8666315793:AAGQ-ejV45YezPFQZnOiIFhhawIePkCg7X4")
CHAT_ID = str(os.getenv("CHAT_ID", "5911994666"))

# Strategy Parameters
INTERVAL_MINUTES = 5   # 5-minute chart
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
ATR_PERIOD = 14

# Top 100 High-Volume & Liquid Coins
HARDCODED_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC",
    "LTC", "BCH", "UNI", "ATOM", "XLM", "TRX", "ETC", "FIL", "LDO", "HBAR",
    "APT", "ARB", "OP", "NEAR", "GRT", "AAVE", "MKR", "EGLD", "THETA", "INJ",
    "RUNE", "SUI", "TIA", "SEI", "IMX", "STX", "FTM", "RENDER", "GALA", "ALGO",
    "VET", "ICP", "FLOW", "SAND", "MANA", "AXS", "CHZ", "CRV", "MINA", "WOO",
    "DYDX", "GMX", "FET", "JUP", "PYTH", "WIF", "PEPE", "SHIB", "FLOKI", "BONK",
    "TIA", "ORDI", "OP", "IMX", "KAS", "BEAM", "STX", "EGLD", "THETA", "STX",
    "LUNC", "USTC", "GALA", "LRC", "ACH", "ANKR", "ENS", "WAVE", "JTO", "BTT",
    "FTT", "GNS", "AGIX", "OCEAN", "MASK", "PEOPLE", "TRB", "RNDR", "BLUR", "GMT",
    "ZIL", "ONE", "ENJ", "BAT", "RVN", "QTUM", "DASH", "XMR", "ZEC", "WAVES"
]

# Clean list duplicates if any
HARDCODED_COINS = sorted(list(set(HARDCODED_COINS)))

LOG_FILE = "triple_ema_bot.log"
STATE_FILE = "triple_bot_state.json"

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
log = logging.getLogger("TripleScalper")

# ═══════════════════════════════════════════════════════════
# DATA FETCHING ENGINE (5m Optimized)
# ═══════════════════════════════════════════════════════════

def fetch_ohlcv_yfinance(symbol: str) -> pd.DataFrame | None:
    ticker = f"{symbol}-USD"
    try:
        # 5-minute interval ke liye 2-3 din ka data downloads safely for 50+ structures
        df = yf.download(tickers=ticker, period="3d", interval="5m", progress=False, auto_adjust=True)
        
        if df.empty or len(df) < EMA_SLOW + 10:
            return None
        
        df = df.reset_index()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[0] else col[1] for col in df.columns]
        
        df = df.rename(columns={
            "Datetime": "ts", "Open": "open", "High": "high", 
            "Low": "low", "Close": "close", "Volume": "vol"
        })
        
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
            
        return df[["ts", "open", "high", "low", "close", "vol"]]
    except Exception as e:
        log.debug(f" [{symbol}] Fetch skip structure: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# MATHS & CORE INDICATORS
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
# TRIPLE EMA STRATEGY DETECTION
# ═══════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame) -> dict | None:
    closes = df["close"]
    
    e9   = calc_ema(closes, EMA_FAST)
    e21  = calc_ema(closes, EMA_MEDIUM)
    e50  = calc_ema(closes, EMA_SLOW)
    
    n = len(df) - 1  # Current candle index
    p = len(df) - 2  # Previous candle index

    # Core Condition 1: Fresh Cross of 9 EMA over 21 EMA
    cross_up   = (e9.iloc[p] <= e21.iloc[p]) and (e9.iloc[n] > e21.iloc[n])
    cross_down = (e9.iloc[p] >= e21.iloc[p]) and (e9.iloc[n] < e21.iloc[n])

    if not (cross_up or cross_down):
        return None

    rsi_val = calc_rsi(closes)
    price = float(closes.iloc[n])
    
    # Core Condition 2 & 3: Filter trend using 50 EMA + RSI Check
    if cross_up:
        # Check trend deewar (Both lines must be ABOVE 50 EMA)
        if e9.iloc[n] < e50.iloc[n] or e21.iloc[n] < e50.iloc[n]:
            return None
        # RSI Confirmation (Strong Momentum, not oversold)
        if rsi_val < 45 or rsi_val > 68:
            return None
        signal_type = "LONG"
        
    elif cross_down:
        # Check trend deewar (Both lines must be BELOW 50 EMA)
        if e9.iloc[n] > e50.iloc[n] or e21.iloc[n] > e50.iloc[n]:
            return None
        # RSI Confirmation
        if rsi_val > 55 or rsi_val < 32:
            return None
        signal_type = "SHORT"

    atr_val = calc_atr(df)
    
    # Scalping Risk Management Matrix
    if signal_type == "LONG":
        sl  = price - atr_val * 1.5
        tp1 = price + atr_val * 1.5
        tp2 = price + atr_val * 3.0
        tp3 = price + atr_val * 4.5
    else:
        sl  = price + atr_val * 1.5
        tp1 = price - atr_val * 1.5
        tp2 = price - atr_val * 3.0
        tp3 = price - atr_val * 4.5

    return {
        "type"   : signal_type,
        "entry"  : price,
        "sl"     : sl,
        "tp1"    : tp1,
        "tp2"    : tp2,
        "tp3"    : tp3,
        "rsi"    : rsi_val,
        "candle_time": str(df["ts"].iloc[n]),
    }

# ═══════════════════════════════════════════════════════════
# MESSAGING & FORMATTING
# ═══════════════════════════════════════════════════════════

def fp(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if   abs(v) < 0.000001: return f"{v:.10f}"
    elif abs(v) < 0.0001  : return f"{v:.8f}"
    elif abs(v) < 0.01    : return f"{v:.6f}"
    elif abs(v) < 1       : return f"{v:.5f}"
    elif abs(v) < 100     : return f"{v:.3f}"
    else                  : return f"{v:,.2f}"

def build_signal_msg(symbol: str, sig: dict) -> str:
    now     = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    is_long = sig["type"] == "LONG"
    emoji   = "🔥 🟢" if is_long else "🔥 🔴"
    direct  = "🚀 *STRONG SCALPING BUY (LONG)*" if is_long else "💥 *STRONG SCALPING SELL (SHORT)*"
    reason  = "⚡ _EMA 9/21 Crossed ABOVE 50 EMA_" if is_long else "⚡ _EMA 9/21 Crossed BELOW 50 EMA_"

    return (
        f"{emoji} *GOLDEN TRIPLE EMA SIGNAL — {symbol}/USDT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{direct}\n"
        f"{reason}\n"
        f"🕐 *Timeframe:* `5-Minute Chart`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry Price:* `$ {fp(sig['entry'])}`\n"
        f"🛑 *Stop Loss  :* `$ {fp(sig['sl'])}`\n"
        f"🎯 *Target 1   :* `$ {fp(sig['tp1'])}`\n"
        f"🎯 *Target 2   :* `$ {fp(sig['tp2'])}`\n"
        f"🎯 *Target 3   :* `$ {fp(sig['tp3'])}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *RSI (14)   :* `{sig['rsi']:.1f} (Trend Verified)`\n"
        f"📡 *Scan Time  :* `{now}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Quick scalping setup. Fasten your SL._"
    )

def build_summary_msg(scanned: int, found: int, skipped: int) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    return (
        f"📡 *Active Scalper Scan Complete — {now}*\n"
        f"──────────────────────────\n"
        f"🔍 Scanned Coins: `{scanned}`/100 Matrix\n"
        f"⚡ Fresh Signals Found: *{found}*\n"
        f"⏭ Skipped Coins: `{skipped}`\n"
        f"⏱ Next scan pulse in `{INTERVAL_MINUTES}` minutes..."
    )

# ═══════════════════════════════════════════════════════════
# RUNNER SYSTEM
# ═══════════════════════════════════════════════════════════

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try: return json.loads(Path(STATE_FILE).read_text())
        except Exception: pass
    return {}

def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))

def signal_key(symbol: str, sig: dict) -> str:
    return f"{symbol}_{sig['type']}_{sig['candle_time']}"

async def run_bot():
    bot   = Bot(token=BOT_TOKEN)
    state = load_state()

    start_msg = (
        f"🛡️ *Golden Triple EMA Scalper Activated!*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Timeframe:* 5 Minutes Pulse\n"
        f"💎 *Targets  :* Top {len(HARDCODED_COINS)} High-Volume Coins\n"
        f"🔥 *Engine   :* 9 / 21 / 50 Momentum Core\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        await bot.send_message(chat_id=CHAT_ID, text=start_msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log.error(f"Telegram start send failed: {e}")

    while True:
        cycle_start   = time.time()
        signals_found = 0
        skipped       = 0

        log.info(f"Scanning {len(HARDCODED_COINS)} coins on 5-minute matrix...")

        for symbol in HARDCODED_COINS:
            try:
                df = fetch_ohlcv_yfinance(symbol)
                if df is None:
                    skipped += 1
                    continue

                sig = detect_signal(df)
                if sig is None:
                    continue

                key = signal_key(symbol, sig)
                if key in state:
                    continue

                msg = build_signal_msg(symbol, sig)
                await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                
                state[key] = True
                save_state(state)
                signals_found += 1
                await asyncio.sleep(0.4)

            except TelegramError as te:
                log.error(f"Telegram error on [{symbol}]: {te}")
                await asyncio.sleep(1)
            except Exception as e:
                skipped += 1

        scanned = len(HARDCODED_COINS) - skipped
        summary = build_summary_msg(scanned, signals_found, skipped)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

        # Cleanup memory state if getting too huge
        if len(state) > 800:
            keys = list(state.keys())
            state = {k: state[k] for k in keys[-800:]}
            save_state(state)

        elapsed = time.time() - cycle_start
        wait    = max(5, INTERVAL_MINUTES * 60 - elapsed)
        log.info(f"Scan finished. Found={signals_found} | Sleep for {wait:.0f}s.")
        await asyncio.sleep(wait)

if __name__ == "__main__":
    log.info("Starting Deployment Version...")
    asyncio.run(run_bot())
