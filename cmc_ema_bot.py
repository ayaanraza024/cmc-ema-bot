# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import requests
    import pandas as pd
    import numpy  as np
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
except ImportError as e:
    print(f"\n❌ Missing library: {e}")
    print("   Run:  pip install python-telegram-bot requests pandas numpy\n")
    raise SystemExit(1)

# ═════════════════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURATION  —  Yahan apni values bharein
# ═════════════════════════════════════════════════════════════════════════════

# --- Telegram ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "")   # Set in Railway environment variables
CHAT_ID   = os.getenv("CHAT_ID",   "")   # Set in Railway environment variables

# --- CoinMarketCap ---
CMC_API_KEY = os.getenv("CMC_API_KEY", "")  # Set in Railway environment variables

# --- Bot Settings ---
INTERVAL_MINUTES  = 15      # Har kitne minute baad scan karna hai
TOP_N_COINS       = 100     # CMC se top kitne coins lene hain (max free: 200)
MIN_VOLUME_USD    = 5_000_000  # Min 24h volume filter ($5M)
EMA_FAST          = 20
EMA_SLOW          = 200
ATR_PERIOD        = 14
CANDLE_INTERVAL   = "15m"   # 15 minute candles
CANDLE_LIMIT      = 250     # EMA 200 ke liye 250+ candles chahiye

# Sirf ye coins scan karo (khali rakhne par TOP_N_COINS use hoga)
CUSTOM_COINS = []
# CUSTOM_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"]

# Ye coins skip karo (stablecoins etc.)
SKIP_COINS = {"USDT","USDC","BUSD","DAI","TUSD","USDP","USDD","FDUSD",
              "PYUSD","USDS","USD1","USDe","WBTC","WETH","STETH","WSTETH",
              "WBETH","BTCB","CBBTC","WBNB","WEETH","LEO","CRV"}

# Log file
LOG_FILE = "cmc_ema_bot.log"

# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════════
#  COINMARKETCAP  — Top coins list fetch
# ═════════════════════════════════════════════════════════════════════════════

CMC_BASE = "https://pro-api.coinmarketcap.com/v1"

def cmc_headers() -> dict:
    return {"X-CMC_PRO_API_KEY": CMC_API_KEY, "Accept": "application/json"}


def fetch_top_coins(limit: int = TOP_N_COINS) -> list[str]:
    """CMC se top-N coins ke symbols lata hai (by market cap)."""
    url = f"{CMC_BASE}/cryptocurrency/listings/latest"
    params = {
        "start": 1,
        "limit": limit,
        "sort": "market_cap",
        "cryptocurrency_type": "coins",
        "convert": "USD",
    }
    try:
        r = requests.get(url, headers=cmc_headers(), params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        symbols = []
        for coin in data.get("data", []):
            sym = coin["symbol"]
            vol = coin.get("quote", {}).get("USD", {}).get("volume_24h", 0) or 0
            if sym in SKIP_COINS:
                continue
            if vol < MIN_VOLUME_USD:
                continue
            symbols.append(sym)
        log.info(f"CMC se {len(symbols)} coins mila (top {limit} se filter ke baad)")
        return symbols
    except Exception as e:
        log.error(f"CMC fetch error: {e}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
#  BINANCE PUBLIC API  — OHLCV candles (free, no key needed)
#  CoinMarketCap historical OHLCV sirf paid plan mein hai,
#  isliye Binance public klines use karte hain — prices same hoti hain.
# ═════════════════════════════════════════════════════════════════════════════

BINANCE_BASE = "https://api.binance.com/api/v3"


def fetch_ohlcv(symbol: str, interval: str = CANDLE_INTERVAL,
                limit: int = CANDLE_LIMIT) -> pd.DataFrame | None:
    """Binance se OHLCV candles fetch karta hai."""
    pair = f"{symbol}USDT"
    url  = f"{BINANCE_BASE}/klines"
    try:
        r = requests.get(url,
                         params={"symbol": pair, "interval": interval, "limit": limit},
                         timeout=10)
        if r.status_code == 400:
            # Pair exist nahi karta Binance pe
            return None
        r.raise_for_status()
        raw = r.json()
        df  = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","vol",
            "close_ts","qvol","trades","tbvol","tqvol","_"
        ])
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["ts","open","high","low","close","vol"]].reset_index(drop=True)
    except Exception as e:
        log.debug(f"  [{symbol}] OHLCV error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame) -> dict | None:
    """
    EMA 20/200 crossover detect karta hai.
    Returns signal dict ya None.
    """
    if len(df) < EMA_SLOW + 10:
        return None

    closes = df["close"]
    e20    = calc_ema(closes, EMA_FAST)
    e200   = calc_ema(closes, EMA_SLOW)
    n, p   = len(df) - 1, len(df) - 2

    curr_above = e20.iloc[n] > e200.iloc[n]
    prev_above = e20.iloc[p] > e200.iloc[p]

    if curr_above == prev_above:
        return None  # No crossover

    price    = float(closes.iloc[n])
    atr_val  = calc_atr(df)
    rsi_val  = calc_rsi(closes)
    strength = signal_strength(e20.iloc[n], e200.iloc[n], closes)

    if curr_above:  # ─── LONG ────────────────────────────────────────────────
        return {
            "type"    : "LONG",
            "entry"   : price,
            "sl"      : price - atr_val * 1.5,
            "tp1"     : price + atr_val * 2.0,
            "tp2"     : price + atr_val * 4.0,
            "tp3"     : price + atr_val * 6.0,
            "ema20"   : float(e20.iloc[n]),
            "ema200"  : float(e200.iloc[n]),
            "atr"     : atr_val,
            "rsi"     : rsi_val,
            "strength": strength,
            "candle_time": str(df["ts"].iloc[n]),
        }
    else:  # ─── SHORT ──────────────────────────────────────────────────────
        return {
            "type"    : "SHORT",
            "entry"   : price,
            "sl"      : price + atr_val * 1.5,
            "tp1"     : price - atr_val * 2.0,
            "tp2"     : price - atr_val * 4.0,
            "tp3"     : price - atr_val * 6.0,
            "ema20"   : float(e20.iloc[n]),
            "ema200"  : float(e200.iloc[n]),
            "atr"     : atr_val,
            "rsi"     : rsi_val,
            "strength": strength,
            "candle_time": str(df["ts"].iloc[n]),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  PRICE FORMATTER
# ═════════════════════════════════════════════════════════════════════════════

def fp(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if   abs(v) < 0.000001: return f"{v:.10f}"
    elif abs(v) < 0.0001  : return f"{v:.8f}"
    elif abs(v) < 0.01    : return f"{v:.6f}"
    elif abs(v) < 1       : return f"{v:.5f}"
    elif abs(v) < 100     : return f"{v:.3f}"
    else                  : return f"{v:,.2f}"


# ═════════════════════════════════════════════════════════════════════════════
#  TELEGRAM MESSAGE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_signal_msg(symbol: str, sig: dict) -> str:
    now      = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    is_long  = sig["type"] == "LONG"
    emoji    = "🟢" if is_long else "🔴"
    direct   = "📈 *LONG  — BUY*" if is_long else "📉 *SHORT — SELL*"
    cross    = ("EMA20 ↑ crossed *ABOVE* EMA200 (Bullish)"
                if is_long else
                "EMA20 ↓ crossed *BELOW* EMA200 (Bearish)")

    # Strength bar
    pct    = sig["strength"]
    filled = round(pct / 10)
    bar    = "█" * filled + "░" * (10 - filled)

    # RSI label
    rsi = sig.get("rsi", 50)
    if   rsi > 70: rsi_label = f"{rsi:.1f} ⚠️ Overbought"
    elif rsi < 30: rsi_label = f"{rsi:.1f} ⚠️ Oversold"
    else         : rsi_label = f"{rsi:.1f} ✅ Neutral"

    rr_ratio = "1 : 2 / 4 / 6"  # Risk:Reward (SL=1.5ATR, TP=2/4/6 ATR)

    return (
        f"{emoji} *CMC SIGNAL — {symbol}/USDT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{direct}\n"
        f"🔀 {cross}\n"
        f"🕐 *Time     :* `{now}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry    :* `$ {fp(sig['entry'])}`\n"
        f"🛑 *Stop Loss:* `$ {fp(sig['sl'])}`\n"
        f"🎯 *TP 1     :* `$ {fp(sig['tp1'])}`\n"
        f"🎯 *TP 2     :* `$ {fp(sig['tp2'])}`\n"
        f"🎯 *TP 3     :* `$ {fp(sig['tp3'])}`\n"
        f"⚖️ *R:R Ratio:* `{rr_ratio}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📶 *EMA 20   :* `$ {fp(sig['ema20'])}`\n"
        f"📶 *EMA 200  :* `$ {fp(sig['ema200'])}`\n"
        f"📏 *ATR (14) :* `$ {fp(sig['atr'])}`\n"
        f"📊 *RSI (14) :* `{rsi_label}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💪 *Strength :* `{bar}` {pct}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Sirf educational. Apna analysis zaroor karein._\n"
        f"📡 _Source: CoinMarketCap Top {TOP_N_COINS}_"
    )


def build_summary_msg(scanned: int, found: int, skipped: int) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    return (
        f"📡 *Scan Complete — {now}*\n"
        f"──────────────────────────\n"
        f"🔍 Scanned : `{scanned}` coins\n"
        f"✅ Signals : `{found}` crossovers found\n"
        f"⏭ Skipped : `{skipped}` (no data / not listed)\n"
        f"⏱ Next scan in `{INTERVAL_MINUTES}` minutes..."
    )


# ═════════════════════════════════════════════════════════════════════════════
#  STATE  — Previously seen signals save karta hai (duplicate avoid)
# ═════════════════════════════════════════════════════════════════════════════

STATE_FILE = "bot_state.json"

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try:
            return json.loads(Path(STATE_FILE).read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


def signal_key(symbol: str, sig: dict) -> str:
    """Unique key: symbol + type + candle time — same signal dobara na bhejo."""
    return f"{symbol}_{sig['type']}_{sig['candle_time']}"


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN BOT LOOP
# ═════════════════════════════════════════════════════════════════════════════

async def run_bot():
    bot   = Bot(token=BOT_TOKEN)
    state = load_state()

    # ── Startup message ──────────────────────────────────────────────────────
    coins_to_scan = CUSTOM_COINS if CUSTOM_COINS else []
    start_msg = (
        f"🤖 *CMC EMA Signal Bot — Started!*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *Source    :* CoinMarketCap Top {TOP_N_COINS}\n"
        f"⏱ *Interval  :* Every `{INTERVAL_MINUTES}` minutes\n"
        f"📐 *Strategy  :* EMA {EMA_FAST} × EMA {EMA_SLOW} Crossover\n"
        f"🕯 *Timeframe :* {CANDLE_INTERVAL.upper()} Candles\n"
        f"💰 *Min Volume:* ${MIN_VOLUME_USD:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Fetching coin list from CMC..."
    )
    await bot.send_message(chat_id=CHAT_ID, text=start_msg, parse_mode=ParseMode.MARKDOWN)
    log.info("Bot started. Fetching coin list from CoinMarketCap...")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        cycle_start   = time.time()
        signals_found = 0
        skipped       = 0

        # ── Step 1: Get coins ────────────────────────────────────────────────
        if CUSTOM_COINS:
            coins = CUSTOM_COINS
        else:
            coins = fetch_top_coins(TOP_N_COINS)
            if not coins:
                log.error("CMC se coins nahi mila. 5 min baad retry...")
                await asyncio.sleep(300)
                continue

        log.info(f"Scanning {len(coins)} coins...")

        # ── Step 2: Scan each coin ───────────────────────────────────────────
        for symbol in coins:
            try:
                df = fetch_ohlcv(symbol)
                if df is None or len(df) < EMA_SLOW + 5:
                    skipped += 1
                    await asyncio.sleep(0.1)
                    continue

                sig = detect_signal(df)
                if sig is None:
                    continue

                key = signal_key(symbol, sig)
                if key in state:
                    log.debug(f"  [{symbol}] Duplicate signal skip")
                    continue

                # ── New signal! ──────────────────────────────────────────────
                msg = build_signal_msg(symbol, sig)
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=msg,
                    parse_mode=ParseMode.MARKDOWN,
                )
                state[key] = True
                save_state(state)
                signals_found += 1
                log.info(
                    f"  ✅ Signal: {symbol}/USDT → {sig['type']} "
                    f"@ ${fp(sig['entry'])}  strength={sig['strength']}%"
                )
                await asyncio.sleep(0.6)  # Telegram rate limit

            except TelegramError as te:
                log.error(f"  Telegram error [{symbol}]: {te}")
                await asyncio.sleep(2)
            except Exception as e:
                log.warning(f"  [{symbol}] Error: {e}")
                skipped += 1
                await asyncio.sleep(0.2)

        # ── Step 3: Summary ───────────────────────────────────────────────────
        scanned = len(coins) - skipped
        summary = build_summary_msg(scanned, signals_found, skipped)
        try:
            await bot.send_message(
                chat_id=CHAT_ID,
                text=summary,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

        # ── Step 4: Old state clean (rakhein sirf last 500 keys) ─────────────
        if len(state) > 500:
            keys = list(state.keys())
            state = {k: state[k] for k in keys[-500:]}
            save_state(state)

        # ── Step 5: Wait ──────────────────────────────────────────────────────
        elapsed = time.time() - cycle_start
        wait    = max(10, INTERVAL_MINUTES * 60 - elapsed)
        log.info(
            f"Cycle done: {scanned} scanned, {signals_found} signals, "
            f"{skipped} skipped. Next in {wait:.0f}s."
        )
        await asyncio.sleep(wait)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Config check ─────────────────────────────────────────────────────────
    errors = []
    if not BOT_TOKEN   : errors.append("BOT_TOKEN set nahi kiya!")
    if not CHAT_ID     : errors.append("CHAT_ID set nahi kiya!")
    if not CMC_API_KEY : errors.append("CMC_API_KEY set nahi kiya!")

    if errors:
        print("\n" + "─"*50)
        print("❌  CONFIGURATION ERROR:")
        for e in errors:
            print(f"    • {e}")
        print("\n  Script ke top mein jakar values fill karein.")
        print("  Ya config.env file banao (dekho README neeche).")
        print("─"*50 + "\n")
        raise SystemExit(1)

    log.info("=" * 55)
    log.info("  CMC EMA Signal Bot  |  Starting...")
    log.info(f"  Top {TOP_N_COINS} coins | EMA {EMA_FAST}/{EMA_SLOW} | {CANDLE_INTERVAL.upper()} | Every {INTERVAL_MINUTES}min")
    log.info("=" * 55)

    asyncio.run(run_bot())
