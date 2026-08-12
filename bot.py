import os, asyncio, logging, json, math, importlib
from datetime import datetime, timezone, timedelta

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EXCHANGE_ID = os.getenv("EXCHANGE", "binance").strip().lower()

SYMBOLS = [x.strip() for x in os.getenv(
    "SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT"
).split(",") if x.strip()]
TIMEFRAMES = [x.strip() for x in os.getenv(
    "TIMEFRAMES", "5m,15m,1h,4h"
).split(",") if x.strip()]
PRIMARY_TF = os.getenv("PRIMARY_TF", "15m").strip()
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "300"))
POLL_SECONDS = max(30, int(os.getenv("POLL_SECONDS", "60")))
MIN_SCORE = int(os.getenv("MIN_SCORE", "65"))
RR = float(os.getenv("RISK_REWARD", "2.0"))
ATR_MULT = float(os.getenv("ATR_SL_MULT", "1.2"))
TRAIL_ATR = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
STATE_FILE = "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def now_utc():
    return datetime.now(timezone.utc)

def fmt(n):
    n = float(n)
    return f"{n:,.4f}" if abs(n) < 100 else f"{n:,.2f}"

def indicators(df):
    df = df.copy()
    c = df["close"]

    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()

    return df

def analyze(df):
    # Ignore the currently forming candle.
    if len(df) < 230:
        return {"action": "NO TRADE", "score": 0, "reason": "Not enough candles"}

    df = indicators(df).dropna().reset_index(drop=True)
    if len(df) < 10:
        return {"action": "NO TRADE", "score": 0, "reason": "Indicators unavailable"}

    x = df.iloc[-2]
    p = float(x["close"])
    atr = float(x["atr"])

    if not math.isfinite(p) or not math.isfinite(atr) or atr <= 0:
        return {"action": "NO TRADE", "score": 0, "reason": "Invalid market data"}

    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []

    # Trend
    if x.ema20 > x.ema50 > x.ema200:
        long_score += 25
        long_reasons.append("EMA trend")
    elif x.ema20 < x.ema50 < x.ema200:
        short_score += 25
        short_reasons.append("EMA trend")

    if p > x.ema200:
        long_score += 8
    elif p < x.ema200:
        short_score += 8

    # RSI momentum, avoiding extreme chasing
    if 52 <= x.rsi <= 68:
        long_score += 15
        long_reasons.append("RSI momentum")
    elif 32 <= x.rsi <= 48:
        short_score += 15
        short_reasons.append("RSI momentum")

    # MACD
    if x.macd > x.macd_signal:
        long_score += 15
        long_reasons.append("MACD bullish")
    elif x.macd < x.macd_signal:
        short_score += 15
        short_reasons.append("MACD bearish")

    # Volume confirmation
    if math.isfinite(float(x.vol_ma)) and x.volume > 1.10 * x.vol_ma:
        if x.close > x.open:
            long_score += 10
            long_reasons.append("volume")
        elif x.close < x.open:
            short_score += 10
            short_reasons.append("volume")

    # Recent breakout/breakdown
    hi = float(df["high"].iloc[-22:-2].max())
    lo = float(df["low"].iloc[-22:-2].min())
    if p > hi:
        long_score += 15
        long_reasons.append("breakout")
    elif p < lo:
        short_score += 15
        short_reasons.append("breakdown")

    if long_score >= short_score:
        side, score, reasons = "LONG", long_score, long_reasons
    else:
        side, score, reasons = "SHORT", short_score, short_reasons

    if score < MIN_SCORE:
        return {
            "action": "NO TRADE", "score": score, "price": p,
            "rsi": float(x.rsi), "atr": atr,
            "reason": "Score below threshold"
        }

    # Risk levels are informational signal levels only.
    dist = max(ATR_MULT * atr, p * 0.003)
    if side == "LONG":
        sl = p - dist
        tp = p + dist * RR
    else:
        sl = p + dist
        tp = p - dist * RR

    return {
        "action": side,
        "score": int(score),
        "price": p,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "rsi": float(x.rsi),
        "reasons": reasons
    }

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        s.setdefault("positions", {})
        s.setdefault("signals", [])
        s.setdefault("cooldowns", {})
        s.setdefault("last_scan", None)
        s.setdefault("scan_count", 0)
        return s
    except Exception:
        return {
            "positions": {},
            "signals": [],
            "cooldowns": {},
            "last_scan": None,
            "scan_count": 0
        }

def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

async def candles(exchange, symbol, tf, limit=CANDLE_LIMIT):
    raw = await exchange.fetch_ohlcv(symbol, tf, limit=limit)
    if not raw:
        raise RuntimeError("Exchange returned no candles")
    return pd.DataFrame(
        raw,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

def cooldown_active(state, key):
    stamp = state["cooldowns"].get(key)
    if not stamp:
        return False
    try:
        t = datetime.fromisoformat(stamp)
        return now_utc() - t < timedelta(minutes=COOLDOWN_MIN)
    except Exception:
        return False

def set_cooldown(state, key):
    state["cooldowns"][key] = now_utc().isoformat()

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Crypto Signal Bot v7 active.\n\n"
        "/status — bot + tracked positions\n"
        "/signals — recent signals\n"
        "/scan — run an immediate market scan\n"
        "/backtest — run a backtest report\n"
        "/ping — connection test\n\n"
        "Automatic scanner is ON.\n"
        "Signal-only mode: it does NOT place exchange orders.\n"
        "Signals are probability-based, not guaranteed profits."
    )

async def ping_cmd(update, context):
    await update.message.reply_text("🟢 Bot is online and responding.")

async def status_cmd(update, context):
    state = load_state()
    positions = state["positions"]
    last = state.get("last_scan") or "not scanned yet"

    text = (
        "📊 STATUS\n\n"
        f"Scanner: 🟢 ON\n"
        f"Exchange: {EXCHANGE_ID}\n"
        f"Pairs: {', '.join(SYMBOLS)}\n"
        f"Timeframes: {', '.join(TIMEFRAMES)}\n"
        f"Primary TF: {PRIMARY_TF}\n"
        f"Last scan: {last}\n"
        f"Scan cycles: {state.get('scan_count', 0)}\n\n"
    )

    if not positions:
        text += "No open tracked positions."
    else:
        for key, p in positions.items():
            text += (
                f"{p['symbol']} {p['side']}\n"
                f"Entry: {fmt(p['entry'])} | SL: {fmt(p['sl'])} | TP: {fmt(p['tp'])}\n\n"
            )

    await update.message.reply_text(text)

async def signals_cmd(update, context):
    state = load_state()
    recent = state["signals"][-10:]
    if not recent:
        await update.message.reply_text("No signals yet. Automatic scanner is running.")
        return

    text = "📜 RECENT SIGNALS\n\n"
    for x in recent:
        text += (
            f"{x['time']} | {x['symbol']} | {x['side']}\n"
            f"Entry {fmt(x['entry'])} | SL {fmt(x['sl'])} | TP {fmt(x['tp'])} | "
            f"Score {x['score']}%\n\n"
        )
    await update.message.reply_text(text)

def update_trailing(pos, price):
    atr = float(pos["atr"])
    if pos["side"] == "LONG":
        pos["best"] = max(pos.get("best", pos["entry"]), price)
        pos["sl"] = max(pos["sl"], pos["best"] - TRAIL_ATR * atr)
    else:
        pos["best"] = min(pos.get("best", pos["entry"]), price)
        pos["sl"] = min(pos["sl"], pos["best"] + TRAIL_ATR * atr)
    return pos

async def monitor(exchange, tg, state):
    for key, pos in list(state["positions"].items()):
        try:
            ticker = await exchange.fetch_ticker(pos["symbol"])
            price = float(ticker["last"])
            pos = update_trailing(pos, price)

            reason = None
            if pos["side"] == "LONG":
                if price <= pos["sl"]:
                    reason = "Trailing/Stop Loss"
                elif price >= pos["tp"]:
                    reason = "Take Profit"
            else:
                if price >= pos["sl"]:
                    reason = "Trailing/Stop Loss"
                elif price <= pos["tp"]:
                    reason = "Take Profit"

            if reason:
                move_pct = (
                    (price - pos["entry"]) / pos["entry"] * 100
                    if pos["side"] == "LONG"
                    else (pos["entry"] - price) / pos["entry"] * 100
                )
                await tg.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        "🔔 POSITION CLOSE SIGNAL\n\n"
                        f"{pos['symbol']} {pos['side']}\n"
                        f"Close: {fmt(price)}\n"
                        f"Reason: {reason}\n"
                        f"Approx move: {move_pct:+.2f}%\n\n"
                        "Signal-only mode: no order was placed."
                    )
                )
                state["signals"].append({
                    "time": now_utc().strftime("%Y-%m-%d %H:%M UTC"),
                    "symbol": pos["symbol"],
                    "side": "CLOSE",
                    "entry": price,
                    "sl": price,
                    "tp": price,
                    "score": 0
                })
                del state["positions"][key]
                set_cooldown(state, key)

        except Exception:
            logging.exception("monitor failed for %s", pos["symbol"])

async def perform_scan(exchange, tg, state, notify=True):
    found = 0
    errors = []

    for symbol in SYMBOLS:
        try:
            results = {}
            for tf in TIMEFRAMES:
                results[tf] = analyze(await candles(exchange, symbol, tf))

            primary = results.get(PRIMARY_TF) or results[TIMEFRAMES[0]]
            if primary["action"] == "NO TRADE":
                continue

            # Require at least two timeframes to agree.
            confirmations = [
                tf for tf, result in results.items()
                if result["action"] == primary["action"]
            ]
            if len(confirmations) < 2:
                continue

            key = f"{symbol}:{PRIMARY_TF}"
            if key in state["positions"] or cooldown_active(state, key):
                continue

            # A stronger signal gets a small bonus from multi-TF agreement.
            confidence = min(
                99,
                int(primary["score"] + max(0, len(confirmations) - 2) * 5)
            )

            position = {
                "symbol": symbol,
                "side": primary["action"],
                "entry": primary["price"],
                "sl": primary["sl"],
                "tp": primary["tp"],
                "atr": primary["atr"],
                "best": primary["price"],
                "opened": now_utc().isoformat(),
                "confirmations": confirmations,
                "score": confidence
            }
            state["positions"][key] = position

            signal = {
                "time": now_utc().strftime("%Y-%m-%d %H:%M UTC"),
                "symbol": symbol,
                "side": primary["action"],
                "entry": primary["price"],
                "sl": primary["sl"],
                "tp": primary["tp"],
                "score": confidence
            }
            state["signals"].append(signal)
            state["signals"] = state["signals"][-100:]
            found += 1

            if notify:
                await tg.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        "🚨 AUTOMATIC CRYPTO SIGNAL\n\n"
                        f"Pair: {symbol}\n"
                        f"Position: {primary['action']}\n"
                        f"Entry: {fmt(primary['price'])}\n"
                        f"Stop Loss: {fmt(primary['sl'])}\n"
                        f"Take Profit: {fmt(primary['tp'])}\n"
                        f"Score: {confidence}%\n"
                        f"Confirmed TFs: {', '.join(confirmations)}\n"
                        f"RSI: {primary['rsi']:.1f}\n"
                        f"Reasons: {', '.join(primary.get('reasons', []))}\n\n"
                        "⚠️ Signal only — no exchange order was placed."
                    )
                )

        except Exception as exc:
            logging.exception("scan failed for %s", symbol)
            errors.append(f"{symbol}: {type(exc).__name__}")

    state["last_scan"] = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    state["scan_count"] = int(state.get("scan_count", 0)) + 1
    save_state(state)
    return found, errors

async def scan_cmd(update, context):
    if update.effective_chat.id != int(CHAT_ID):
        await update.message.reply_text("Not authorized.")
        return

    await update.message.reply_text("🔎 Scanning BTC/ETH/SOL across multiple timeframes...")
    state = load_state()
    found, errors = await perform_scan(context.application.bot_data["exchange"],
                                       context.application.bot, state, notify=False)
    msg = f"✅ Scan complete.\nNew signals: {found}"
    if errors:
        msg += "\nErrors: " + ", ".join(errors)
    if found == 0:
        msg += "\nNo qualifying setup right now."
    await update.message.reply_text(msg)

def backtest_df(df):
    # Backtest the same core signal logic without look-ahead.
    if len(df) < 260:
        return {"trades": 0, "win_rate": 0, "net_pct": 0, "avg_pct": 0}

    trades = []
    pos = None

    for i in range(230, len(df) - 1):
        window = df.iloc[:i+1].copy()

        if pos is None:
            sig = analyze(window)
            if sig["action"] == "NO TRADE":
                continue
            pos = {
                "side": sig["action"],
                "entry": sig["price"],
                "sl": sig["sl"],
                "tp": sig["tp"],
                "atr": sig["atr"],
                "best": sig["price"]
            }
            continue

        row = df.iloc[i+1]
        high = float(row.high)
        low = float(row.low)

        if pos["side"] == "LONG":
            pos["best"] = max(pos["best"], high)
            pos["sl"] = max(
                pos["sl"], pos["best"] - TRAIL_ATR * pos["atr"]
            )
            if low <= pos["sl"]:
                exitp = pos["sl"]
            elif high >= pos["tp"]:
                exitp = pos["tp"]
            else:
                exitp = None
            pnl = (
                (exitp - pos["entry"]) / pos["entry"] * 100
                if exitp else None
            )
        else:
            pos["best"] = min(pos["best"], low)
            pos["sl"] = min(
                pos["sl"], pos["best"] + TRAIL_ATR * pos["atr"]
            )
            if high >= pos["sl"]:
                exitp = pos["sl"]
            elif low <= pos["tp"]:
                exitp = pos["tp"]
            else:
                exitp = None
            pnl = (
                (pos["entry"] - exitp) / pos["entry"] * 100
                if exitp else None
            )

        if pnl is not None:
            trades.append(pnl)
            pos = None

    if not trades:
        return {"trades": 0, "win_rate": 0, "net_pct": 0, "avg_pct": 0}

    wins = sum(x > 0 for x in trades)
    return {
        "trades": len(trades),
        "win_rate": wins / len(trades) * 100,
        "net_pct": sum(trades),
        "avg_pct": sum(trades) / len(trades)
    }

async def run_backtests(exchange, tg):
    lines = ["🧪 BACKTEST REPORT (informational)"]
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df = await candles(exchange, symbol, tf, 1000)
                r = backtest_df(df)
                lines.append(
                    f"{symbol} {tf}: {r['trades']} trades | "
                    f"Win {r['win_rate']:.1f}% | Net {r['net_pct']:.2f}%"
                )
            except Exception as exc:
                logging.exception("backtest failed for %s %s", symbol, tf)
                lines.append(
                    f"{symbol} {tf}: error — {type(exc).__name__}: "
                    f"{str(exc)[:80]}"
                )
    await tg.send_message(chat_id=CHAT_ID, text="\n".join(lines))

async def backtest_cmd(update, context):
    if update.effective_chat.id != int(CHAT_ID):
        await update.message.reply_text("Not authorized.")
        return
    await update.message.reply_text("🧪 Backtest started. I'll send the report when finished.")
    await run_backtests(context.application.bot_data["exchange"],
                        context.application.bot)

async def main():
    if not TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Railway Variables"
        )

    allowed = {
        "binance", "kraken", "coinbase", "okx",
        "bybit", "kucoin", "gateio", "bitget"
    }
    if EXCHANGE_ID not in allowed:
        raise RuntimeError(f"Unsupported EXCHANGE={EXCHANGE_ID}")

    try:
        exchange_module = importlib.import_module(
            f"ccxt.async_support.{EXCHANGE_ID}"
        )
        exchange_class = getattr(exchange_module, EXCHANGE_ID)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"CCXT exchange '{EXCHANGE_ID}' could not be loaded: {exc}"
        ) from exc

    exchange = exchange_class({"enableRateLimit": True})

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))

    app.bot_data["exchange"] = exchange

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    tg = app.bot
    state = load_state()

    try:
        logging.info(
            "Automatic scanner started: symbols=%s timeframes=%s poll=%ss",
            SYMBOLS, TIMEFRAMES, POLL_SECONDS
        )

        # Immediate first scan.
        await perform_scan(exchange, tg, state, notify=True)

        while True:
            try:
                await perform_scan(exchange, tg, state, notify=True)
                await monitor(exchange, tg, state)
                save_state(state)
            except Exception as exc:
                logging.exception("MAIN LOOP ERROR: %s", exc)

            await asyncio.sleep(POLL_SECONDS)

    finally:
        await exchange.close()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as exc:
        logging.exception("FATAL ERROR: %s", exc)
        raise
