import os, asyncio, logging, json, math
from datetime import datetime, timezone
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()
TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")
EXCHANGE_ID=os.getenv("EXCHANGE","binance").strip().lower()
SYMBOLS=[x.strip() for x in os.getenv("SYMBOLS","BTC/USDT,ETH/USDT,SOL/USDT").split(",") if x.strip()]
TIMEFRAMES=[x.strip() for x in os.getenv("TIMEFRAMES","5m,15m,1h,4h").split(",") if x.strip()]
PRIMARY_TF=os.getenv("PRIMARY_TF","15m")
CANDLE_LIMIT=int(os.getenv("CANDLE_LIMIT","300"))
POLL_SECONDS=int(os.getenv("POLL_SECONDS","60"))
MIN_SCORE=int(os.getenv("MIN_SCORE","70"))
RR=float(os.getenv("RISK_REWARD","2.0"))
ATR_MULT=float(os.getenv("ATR_SL_MULT","1.2"))
TRAIL_ATR=float(os.getenv("TRAIL_ATR_MULT","1.0"))
STATE_FILE="state.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def fmt(n):
    return f"{n:,.4f}" if n < 100 else f"{n:,.2f}"

def indicators(df):
    c=df.close
    df["ema20"]=c.ewm(span=20,adjust=False).mean()
    df["ema50"]=c.ewm(span=50,adjust=False).mean()
    df["ema200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff()
    gain=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    df["rsi"]=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26
    df["macd_signal"]=df.macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),
                  (df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    df["atr"]=tr.rolling(14).mean()
    df["vol_ma"]=df.volume.rolling(20).mean()
    return df

def analyze(df):
    df=indicators(df).dropna()
    x=df.iloc[-1]; p=float(x.close); atr=float(x.atr)
    ls=ss=0; reasons=[]
    if x.ema20>x.ema50>x.ema200: ls+=25; reasons.append("EMA bullish")
    if x.ema20<x.ema50<x.ema200: ss+=25; reasons.append("EMA bearish")
    if p>x.ema200: ls+=10
    if p<x.ema200: ss+=10
    if 50<=x.rsi<=68: ls+=15; reasons.append("RSI long")
    if 32<=x.rsi<=50: ss+=15; reasons.append("RSI short")
    if x.rsi<30: ls+=8; reasons.append("oversold")
    if x.rsi>70: ss+=8; reasons.append("overbought")
    if x.macd>x.macd_signal: ls+=15; reasons.append("MACD bullish")
    if x.macd<x.macd_signal: ss+=15; reasons.append("MACD bearish")
    if x.volume>1.15*x.vol_ma:
        if x.close>x.open: ls+=10; reasons.append("volume")
        elif x.close<x.open: ss+=10; reasons.append("volume")
    hi=df.high.iloc[-21:-1].max(); lo=df.low.iloc[-21:-1].min()
    if p>hi: ls+=15; reasons.append("breakout")
    if p<lo: ss+=15; reasons.append("breakdown")
    score=max(ls,ss)
    if score<MIN_SCORE:
        return {"action":"NO TRADE","score":score,"price":p,"rsi":float(x.rsi),"atr":atr}
    side="LONG" if ls>ss else "SHORT"
    dist=max(ATR_MULT*atr,p*0.003)
    sl=p-dist
    tp=p+dist*RR if side=="LONG" else p-dist*RR
    return {"action":side,"score":score,"price":p,"sl":sl,"tp":tp,"atr":atr,"rsi":float(x.rsi),"reasons":reasons}

def load_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return {"positions":{}, "signals":[]}

def save_state(s):
    with open(STATE_FILE,"w") as f: json.dump(s,f,indent=2)

async def candles(exchange,symbol,tf,limit=CANDLE_LIMIT):
    d=await exchange.fetch_ohlcv(symbol,tf,limit=limit)
    return pd.DataFrame(d,columns=["time","open","high","low","close","volume"])

async def start_cmd(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Crypto Signal Bot active.\n\n"
        "/status — open positions & bot status\n"
        "/signals — recent signals\n"
        "/start — help\n\n"
        "Signal-only mode: it does not place exchange orders."
    )

async def status_cmd(update,context):
    s=load_state()
    pos=s["positions"]
    if not pos:
        text="📊 STATUS\n\nNo open tracked positions."
    else:
        text="📊 STATUS\n\n"
        for k,p in pos.items():
            text+=f"{k}: {p['side']} | Entry {fmt(p['entry'])} | SL {fmt(p['sl'])} | TP {fmt(p['tp'])}\n"
    await update.message.reply_text(text)

async def ping_cmd(update,context):
    await update.message.reply_text("🟢 Bot is online and responding.")

async def signals_cmd(update,context):
    s=load_state()
    recent=s["signals"][-10:]
    if not recent:
        await update.message.reply_text("No signals yet.")
        return
    text="📜 RECENT SIGNALS\n\n"
    for x in recent:
        text+=f"{x['time']} {x['symbol']} {x['side']} @ {fmt(x['entry'])} ({x['score']}%)\n"
    await update.message.reply_text(text)

def update_trailing(pos, price):
    atr=pos["atr"]
    if pos["side"]=="LONG":
        pos["best"]=max(pos.get("best",pos["entry"]),price)
        pos["sl"]=max(pos["sl"],pos["best"]-TRAIL_ATR*atr)
    else:
        pos["best"]=min(pos.get("best",pos["entry"]),price)
        pos["sl"]=min(pos["sl"],pos["best"]+TRAIL_ATR*atr)
    return pos

async def monitor(exchange,tg,state):
    for key,pos in list(state["positions"].items()):
        symbol=pos["symbol"]
        try:
            ticker=await exchange.fetch_ticker(symbol)
            price=float(ticker["last"])
            pos=update_trailing(pos,price)
            close_reason=None
            if pos["side"]=="LONG":
                if price<=pos["sl"]: close_reason="Trailing/Stop Loss"
                elif price>=pos["tp"]: close_reason="Take Profit"
            else:
                if price>=pos["sl"]: close_reason="Trailing/Stop Loss"
                elif price<=pos["tp"]: close_reason="Take Profit"
            if close_reason:
                pnl=(price-pos["entry"]) if pos["side"]=="LONG" else (pos["entry"]-price)
                await tg.send_message(chat_id=CHAT_ID,
                    text=f"🔔 POSITION CLOSE\n\n{symbol}\nSide: {pos['side']}\nClose: {fmt(price)}\nReason: {close_reason}\nApprox move: {pnl:+.4f}")
                del state["positions"][key]
        except Exception:
            logging.exception("monitor failed for %s",symbol)

async def scan(exchange,tg,state):
    for symbol in SYMBOLS:
        try:
            # Multi-timeframe confirmation: primary signal must agree with at least one other TF.
            results={}
            for tf in TIMEFRAMES:
                df=await candles(exchange,symbol,tf)
                results[tf]=analyze(df)
            sig=results.get(PRIMARY_TF) or results[TIMEFRAMES[0]]
            if sig["action"]=="NO TRADE": continue
            confirmations=sum(1 for tf,r in results.items() if r["action"]==sig["action"])
            if confirmations < 2: continue
            key=f"{symbol}:{PRIMARY_TF}"
            existing=state["positions"].get(key)
            if existing:
                continue
            state["positions"][key]={
                "symbol":symbol,"side":sig["action"],"entry":sig["price"],
                "sl":sig["sl"],"tp":sig["tp"],"atr":sig["atr"],"best":sig["price"],
                "opened":datetime.now(timezone.utc).isoformat()
            }
            state["signals"].append({
                "time":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "symbol":symbol,"side":sig["action"],"entry":sig["price"],"score":sig["score"]
            })
            state["signals"]=state["signals"][-100:]
            await tg.send_message(chat_id=CHAT_ID,
                text=f"🚨 MULTI-TF SIGNAL\n\nPair: {symbol}\nPosition: {sig['action']}\n"
                     f"Entry: {fmt(sig['price'])}\nSL: {fmt(sig['sl'])}\nTP: {fmt(sig['tp'])}\n"
                     f"Score: {sig['score']}%\nConfirmed TFs: {confirmations}/{len(TIMEFRAMES)}\n"
                     f"RSI: {sig['rsi']:.1f}\nReasons: {', '.join(sig.get('reasons',[]))}")
        except Exception:
            logging.exception("scan failed for %s",symbol)

def backtest_df(df):
    df=indicators(df).dropna().reset_index(drop=True)
    if len(df) < 30:
        return {"trades":0,"win_rate":0,"net_pct":0,"avg_pct":0}
    trades=[]; pos=None
    for i in range(0,len(df)):
        if pos is None:
            sig=analyze(df.iloc[:i+1].copy())
            if sig["action"]=="NO TRADE": continue
            pos={"side":sig["action"],"entry":sig["price"],"sl":sig["sl"],"tp":sig["tp"],"atr":sig["atr"],"best":sig["price"]}
        else:
            row=df.iloc[i]; high=float(row.high); low=float(row.low)
            if pos["side"]=="LONG":
                pos["best"]=max(pos["best"],high); pos["sl"]=max(pos["sl"],pos["best"]-TRAIL_ATR*pos["atr"])
                exitp=pos["sl"] if low<=pos["sl"] else (pos["tp"] if high>=pos["tp"] else None)
                pnl=(exitp-pos["entry"])/pos["entry"]*100 if exitp else None
            else:
                pos["best"]=min(pos["best"],low); pos["sl"]=min(pos["sl"],pos["best"]+TRAIL_ATR*pos["atr"])
                exitp=pos["sl"] if high>=pos["sl"] else (pos["tp"] if low<=pos["tp"] else None)
                pnl=(pos["entry"]-exitp)/pos["entry"]*100 if exitp else None
            if pnl is not None:
                trades.append(pnl); pos=None
    if not trades: return {"trades":0,"win_rate":0,"net_pct":0,"avg_pct":0}
    wins=sum(x>0 for x in trades)
    return {"trades":len(trades),"win_rate":wins/len(trades)*100,"net_pct":sum(trades),"avg_pct":sum(trades)/len(trades)}

async def run_backtests(exchange,tg):
    lines=["🧪 BACKTEST REPORT"]
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df=await candles(exchange,symbol,tf,1000)
                r=backtest_df(df)
                lines.append(f"{symbol} {tf}: {r['trades']} trades | Win {r['win_rate']:.1f}% | Net {r['net_pct']:.2f}%")
            except Exception as e:
                logging.exception("backtest failed for %s %s", symbol, tf)
                lines.append(f"{symbol} {tf}: error — {type(e).__name__}: {str(e)[:100]}")
    await tg.send_message(chat_id=CHAT_ID,text="\n".join(lines))

async def main():
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Railway Variables")
    if EXCHANGE_ID not in {"binance","kraken","coinbase","okx","bybit","kucoin","gateio","bitget"}:
        raise RuntimeError(f"Unsupported EXCHANGE={EXCHANGE_ID}")
    # CCXT async exchange classes are not exposed consistently as direct
    # attributes across package versions. Use a module fallback for Binance.
    import importlib
    try:
        exchange_module = importlib.import_module(f"ccxt.async_support.{EXCHANGE_ID}")
        excls = getattr(exchange_module, EXCHANGE_ID)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"CCXT exchange '{EXCHANGE_ID}' could not be loaded: {e}"
        ) from e
    exchange=excls({"enableRateLimit":True})
    app=Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start_cmd))
    app.add_handler(CommandHandler("status",status_cmd))
    app.add_handler(CommandHandler("signals",signals_cmd))
    app.add_handler(CommandHandler("ping",ping_cmd))
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    tg=app.bot; state=load_state()
    try:
        counter=0
        while True:
            try:
                await scan(exchange,tg,state)
                await monitor(exchange,tg,state)
                save_state(state)
            except Exception as exc:
                logging.exception("MAIN LOOP ERROR: %s", exc)
            counter+=1
            if counter % max(1,int(3600/POLL_SECONDS))==0:
                await run_backtests(exchange,tg)
            await asyncio.sleep(POLL_SECONDS)
    finally:
        await exchange.close()
        await app.updater.stop(); await app.stop(); await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as exc:
        logging.exception("FATAL STARTUP/RUNTIME ERROR: %s", exc)
        raise
