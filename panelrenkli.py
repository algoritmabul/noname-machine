"""Local OHLCV research dashboard inspired by the supplied trading-panel video."""
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from veri_periyot import VeriPeriyot

GLM_DIZINI = r"D:\new\yeniden\SMOM\GLM"
if os.path.isdir(GLM_DIZINI) and GLM_DIZINI not in sys.path:
    sys.path.insert(0, GLM_DIZINI)
try:
    from indrenk import Indrenk
except Exception:
    Indrenk = None

# ---- FIBO KANAL ENTEGRASYONU (başlangıç) --------------------------------
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, NullFormatter

FIBO_MOD = "matriks"   # fibo_kanal CLI varsayılanı (--mod). Diğerleri: "kanal" | "aralik" | "aralik_asof"
_DOSYA_DIZINI = os.path.dirname(os.path.abspath(__file__))
if _DOSYA_DIZINI and _DOSYA_DIZINI not in sys.path:
    sys.path.insert(0, _DOSYA_DIZINI)
try:
    from fibo_kanal import (
        PERIYOT_CAPA,
        kanal_hesapla,
        seviye_adi,
        veri_yukle,
    )
    FIBO_KANAL_OK = True
    FIBO_IMPORT_HATASI = ""
except Exception as _fibo_hata:
    FIBO_KANAL_OK = False
    FIBO_IMPORT_HATASI = str(_fibo_hata)

matplotlib.use("TkAgg")   # fibo_kanal backend'i Agg'e çevirebilir; panelde TkAgg kalmalı
# ---- FIBO KANAL ENTEGRASYONU (bitiş) --------------------------------------

# ================= MODERN DARK THEME RENK PALETİ =================
BG = "#121214"         # Ana arka plan (Koyu Antrasit)
PANEL = "#1e1f24"      # Kartlar ve Grafik arka planı
PANEL_ALT = "#282a31"  # Inputlar, Tablo başlıkları
GRID = "#363945"       # Çizgiler, Kenarlıklar
TEXT = "#e4e6eb"       # Ana yazı rengi (Beyaza yakın)
MUTED = "#8b8d98"      # İkincil yazı rengi (Gri)
GREEN = "#22c55e"      # Yükseliş / Pozitif (Modern Yeşil)
RED = "#ef4444"        # Düşüş / Negatif (Modern Kırmızı)
BLUE = "#3b82f6"       # Bilgi / Nötr (Modern Mavi)
GOLD = "#f59e0b"       # Uyarı / Vurgu (Modern Altın/Sarı)

PERIYOTLAR = ["60dk", "240dk", "gunluk", "haftalik", "aylik"]
ISLEM_MALIYETI = 0.0010  # 0.1% per side (commission + slippage)
OOS_ORAN = 0.30  # last 30% of the series is out-of-sample

# Dark theme için daha parlak aile renkleri
AILE_RENKLERI = {
    "Trend": "#f87171",
    "Mean reversion": "#fb923c",
    "Breakout": "#fbbf24",
    "Volume edge": "#4ade80",
}

RESEARCH_PANELLERI = ("mesh", "loop", "vol")
PANEL_SATIRLARI = (
    (("grid", "wide"), ("orders", "narrow")),
    (("renk", "left"), None),
    (("macd", "wide"), ("plan", "narrow")),
    (("fibo", "wide"), None),                      # FIBO KANAL
    (("equity", "wide"), ("risk", "narrow")),
)

def sayi_yaz(value, decimals=2):
    if pd.isna(value):
        return "—"
    value = float(value)
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "§").replace(".", ",").replace("§", ".")

def sembolleri_bul(veri):
    symbols = set()
    for directory in (veri.csv_60dk_dizini, veri.csv_gunluk_dizini):
        if directory.exists():
            symbols.update(path.stem.upper() for path in directory.glob("*.csv"))
    return sorted(symbols)

def yillik_bar_sayisi(periyot):
    return {
        "60dk": 252 * 9,
        "240dk": 252 * 2,
        "gunluk": 252,
        "haftalik": 52,
        "aylik": 12,
    }.get(periyot, 252)

def mc_ufku(periyot):
    """Bootstrap horizon covering roughly one quarter of trading days per period."""
    return {
        "60dk": 540,
        "240dk": 120,
        "gunluk": 60,
        "haftalik": 12,
        "aylik": 3,
    }.get(periyot, 60)

SINYAL_AL_ESIK = 2
SINYAL_SAT_ESIK = -2

def macd_serileri(df, hizli=12, yavas=26, sinyal=9):
    close = df["Close"].astype(float)
    macd = close.ewm(span=hizli, adjust=False).mean() - close.ewm(span=yavas, adjust=False).mean()
    signal = macd.ewm(span=sinyal, adjust=False).mean()
    return macd, signal, macd - signal

def atr_hesapla(df, pencere=14):
    previous_close = df["Close"].shift(1)
    parts = pd.concat(
        [df["High"] - df["Low"], (df["High"] - previous_close).abs(), (df["Low"] - previous_close).abs()],
        axis=1,
    )
    value = float(parts.max(axis=1).tail(pencere).mean())
    return value if np.isfinite(value) and value > 0 else 0.0

def giris_plani(df, gunluk_df=None):
    """Entry setup for a long trade: trigger, invalidation, size and quality."""
    daily = gunluk_df if gunluk_df is not None and len(gunluk_df) else df
    upto = daily.index <= df.index[-1]
    if upto.any():
        daily = daily[upto]
    close = daily["Close"].astype(float)
    high = daily["High"].astype(float)
    low = daily["Low"].astype(float)
    volume = daily["Volume"].fillna(0).astype(float)
    last = float(close.iloc[-1])
    atr = atr_hesapla(daily)
    if atr <= 0:
        atr = max(last * 0.01, 0.01)
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    high20 = float(high.tail(20).max())
    low20 = float(low.tail(20).min())
    swing_low = float(low.tail(5).min())
    average_volume = float(volume.tail(20).mean())
    volume_boost = float(volume.tail(5).mean()) / average_volume if average_volume > 0 else 1.0
    extension = (last - ema20) / atr

    if last < ema50 < ema200:
        setup, trend = "YOK", "DOWN"
    elif last > ema50 > ema200:
        trend = "UP"
        if extension > 1.5:
            setup = "GEÇ"
        elif last >= high20 - 0.35 * atr:
            setup = "KIRILIM"
        elif abs(extension) <= 0.7:
            setup = "GERİ ÇEKİLME"
        else:
            setup = "BEKLE"
    else:
        trend, setup = "MIXED", "BEKLE"

    trigger = None
    trigger_text = "—"
    if setup == "KIRILIM":
        trigger = high20 + 0.10 * atr
        trigger_text = f"{sayi_yaz(trigger)} üstü kapanış"
    elif setup == "GERİ ÇEKİLME":
        trigger = max(last, ema20 + 0.10 * atr)
        if trigger > last * 1.001:
            trigger_text = f"{sayi_yaz(trigger)} üstü kapanış"
        else:
            trigger_text = f"{sayi_yaz(trigger)} piyasa"
    elif setup == "GEÇ":
        trigger = ema20 + 0.50 * atr
        trigger_text = f"EMA20 {sayi_yaz(ema20)} geri çekilmesi"

    stop = target = risk_per_share = 0.0
    shares = 0
    risk_atr = 0.0
    if trigger is not None:
        stop = max(swing_low, trigger - 1.5 * atr)
        stop = min(stop, trigger - 0.3 * atr)
        risk_per_share = trigger - stop
        target = trigger + 2.0 * risk_per_share
        shares = int(1000.0 / risk_per_share) if risk_per_share > 0 else 0
        risk_atr = risk_per_share / atr

    quality = 30 if trend == "UP" else (10 if trend == "MIXED" else 0)
    quality += {"KIRILIM": 20, "GERİ ÇEKİLME": 25, "GEÇ": 5}.get(setup, 0)
    quality += 15 if volume_boost >= 1.2 else (8 if volume_boost >= 0.8 else 0)
    atr_pct = atr / last * 100.0
    if 2.0 <= atr_pct <= 8.0:
        quality += 15
    elif atr_pct < 2.0:
        quality += 10
    elif atr_pct <= 12.0:
        quality += 6
    if trigger is not None:
        if risk_atr <= 1.6:
            quality += 15
        elif risk_atr <= 2.2:
            quality += 8
    quality = min(quality, 100)

    if setup == "YOK":
        status, status_color = "NO TRADE", RED
    elif setup == "BEKLE":
        status, status_color = "WATCH", GOLD
    elif setup == "GEÇ":
        status, status_color = "WAIT · PULLBACK", GOLD
    elif quality >= 60:
        status, status_color = "READY", GREEN
    else:
        status, status_color = "WATCH", GOLD

    return {
        "setup": setup, "status": status, "status_color": status_color, "quality": quality,
        "trigger": trigger, "trigger_text": trigger_text, "stop": stop, "target": target,
        "risk_per_share": risk_per_share, "shares": shares, "atr": atr, "atr_pct": atr_pct,
        "extension": extension, "high20": high20, "low20": low20, "ema20": ema20, "trend": trend,
        "volume_boost": volume_boost,
    }

def sinyal_hesapla(df, analysis, gunluk_df=None):
    """Rule-based verdict from four votes: daily trend, EMA20 momentum, MACD, model state."""
    close = df["Close"].astype(float)
    last = float(close.iloc[-1])
    daily = gunluk_df if gunluk_df is not None and len(gunluk_df) else df
    upto = daily.index <= df.index[-1]
    if upto.any():
        daily = daily[upto]
    daily_close = daily["Close"].astype(float)
    ema50 = float(daily_close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(daily_close.ewm(span=200, adjust=False).mean().iloc[-1])

    if last > ema50 > ema200:
        trend, trend_text = 1, "trend ↑"
    elif last < ema50 < ema200:
        trend, trend_text = -1, "trend ↓"
    else:
        trend, trend_text = 0, "trend yatay"

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    momentum, momentum_text = (1, "fiyat > EMA20") if last > ema20 else (-1, "fiyat < EMA20")
    macd, signal_line, histogram = macd_serileri(df)
    son_macd = float(macd.iloc[-1])
    son_hist = float(histogram.iloc[-1])
    if son_hist > 0 and son_macd > 0:
        macd_vote, macd_text = 1, "MACD ↑"
    elif son_hist < 0 and son_macd < 0:
        macd_vote, macd_text = -1, "MACD ↓"
    else:
        macd_vote, macd_text = 0, "MACD kararsız"

    leader = analysis["leader"]
    model_long = bool(len(leader["position"]) and leader["position"][-1] > 0)
    model_vote = 1 if model_long else 0
    model_text = "model long" if model_long else "model flat"

    score = trend + momentum + macd_vote + model_vote
    if score >= SINYAL_AL_ESIK:
        verdict, color = "AL", GREEN
    elif score <= SINYAL_SAT_ESIK:
        verdict, color = "SAT", RED
    else:
        verdict, color = "KARARSIZ", GOLD

    atr = atr_hesapla(daily)
    return {
        "verdict": verdict, "score": score, "color": color,
        "reasons": "  ·  ".join((trend_text, momentum_text, macd_text, model_text)),
        "atr_pct": (atr / last * 100.0) if last else 0.0,
    }

def islem_istatistikleri(position, returns):
    """Per-trade statistics from a 0/1 position series and its net returns."""
    pos = np.asarray(position, dtype=float)
    r = np.asarray(returns, dtype=float)
    empty = {"win_rate": 0.0, "profit_factor": 0.0, "exposure": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "trade_count": 0}
    if len(pos) == 0 or len(r) != len(pos):
        return empty
    changes = np.diff(np.r_[0.0, pos, 0.0])
    entries = np.flatnonzero(changes > 0)
    exits = np.flatnonzero(changes < 0)
    if len(exits) < len(entries):
        exits = np.r_[exits, len(pos)]
    if len(entries) == 0:
        return {**empty, "exposure": float(np.mean(pos > 0))}
    growth = np.cumprod(np.maximum(1.0 + r, 1e-9))
    previous = np.r_[1.0, growth[:-1]]
    pnl = growth[np.minimum(exits, len(pos)) - 1] / previous[entries] - 1.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss > 1e-12:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    return {
        "win_rate": float(len(wins) / len(pnl) * 100.0),
        "profit_factor": profit_factor,
        "exposure": float(np.mean(pos > 0) * 100.0),
        "avg_win": float(wins.mean() * 100.0) if len(wins) else 0.0,
        "avg_loss": float(losses.mean() * 100.0) if len(losses) else 0.0,
        "trade_count": int(len(pnl)),
    }

def sinyal_geri_testi(df, analysis, periyot, gunluk_df=None, ileri=5):
    """Historical hit rate of the AL/SAT rule: forward return after each verdict."""
    close = df["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    momentum = np.where(close.to_numpy() > ema20.to_numpy(), 1, -1)
    macd, signal_line, histogram = macd_serileri(df)
    macd_vote = np.where(
        (histogram > 0) & (macd > 0), 1,
        np.where((histogram < 0) & (macd < 0), -1, 0),
    )
    daily = gunluk_df if gunluk_df is not None and len(gunluk_df) else df
    daily_close = daily["Close"].astype(float)
    ema50 = daily_close.ewm(span=50, adjust=False).mean()
    ema200 = daily_close.ewm(span=200, adjust=False).mean()
    trend_daily = np.where(
        (daily_close > ema50) & (ema50 > ema200), 1,
        np.where((daily_close < ema50) & (ema50 < ema200), -1, 0),
    )
    if periyot in ("60dk", "240dk"):
        keys = df.index.normalize()
        daily_index = daily.index.searchsorted(keys, side="left") - 1
    else:
        daily_index = daily.index.searchsorted(df.index, side="right") - 1
    daily_index = np.clip(daily_index, 0, len(trend_daily) - 1)
    trend = trend_daily[daily_index]
    model = (np.asarray(analysis["leader"]["position"], dtype=float) > 0).astype(int)
    score = trend + momentum + macd_vote + model
    forward = close.pct_change(ileri).shift(-ileri).to_numpy()
    valid = np.isfinite(forward)

    def summarize(mask):
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {"n": 0, "hit": 0.0, "avg": 0.0}
        values = forward[mask]
        return {"n": count, "hit": float(np.mean(values > 0) * 100.0), "avg": float(np.mean(values) * 100.0)}

    al_mask = (score >= SINYAL_AL_ESIK) & valid
    sat_mask = (score <= SINYAL_SAT_ESIK) & valid
    baseline = float(np.nanmean(forward[valid]) * 100.0) if np.any(valid) else 0.0
    return {"al": summarize(al_mask), "sat": summarize(sat_mask), "baseline": baseline, "forward": ileri}

def kuralli_pozisyon(giris, cikis):
    """Create a long/flat position and shift it one bar to avoid look-ahead."""
    state = 0.0
    raw = np.zeros(len(giris), dtype=float)
    for index, (enter, leave) in enumerate(zip(giris, cikis)):
        if state == 0.0 and enter:
            state = 1.0
        elif state == 1.0 and leave:
            state = 0.0
        raw[index] = state
    if len(raw) > 1:
        return np.r_[0.0, raw[:-1]]
    return raw

def strateji_adaylarini_hesapla(df, periyot):
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].fillna(0).astype(float)
    asset_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    annual_bars = yillik_bar_sayisi(periyot)
    candidates = []

    def add_candidate(name, family, position):
        position = np.asarray(position, dtype=float)
        costs = np.abs(np.diff(np.r_[0.0, position])) * ISLEM_MALIYETI
        strategy_returns = asset_returns * position - costs
        equity = np.cumprod(np.maximum(1.0 + strategy_returns, 0.0))
        peaks = np.maximum.accumulate(equity)
        drawdown = np.divide(equity, peaks, out=np.ones_like(equity), where=peaks != 0) - 1.0
        total_return = float(equity[-1] - 1.0) if len(equity) else 0.0
        max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
        standard_deviation = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
        sharpe = (
            float(np.mean(strategy_returns) / standard_deviation * np.sqrt(annual_bars))
            if standard_deviation > 1e-12 else 0.0
        )
        trades = int(np.count_nonzero(np.diff(np.r_[0.0, position]) > 0))
        split = int(len(strategy_returns) * (1.0 - OOS_ORAN))
        is_returns = strategy_returns[:split]
        oos_returns = strategy_returns[split:]
        is_return = float(np.prod(1.0 + is_returns) - 1.0) if len(is_returns) else 0.0
        oos_return = float(np.prod(1.0 + oos_returns) - 1.0) if len(oos_returns) else 0.0
        if len(is_returns) > 1:
            is_std = float(np.std(is_returns, ddof=1))
            is_sharpe = float(np.mean(is_returns) / is_std * np.sqrt(annual_bars)) if is_std > 1e-12 else 0.0
            is_equity = np.cumprod(np.maximum(1.0 + is_returns, 0.0))
            is_peaks = np.maximum.accumulate(is_equity)
            is_drawdown = float(
                np.min(np.divide(is_equity, is_peaks, out=np.ones_like(is_equity), where=is_peaks != 0) - 1.0)
            )
        else:
            is_sharpe = 0.0
            is_drawdown = 0.0
        stats = islem_istatistikleri(position, strategy_returns)
        score = is_sharpe + is_return - abs(is_drawdown)
        candidates.append({
            "name": name, "family": family, "position": position, "returns": strategy_returns,
            "equity": equity, "total_return": total_return, "max_drawdown": max_drawdown,
            "sharpe": sharpe, "trades": trades, "is_return": is_return, "oos_return": oos_return,
            "score": score, **stats,
        })

    for fast in (5, 8, 13, 21, 34):
        for slow in (34, 55, 89):
            if fast >= slow: continue
            fast_avg = close.rolling(fast, min_periods=fast).mean()
            slow_avg = close.rolling(slow, min_periods=slow).mean()
            position = (fast_avg > slow_avg).fillna(False).astype(float).to_numpy()
            if len(position) > 1: position = np.r_[0.0, position[:-1]]
            add_candidate(f"MA {fast}/{slow}", "Trend", position)

    for window in (10, 20, 30, 50):
        average = close.rolling(window, min_periods=window).mean()
        deviation = close.rolling(window, min_periods=window).std().replace(0, np.nan)
        z_score = (close - average) / deviation
        for threshold in (1.0, 1.5, 2.0):
            enter = (z_score < -threshold).fillna(False).to_numpy()
            leave = (z_score >= 0).fillna(False).to_numpy()
            add_candidate(f"MR {window}/{threshold:g}σ", "Mean reversion", kuralli_pozisyon(enter, leave))

    for window in (10, 20, 40, 60):
        prior_high = high.rolling(window, min_periods=window).max().shift(1)
        prior_low = low.rolling(window, min_periods=window).min().shift(1)
        enter = (close > prior_high).fillna(False).to_numpy()
        leave = (close < prior_low).fillna(False).to_numpy()
        add_candidate(f"BO {window}", "Breakout", kuralli_pozisyon(enter, leave))

    for window in (10, 20, 40):
        prior_high = high.rolling(window, min_periods=window).max().shift(1)
        prior_low = low.rolling(window, min_periods=window).min().shift(1)
        average_volume = volume.rolling(window, min_periods=window).mean().shift(1)
        for multiplier in (1.5, 2.0):
            enter = ((close > prior_high) & (volume > average_volume * multiplier)).fillna(False).to_numpy()
            leave = (close < prior_low).fillna(False).to_numpy()
            add_candidate(f"VOL {window}/{multiplier:g}x", "Volume edge", kuralli_pozisyon(enter, leave))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    survivors = [item for item in candidates if item["is_return"] > 0 and item["max_drawdown"] > -0.50 and item["trades"] >= 2]
    leader = max(survivors or candidates, key=lambda item: item["score"])
    return candidates, survivors, leader

def monte_carlo_hesapla(returns, seed, path_count=2048, horizon=90):
    history = np.asarray(returns, dtype=float)
    history = history[np.isfinite(history)]
    if len(history) == 0: history = np.array([0.0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(history, size=(path_count, horizon), replace=True)
    simulated = 100.0 * np.cumprod(np.maximum(1.0 + sampled, 0.0), axis=1)
    paths = np.column_stack((np.full(path_count, 100.0), simulated))
    terminal = paths[:, -1]
    return {
        "paths": paths, "terminal": terminal,
        "p05": float(np.percentile(terminal, 5) - 100.0),
        "median": float(np.percentile(terminal, 50) - 100.0),
        "p95": float(np.percentile(terminal, 95) - 100.0),
        "loss_probability": float(np.mean(terminal < 100.0) * 100.0),
        "path_count": path_count, "horizon": horizon,
    }

class PiyasaPaneli(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NO NAME Machine · BIST")
        
        # Pencere boyut ve konum ayarları
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.screen_width = screen_width
        self.screen_height = screen_height
        window_width = max(700, min(1500, screen_width - 32))
        window_height = max(560, min(1000, screen_height - 72))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, (screen_height - window_height) // 2)
        self.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        self.minsize(min(1180, window_width), min(700, window_height))
        self.configure(bg=BG)

        self.veri = VeriPeriyot()
        self.semboller = sembolleri_bul(self.veri)
        self._kart_degerleri = {}
        self.stage_buttons = []
        self.active_stage = 2
        self._updating_candidates = False
        self._animation_active = False
        self._animation_job = None
        self._animation_indices = []
        self._animation_frame = 0
        self.analiz = None
        self.df = None
        self.gunluk_df = None
        self._renk_key = None
        self._renk_verisi = None
        self._fibo_onbellek = None   
        self._panel_gorunur = {
            "grid": True, "orders": True, "macd": True, "fibo": True, "plan": True,
            "renk": True, "equity": True, "risk": True, "mesh": False, "loop": False, "vol": False,
        }
        
        self._stil_ayarla()
        self._arayuz_kur()
        self.bind_all("<Control-Return>", self._animasyon_kisayol)

        try:
            from panel_tvdata_ekleri import tv_butonu_ekle
            self._tv_button = tv_butonu_ekle(self)
        except Exception as _tv_hata:
            self._tv_button = None
            print(f"TV veri eki yüklenemedi: {_tv_hata}")

        if not self.semboller:
            messagebox.showerror(
                "Veri bulunamadı",
                "CSV klasörlerinde sembol dosyası bulunamadı.\n\n"
                f"60dk: {self.veri.csv_60dk_dizini}\n"
                f"Günlük: {self.veri.csv_gunluk_dizini}",
                parent=self,
            )
            self.destroy()
            return

        default_symbol = "THYAO" if "THYAO" in self.semboller else self.semboller[0]
        self.sembol_var.set(default_symbol)
        self.periyot_var.set("gunluk")
        self.after(100, self.veri_yukle)

    def _gulen_yuz_animasyonu(self):
        if not self.winfo_exists(): return
        faces = ["😊", "😄", "😁", "😃", "🙂"]
        self._smile_index = (self._smile_index + 1) % len(faces)
        self.smile_var.set(faces[self._smile_index])
        self.after(180, self._gulen_yuz_animasyonu)

    def _stil_ayarla(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # Dark Theme Combobox
        style.configure(
            "TCombobox",
            fieldbackground=PANEL_ALT,
            background=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=GRID,
            lightcolor=GRID,
            darkcolor=GRID,
            padding=7,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_ALT)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", BLUE)],
            selectforeground=[("readonly", TEXT)],
        )
        
        # Dark Theme Treeview
        style.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=26,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=MUTED,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", TEXT)])

    def _arayuz_kur(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # HEADER
        header = tk.Frame(self, bg=BG, padx=24, pady=16)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_wrap = tk.Frame(header, bg=BG)
        title_wrap.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_wrap, text="NØ NΛMΞ -꧁༒ MACHINE༒꧂", bg=BG, fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")
        
        self._smile_index = 0
        self.smile_var = tk.StringVar(value="😊")
        tk.Label(
            title_wrap, textvariable=self.smile_var, bg=BG, fg=GOLD,
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.after(180, self._gulen_yuz_animasyonu)

        tk.Label(
            header, text="BIST  ·  LOCAL OHLCV  ·  HISTORICAL RESEARCH  ·  PAPER SIMULATION",
            bg=BG, fg=MUTED, font=("Segoe UI", 9, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        controls = tk.Frame(header, bg=BG)
        controls.grid(row=0, column=1, rowspan=2, sticky="e")
        
        tk.Label(controls, text="SYMBOL", bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.sembol_var = tk.StringVar()
        self.sembol_combo = ttk.Combobox(
            controls, textvariable=self.sembol_var, values=self.semboller, width=16, state="normal",
        )
        self.sembol_combo.grid(row=1, column=0, padx=(0, 12))
        self.sembol_combo.bind("<Return>", lambda _event: self.veri_yukle())

        tk.Label(controls, text="PERIOD", bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).grid(
            row=0, column=1, sticky="w", padx=(0, 8)
        )
        self.periyot_var = tk.StringVar()
        self.periyot_combo = ttk.Combobox(
            controls, textvariable=self.periyot_var, values=PERIYOTLAR, width=11, state="readonly",
        )
        self.periyot_combo.grid(row=1, column=1, padx=(0, 12))
        self.periyot_combo.bind("<<ComboboxSelected>>", lambda _event: self.veri_yukle())

        self.load_button = tk.Button(
            controls, text="LOAD DATA", command=self.veri_yukle,
            bg=BLUE, fg="white", activebackground="#2563eb", activeforeground="white",
            relief="flat", padx=16, pady=9, font=("Segoe UI", 9, "bold"), cursor="hand2",
        )
        self.load_button.grid(row=1, column=2)

        # METRICS CARDS
        metrics = tk.Frame(self, bg=BG, padx=20, pady=4)
        metrics.grid(row=1, column=0, sticky="ew")
        for column in range(6):
            metrics.grid_columnconfigure(column, weight=1, uniform="metric")

        cards = [
            ("CONFIGS TESTED", "configs"), ("KILL RATE", "kill_rate"), ("SURVIVORS", "survivors"),
            ("TOP MODEL", "top_model"), ("BEST RETURN", "best_return"), ("MAX DRAWDOWN", "max_dd"),
        ]
        for column, (label, key) in enumerate(cards):
            card = tk.Frame(metrics, bg=PANEL, padx=13, pady=10, highlightthickness=1, highlightbackground=GRID)
            card.grid(row=0, column=column, sticky="ew", padx=4, pady=4)
            tk.Label(card, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            value = tk.StringVar(value="—")
            self._kart_degerleri[key] = value
            tk.Label(card, textvariable=value, bg=PANEL, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(
                anchor="w", pady=(5, 0)
            )

        # STAGES
        stage_frame = tk.Frame(self, bg=BG, padx=24, pady=8)
        stage_frame.grid(row=2, column=0, sticky="ew")
        stage_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            stage_frame, text="CLOSED-LOOP SEARCH  ·  GENERATE → BACKTEST → SCORE → REFINE → GATE → PAPER DEPLOY",
            bg=BG, fg=MUTED, font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))

        stages = ["01 RESEARCH", "02 CODE", "03 BACKTEST", "04 LIVE", "05 POST-MORTEM", "06 FINE-TUNE"]
        stage_strip = tk.Frame(stage_frame, bg=BG)
        stage_strip.grid(row=1, column=0, sticky="ew")
        for index in range(len(stages)):
            stage_strip.grid_columnconfigure(index, weight=1, uniform="stage")

        for index, label in enumerate(stages):
            active = index == 2
            button = tk.Button(
                stage_strip, text=label,
                bg=RED if active else PANEL, fg="white" if active else MUTED,
                activebackground=RED, activeforeground="white",
                font=("Segoe UI", 8, "bold"), padx=13, pady=7, relief="flat", cursor="hand2",
                command=lambda stage=index: self._stage_gecis(stage),
            )
            button.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 3, 3 if index < len(stages) - 1 else 0))
            self.stage_buttons.append(button)

        self.stage_note_var = tk.StringVar(
            value="BACKTEST · 36 rule configurations are evaluated against the selected historical series."
        )
        tk.Label(
            stage_frame, textvariable=self.stage_note_var, bg=BG, fg=MUTED, font=("Segoe UI", 8), anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # ANIMATION CONTROLS
        animation_controls = tk.Frame(stage_frame, bg=BG)
        animation_controls.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        self.animation_button = tk.Button(
            animation_controls, text="▶ PLAY ALL · LAST 100", command=self._animasyon_toggle, state="disabled",
            bg=GREEN, fg="white", activebackground="#16a34a", activeforeground="white",
            relief="flat", padx=12, pady=5, font=("Segoe UI", 8, "bold"), cursor="hand2",
        )
        self.animation_button.pack(side="left")
        
        self.animation_reset_button = tk.Button(
            animation_controls, text="↺ RESET", command=self._animasyon_sifirla, state="disabled",
            bg=PANEL, fg=MUTED, activebackground=PANEL_ALT, relief="flat", padx=10, pady=5,
            font=("Segoe UI", 8, "bold"), cursor="hand2",
        )
        self.animation_reset_button.pack(side="left", padx=(6, 12))
        
        self.research_button = tk.Button(
            animation_controls, text="◱ RESEARCH · OFF", command=self._arastirma_degistir,
            bg=PANEL, fg=MUTED, activebackground=PANEL_ALT, activeforeground=TEXT,
            relief="flat", padx=10, pady=5, font=("Segoe UI", 8, "bold"), cursor="hand2",
        )
        self.research_button.pack(side="left", padx=(0, 12))
        
        self.animation_progress = ttk.Progressbar(
            animation_controls, orient="horizontal", mode="determinate", maximum=100, length=220,
        )
        self.animation_progress.pack(side="left", padx=(0, 10))
        
        self.animation_status_var = tk.StringVar(
            value="IDLE · all charts static; click PLAY ALL or press Ctrl+Enter"
        )
        tk.Label(
            animation_controls, textvariable=self.animation_status_var, bg=BG, fg=MUTED,
            font=("Segoe UI", 8), anchor="w",
        ).pack(side="left", fill="x", expand=True)

        # SIGNAL BADGE
        self.signal_frame = tk.Frame(self, bg=PANEL, padx=24, pady=9, highlightthickness=1, highlightbackground=GRID)
        self.signal_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=(4, 6))
        
        self.signal_value_var = tk.StringVar(value="—")
        self.signal_reason_var = tk.StringVar(value="load a symbol to evaluate the selected period")
        self.signal_test_var = tk.StringVar(value="")
        self.signal_note_var = tk.StringVar(value="")
        
        self.signal_badge = tk.Label(
            self.signal_frame, textvariable=self.signal_value_var, bg=MUTED, fg="white",
            font=("Segoe UI", 14, "bold"), padx=18, pady=3,
        )
        self.signal_badge.pack(side="left")
        
        tk.Label(
            self.signal_frame, text="SIGNAL", bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold"),
        ).pack(side="left", padx=(14, 12))
        
        signal_text = tk.Frame(self.signal_frame, bg=PANEL)
        signal_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            signal_text, textvariable=self.signal_reason_var, bg=PANEL, fg=TEXT,
            font=("Segoe UI", 9), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            signal_text, textvariable=self.signal_test_var, bg=PANEL, fg=MUTED,
            font=("Segoe UI", 7), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            self.signal_frame, textvariable=self.signal_note_var, bg=PANEL, fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right")

        # BODY & CANVAS
        body = tk.Frame(self, bg=BG)
        body.grid(row=4, column=0, sticky="nsew", padx=18, pady=(2, 4))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        self.page_canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=scrollbar.set)
        self.page_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.page = tk.Frame(self.page_canvas, bg=BG)
        self.page_window = self.page_canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", lambda _event: self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all")))
        self.page_canvas.bind("<Configure>", self._canvas_resize)
        self.page_canvas.bind_all("<MouseWheel>", self._mousewheel)

        self.figure_frame = tk.Frame(self.page, bg=PANEL, padx=7, pady=7, highlightthickness=1, highlightbackground=GRID)
        self.figure_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(3, 10))
        self.page.grid_columnconfigure(0, weight=1)

        figure_width = 16.0   # Sabit genişlik (eski: max 13.5)
        figure_height = 14.0  # Sabit yükseklik (eski: max 10.0)
        self.figure = Figure(figsize=(figure_width, figure_height), dpi=100, facecolor=PANEL)
        self._axes = {}
        self._panelleri_kur()

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

        # CANDIDATE TABLE
        self.candidate_frame = tk.Frame(self.page, bg=PANEL, padx=10, pady=8, highlightthickness=1, highlightbackground=GRID)
        self.candidate_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 10))
        tk.Label(
            self.candidate_frame, text="CANDIDATE CONFIGURATIONS  ·  SELECT A MODEL TO UPDATE BACKTEST AND RISK PANELS",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        candidate_columns = ("model", "family", "score", "return", "drawdown", "sharpe", "trades", "oos", "state")
        self.candidate_table = ttk.Treeview(
            self.candidate_frame, columns=candidate_columns, show="headings", height=8, selectmode="browse",
        )
        candidate_headings = {
            "model": "MODEL", "family": "FAMILY", "score": "SCORE", "return": "GROSS RETURN",
            "drawdown": "MAX DD", "sharpe": "SHARPE", "trades": "TRADES", "oos": "OOS RETURN", "state": "POSITION",
        }
        candidate_widths = {
            "model": 160, "family": 150, "score": 105, "return": 115, "drawdown": 105,
            "sharpe": 95, "trades": 75, "oos": 110, "state": 80,
        }
        for column in candidate_columns:
            self.candidate_table.heading(column, text=candidate_headings[column])
            self.candidate_table.column(
                column, width=candidate_widths[column], anchor="w" if column in ("model", "family") else "e",
            )
        self.candidate_table.pack(fill="x", expand=True)
        self.candidate_table.bind("<<TreeviewSelect>>", self._candidate_selected)

        # RECENT BARS TABLE
        table_frame = tk.Frame(self.page, bg=PANEL, padx=10, pady=8, highlightthickness=1, highlightbackground=GRID)
        table_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 12))
        tk.Label(table_frame, text="RECENT BARS", bg=PANEL, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        columns = ("date", "open", "high", "low", "close", "volume")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        headings = {"date": "DATE", "open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"}
        widths = {"date": 250, "open": 145, "high": 145, "low": 145, "close": 145, "volume": 180}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="w" if column == "date" else "e")
        self.table.pack(fill="x", expand=True)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self, textvariable=self.status_var, bg=BG, fg=MUTED, anchor="w", padx=24, pady=7, font=("Segoe UI", 9),
        ).grid(row=5, column=0, sticky="ew")

    def _panelleri_kur(self):
        """Rebuild the gridspec so hidden panels leave no empty holes behind."""
        self.figure.clear()
        self.figure.set_facecolor(PANEL)
        trading_rows = []
        for left, right in PANEL_SATIRLARI:
            left_visible = self._panel_gorunur[left[0]]
            right_visible = right is not None and self._panel_gorunur[right[0]]
            if left_visible or right_visible:
                trading_rows.append((left if left_visible else None, right if right_visible else None))
        research = [key for key in RESEARCH_PANELLERI if self._panel_gorunur[key]]
        full_research = len(research) == len(RESEARCH_PANELLERI)
        row_count = max(len(trading_rows) + (2 if full_research else len(research)), 1)
        height_ratios = [4.0, 3.0] if full_research else [0.85] * len(research)
        trading_heights = [
            9.0 if any(p is not None and p[0] == "fibo" for p in (left, right))
            else 6.5 if any(p is not None and p[0] == "grid" for p in (left, right))
            else 5.0
            for left, right in trading_rows
        ]
        height_ratios += trading_heights or [1.0]
        height_ratios = height_ratios[:row_count]
        grid = self.figure.add_gridspec(
            row_count, 2, width_ratios=(2.4, 1.0), height_ratios=height_ratios, hspace=0.42, wspace=0.26,
        )
        axes = {}
        index = 0
        if full_research:
            axes["mesh"] = self.figure.add_subplot(grid[0:2, 0])
            axes["loop"] = self.figure.add_subplot(grid[0, 1])
            axes["vol"] = self.figure.add_subplot(grid[1, 1])
            index = 2
        else:
            for key in research:
                axes[key] = self.figure.add_subplot(grid[index, :])
                index += 1
        for left, right in trading_rows:
            if left is not None and right is not None:
                axes[left[0]] = self.figure.add_subplot(grid[index, 0])
                axes[right[0]] = self.figure.add_subplot(grid[index, 1])
            elif left is not None:
                if left[1] == "left":
                    axes[left[0]] = self.figure.add_subplot(grid[index, 0])
                else:
                    axes[left[0]] = self.figure.add_subplot(grid[index, :])
            elif right is not None:
                axes[right[0]] = self.figure.add_subplot(grid[index, :])
            index += 1
        self._axes = axes

    def _panel_ciz(self, key, ax, df, analysis, order_rows, grid_step):
        if key == "grid": self._draw_grid(ax, df, order_rows, grid_step)
        elif key == "orders": self._draw_orders(ax, order_rows)
        elif key == "macd": self._draw_macd(ax, df)
        elif key == "plan": self._draw_plan(ax, df)
        elif key == "renk": self._draw_renk(ax, df)
        elif key == "equity": self._draw_equity(ax, df, analysis)
        elif key == "risk": self._draw_risk(ax, df, analysis)
        elif key == "mesh": self._draw_mesh(ax, analysis["candidates"])
        elif key == "loop": self._draw_loop(ax, analysis)
        elif key == "vol": self._draw_vol_surface(ax, df)
        elif key == "fibo": self._draw_fibo(ax, df)

    def _fibo_kanal_al(self):
        sembol = str(self.sembol_var.get()).strip().upper()
        periyot = str(self.periyot_var.get()).strip().lower()
        anahtar = (sembol, periyot, FIBO_MOD)
        onbellek = getattr(self, "_fibo_onbellek", None)
        if onbellek is not None and onbellek[0] == anahtar:
            _k, kn, yol, mtime, boyut, hata, hata_zamani = onbellek
            if yol is not None:
                try:
                    st = yol.stat()
                    if st.st_mtime_ns == mtime and st.st_size == boyut:
                        self._fibo_son_hata = hata
                        return kn
                except OSError:
                    pass
            else:
                import time
                if time.monotonic() - hata_zamani < 5.0:
                    self._fibo_son_hata = hata
                    return kn
        kn = None
        yol = None
        mtime = boyut = 0
        hata = ""
        hata_zamani = 0.0
        try:
            df_k, yol = veri_yukle(sembol, periyot)
            st = yol.stat()
            mtime, boyut = st.st_mtime_ns, st.st_size
            kn = kanal_hesapla(df_k, capa=PERIYOT_CAPA[periyot], mod=FIBO_MOD)
        except Exception as _hata:
            kn = None
            hata = str(_hata)
            import time
            hata_zamani = time.monotonic()
        self._fibo_son_hata = hata
        self._fibo_onbellek = (anahtar, kn, yol, mtime, boyut, hata, hata_zamani)
        return kn

    def _draw_fibo(self, ax, _df):
        self._axes_stili(ax, f"FIBONACCI KANAL · fibo_kanal [{FIBO_MOD}]")
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.25, axis="y")
        if not FIBO_KANAL_OK:
            ax.text(0.5, 0.55, "fibo_kanal.py import edilemedi", transform=ax.transAxes,
                    color=RED, ha="center", va="center", fontsize=10, fontweight="bold")
            ax.text(0.5, 0.40, FIBO_IMPORT_HATASI, transform=ax.transAxes,
                    color=MUTED, ha="center", va="center", fontsize=7.5)
            return
        kn = self._fibo_kanal_al()
        if kn is None:
            hata = (getattr(self, "_fibo_son_hata", "") or "kanal hesaplanamadı").strip()
            hata = hata.splitlines()[0] if hata else ""
            if len(hata) > 110: hata = hata[:107] + "..."
            ax.text(0.5, 0.5, hata, transform=ax.transAxes, color=RED, ha="center", va="center", fontsize=8.5)
            return
        t = kn.tablo
        x = t.index
        alt, ust = t["L0.000"], t["L1.000"]
        gecerli = alt.notna() & ust.notna() & (alt > 0) & (ust > 0)
        ax.fill_between(x[gecerli], alt[gecerli], ust[gecerli], color=GOLD, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, t["Close"], color=TEXT, lw=1.1, zorder=6, label="Kapanış")
        renkler = [cm.turbo(v) for v in np.linspace(0.03, 0.97, len(kn.fibs))]
        for i, k in enumerate(kn.fibs):
            seri = t[seviye_adi(k)]
            ok = seri.notna() & (seri > 0)
            if not ok.any(): continue
            genisletme = k > 1.0
            ax.plot(
                x[ok], seri[ok], color=renkler[i],
                lw=0.8 if genisletme else 1.1, ls=(0, (4, 2)) if genisletme else "-",
                alpha=0.9, zorder=3, label=f"{k:.3f}  {sayi_yaz(seri[ok].iloc[-1])}",
            )
        ax.axvline(kn.capa_tarih, color=RED, lw=1.0, ls="--", alpha=0.8, zorder=4)
        ax.scatter([kn.capa_tarih], [t["Close"].iloc[kn.capa_pos]], s=26, color=RED, zorder=7, edgecolor="white", linewidth=0.7)
        if kn.dip_t is not None and x[0] <= kn.dip_t <= x[-1]:
            ax.scatter([kn.dip_t], [kn.dip], marker="v", s=30, color=GREEN, zorder=8, edgecolor="white", linewidth=0.6)
        if kn.tepe_t is not None and x[0] <= kn.tepe_t <= x[-1]:
            ax.scatter([kn.tepe_t], [kn.tepe], marker="^", s=30, color=RED, zorder=8, edgecolor="white", linewidth=0.6)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        gun = (x[-1] - x[0]).days
        if gun > 1500:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=2 if gun > 5500 else 1))
            tarih_formati = "%Y"
        elif gun > 400:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            tarih_formati = "%m.%y"
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=11))
            tarih_formati = "%d.%m.%y"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(tarih_formati))
        ax.margins(x=0.012)
        uyari = ("  ·  yükselen destek yok (DataKanal Y = 0)" if kn.mod == "matriks" and kn.matriks_temas_pos is None else "")
        ax.text(0.99, 0.96, f"{kn.mod}  ·  çapa {kn.capa_tarih:%d.%m.%Y}  ·  {len(t)} bar{uyari}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        ax.legend(loc="upper left", frameon=False, labelcolor=MUTED, fontsize=7.5, ncol=4, handlelength=1.7, columnspacing=0.9, labelspacing=0.32)

    def _arastirma_degistir(self):
        show = not self._panel_gorunur["mesh"]
        for key in RESEARCH_PANELLERI:
            self._panel_gorunur[key] = show
        self._panelleri_kur()
        self.research_button.configure(
            text="◱ RESEARCH · ON" if show else "◱ RESEARCH · OFF",
            bg=PANEL_ALT if show else PANEL, fg=TEXT if show else MUTED,
        )
        if self.df is not None and self.analiz is not None:
            order_rows, grid_step = self._order_rows(self.df)
            self._grafikleri_guncelle(self.df, self.analiz, order_rows, grid_step)
        self.status_var.set("Research panels shown" if show else "Research panels hidden")

    def _canvas_resize(self, event):
        self.page_canvas.itemconfigure(self.page_window, width=event.width)

    def _mousewheel(self, event):
        if self.winfo_exists():
            self.page_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _stage_gecis(self, stage_index):
        descriptions = [
            "RESEARCH · inspect the strategy mesh and historical volatility surface.",
            "CODE · review candidate rules; select a configuration from the table below.",
            "BACKTEST · compare the selected rule, price benchmark, and historical drawdown.",
            "LIVE · offline paper grid: ATR levels and historical touches; no live order connection.",
            "POST-MORTEM · inspect the MACD line, signal and histogram against the selected rule.",
            "FINE-TUNE · select another candidate configuration below to refresh model and risk panels.",
        ]
        self.active_stage = stage_index
        for index, button in enumerate(self.stage_buttons):
            active = index == stage_index
            button.configure(bg=RED if active else PANEL, fg="white" if active else MUTED)
        self.stage_note_var.set(descriptions[stage_index])
        self.update_idletasks()
        if stage_index in (1, 5):
            target_y = self.candidate_frame.winfo_y()
        else:
            anchor_key = {0: "mesh", 2: "equity", 3: "grid", 4: "macd"}[stage_index]
            anchor = self._axes.get(anchor_key) or next(iter(self._axes.values()), None)
            figure_height = max(self.figure_canvas.get_tk_widget().winfo_height(), 1)
            target_y = self.figure_frame.winfo_y()
            if anchor is not None:
                target_y += figure_height * (1.0 - anchor.get_position().y1)
        page_height = max(self.page.winfo_height(), 1)
        viewport_height = max(self.page_canvas.winfo_height(), 1)
        max_top = max(0, page_height - viewport_height)
        top = min(max(0, target_y - viewport_height * 0.18), max_top)
        self.page_canvas.yview_moveto(top / page_height)
        self.status_var.set(f"Stage {stage_index + 1:02d}/06 · {descriptions[stage_index]}")

    def _aday_tablosu_guncelle(self, analysis):
        self._updating_candidates = True
        try:
            for row in self.candidate_table.get_children():
                self.candidate_table.delete(row)
            for index, candidate in enumerate(analysis["candidates"]):
                self.candidate_table.insert(
                    "", "end", iid=str(index),
                    values=(
                        candidate["name"], candidate["family"], sayi_yaz(candidate["score"]),
                        f"{sayi_yaz(candidate['total_return'] * 100, 1)}%",
                        f"{sayi_yaz(candidate['max_drawdown'] * 100, 1)}%",
                        sayi_yaz(candidate["sharpe"]), str(candidate["trades"]),
                        f"{sayi_yaz(candidate.get('oos_return', 0.0) * 100, 1)}%",
                        "LONG" if len(candidate["position"]) and candidate["position"][-1] > 0 else "FLAT",
                    ),
                )
        finally:
            self._updating_candidates = False

    def _animasyon_analizi(self, end_index):
        annual_bars = yillik_bar_sayisi(self.periyot_var.get())
        candidates = []
        for original in self.analiz["candidates"]:
            returns = original["returns"][: end_index + 1]
            position = original["position"][: end_index + 1]
            equity = np.cumprod(np.maximum(1.0 + returns, 0.0))
            peaks = np.maximum.accumulate(equity)
            drawdown = np.divide(equity, peaks, out=np.ones_like(equity), where=peaks != 0) - 1.0
            total_return = float(equity[-1] - 1.0) if len(equity) else 0.0
            max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
            standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            sharpe = (
                float(np.mean(returns) / standard_deviation * np.sqrt(annual_bars))
                if standard_deviation > 1e-12 else 0.0
            )
            trades = int(np.count_nonzero(np.diff(np.r_[0.0, position]) > 0))
            split = int(len(returns) * (1.0 - OOS_ORAN))
            candidate = dict(original)
            candidate.update(
                returns=returns, equity=equity, total_return=total_return, max_drawdown=max_drawdown,
                sharpe=sharpe, trades=trades,
                is_return=float(np.prod(1.0 + returns[:split]) - 1.0) if split else 0.0,
                oos_return=float(np.prod(1.0 + returns[split:]) - 1.0) if len(returns) > split else 0.0,
                score=sharpe + total_return - abs(max_drawdown),
            )
            candidates.append(candidate)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        survivors = [
            item for item in candidates
            if item["total_return"] > 0 and item["max_drawdown"] > -0.50 and item["trades"] >= 2
        ]
        leader = max(survivors or candidates, key=lambda item: item["score"])
        return {"candidates": candidates, "survivors": survivors, "leader": leader}

    def _animasyon_kisayol(self, _event=None):
        self._animasyon_toggle()
        return "break"

    def _animasyon_toggle(self):
        if self.df is None or self.analiz is None: return
        if self._animation_active:
            self._animation_active = False
            if self._animation_job is not None:
                self.after_cancel(self._animation_job)
                self._animation_job = None
            self.animation_button.configure(text="▶ RESUME")
            self.animation_status_var.set(f"PAUSED · {self._animation_frame}/{len(self._animation_indices)} bars")
            return
        if not self._animation_indices:
            start = max(0, len(self.df) - 100)
            self._animation_indices = list(range(start, len(self.df)))
            self._animation_frame = 0
        elif self._animation_frame >= len(self._animation_indices):
            self._animation_frame = 0
            self.animation_progress.configure(value=0)
        self.animation_progress.configure(maximum=max(1, len(self._animation_indices)))
        self._animation_active = True
        self.animation_button.configure(text="⏸ PAUSE")
        self.animation_reset_button.configure(state="normal")
        self.candidate_table.state(["disabled"])
        self._animasyon_adimi()

    def _animasyon_adimi(self):
        if not self._animation_active or self.df is None or self.analiz is None: return
        if self._animation_frame >= len(self._animation_indices):
            self._animation_active = False
            self._animation_job = None
            self.animation_button.configure(text="▶ REPLAY ALL · LAST 100")
            self.candidate_table.state(["!disabled"])
            self.animation_status_var.set("DONE · all charts reached the latest bar")
            return
        end_index = self._animation_indices[self._animation_frame]
        frame_analysis = self._animasyon_analizi(end_index)
        frame_number = self._animation_frame + 1
        timestamp = self.df.index[end_index]
        date_label = timestamp.strftime("%d.%m.%Y %H:%M")
        progress_label = f"AS OF {date_label} · {frame_number:03d}/{len(self._animation_indices):03d}"
        frame_df = self.df.iloc[: end_index + 1]
        order_rows, grid_step = self._order_rows(frame_df)
        self._aday_tablosu_guncelle(frame_analysis)
        self._kartlari_guncelle(frame_analysis)
        self._sinyal_guncelle(frame_df, frame_analysis)
        self._grafikleri_guncelle(frame_df, frame_analysis, order_rows, grid_step)
        self._tabloyu_guncelle(frame_df)
        progress_axis = self._axes.get("grid") or next(iter(self._axes.values()))
        progress_axis.text(
            0.98, 0.94, progress_label, transform=progress_axis.transAxes, color=TEXT, fontsize=7, ha="right", va="top",
            bbox={"facecolor": PANEL_ALT, "edgecolor": GRID, "boxstyle": "round,pad=0.25"}, zorder=5,
        )
        self.figure_canvas.draw_idle()
        self.animation_progress.configure(value=frame_number)
        self.animation_status_var.set(f"PLAYING · {frame_number}/{len(self._animation_indices)} · {date_label} · all charts")
        self._animation_frame += 1
        if self._animation_frame >= len(self._animation_indices):
            self._animation_active = False
            self._animation_job = None
            self.animation_button.configure(text="▶ REPLAY ALL · LAST 100")
            self.candidate_table.state(["!disabled"])
            self.animation_status_var.set(f"DONE · {len(self._animation_indices)} bars · latest bar {date_label} reached")
        else:
            self._animation_job = self.after(100, self._animasyon_adimi)

    def _animasyon_sifirla(self):
        self._animation_active = False
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None
        self._animation_indices = []
        self._animation_frame = 0
        self.animation_progress.configure(maximum=100, value=0)
        self.animation_button.configure(
            text="▶ PLAY ALL · LAST 100", state="normal" if self.analiz is not None else "disabled",
        )
        self.animation_reset_button.configure(state="disabled")
        self.candidate_table.state(["!disabled"])
        if self.analiz is not None and self.df is not None:
            self._aday_tablosu_guncelle(self.analiz)
            self._kartlari_guncelle(self.analiz)
            order_rows, grid_step = self._order_rows(self.df)
            self._grafikleri_guncelle(self.df, self.analiz, order_rows, grid_step)
            self._tabloyu_guncelle(self.df)
            self.animation_status_var.set("IDLE · all charts static; play to animate last 100 bars")
        else:
            self.animation_status_var.set("IDLE · load market data to animate")

    def _candidate_selected(self, _event=None):
        if self.analiz is None or self._updating_candidates: return
        selected = self.candidate_table.selection()
        if not selected: return
        candidate_index = int(selected[0])
        candidate = self.analiz["candidates"][candidate_index]
        self.analiz["leader"] = candidate
        self._kartlari_guncelle(self.analiz)
        self._sinyal_guncelle(self.df, self.analiz, test=True)
        order_rows, grid_step = self._order_rows(self.df)
        self._grafikleri_guncelle(self.df, self.analiz, order_rows, grid_step)
        self.status_var.set(f"Selected {candidate['name']} · charts and risk panels refreshed")

    @staticmethod
    def _axes_stili(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, loc="left", color=TEXT, pad=8, fontsize=10, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    @staticmethod
    def _x_etiketleri(ax, index):
        if len(index) == 0: return
        count = min(7, len(index))
        ticks = np.linspace(0, len(index) - 1, count, dtype=int)
        labels = [index[position].strftime("%d %b %y") for position in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=0, ha="center")

    def _sinyal_guncelle(self, df, analysis, test=False):
        if df is None or analysis is None or len(df) == 0: return
        signal = sinyal_hesapla(df, analysis, self.gunluk_df)
        self.signal_value_var.set(signal["verdict"])
        self.signal_badge.configure(bg=signal["color"])
        self.signal_reason_var.set(signal["reasons"])
        self.signal_note_var.set(
            f"{self.sembol_var.get()} · {self.periyot_var.get()} · ATR {sayi_yaz(signal['atr_pct'], 1)}%"
        )
        if not test: return
        stats = sinyal_geri_testi(df, analysis, self.periyot_var.get(), self.gunluk_df)
        self.signal_test_var.set(
            f"TEST · AL {stats['al']['n']} sinyal · {stats['forward']}b isabet %{sayi_yaz(stats['al']['hit'], 1)}"
            f" · ort {sayi_yaz(stats['al']['avg'], 2)}%   |   SAT {stats['sat']['n']} · ort {sayi_yaz(stats['sat']['avg'], 2)}%"
            f"   |   taban {sayi_yaz(stats['baseline'], 2)}%"
        )

    def _kartlari_guncelle(self, analysis):
        candidates = analysis["candidates"]
        survivors = analysis["survivors"]
        leader = analysis["leader"]
        kill_rate = (1.0 - len(survivors) / len(candidates)) * 100 if candidates else 0.0
        self._kart_degerleri["configs"].set(f"{len(candidates):,}".replace(",", "."))
        self._kart_degerleri["kill_rate"].set(f"{sayi_yaz(kill_rate, 1)}%")
        self._kart_degerleri["survivors"].set(str(len(survivors)))
        self._kart_degerleri["top_model"].set(leader["name"])
        self._kart_degerleri["best_return"].set(f"{sayi_yaz(leader['total_return'] * 100, 1)}%")
        self._kart_degerleri["max_dd"].set(f"{sayi_yaz(leader['max_drawdown'] * 100, 1)}%")

    @staticmethod
    def _korelasyon(vectors):
        rows, columns = vectors.shape
        correlations = np.zeros((rows, rows), dtype=float)
        if columns < 2: return correlations
        centered = vectors - vectors.mean(axis=1, keepdims=True)
        stddev = np.sqrt(np.mean(centered ** 2, axis=1))
        live = stddev > 1e-12
        if np.count_nonzero(live) > 1:
            normalized = centered[live] / stddev[live, None]
            block = (normalized @ normalized.T) / columns
            correlations[np.ix_(live, live)] = np.clip(block, -1.0, 1.0)
        np.fill_diagonal(correlations, 1.0)
        return correlations

    def _draw_mesh(self, ax, candidates, progress_label=None):
        self._axes_stili(ax, "STRATEGY MESH  ·  CANDIDATE CONFIGURATIONS")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.05, 0.95)
        ax.set_frame_on(False)
        chosen = sorted(candidates, key=lambda item: item["score"], reverse=True)[:24]
        if not chosen:
            ax.text(0, 0, "No strategy configurations", color=MUTED, ha="center")
            return
        sample_length = min(60, min(len(item["returns"]) for item in chosen))
        vectors = np.vstack([item["returns"][-sample_length:] for item in chosen])
        correlations = self._korelasyon(vectors)
        family_order = list(AILE_RENKLERI)
        centers = {"Trend": (-0.70, 0.26), "Mean reversion": (0.68, 0.25), "Breakout": (-0.34, -0.47), "Volume edge": (0.42, -0.48)}
        recent_drift = {}
        for item_index, item in enumerate(chosen):
            recent = item["returns"][-min(20, len(item["returns"])) :]
            recent_total = float(np.prod(1.0 + recent) - 1.0) if len(recent) else 0.0
            recent_drift[item_index] = float(np.clip(recent_total * 3.0, -0.24, 0.24))
        family_drift = {}
        for family in family_order:
            members = [i for i, item in enumerate(chosen) if item["family"] == family]
            family_drift[family] = float(np.mean([recent_drift[i] for i in members])) if members else 0.0
        coords = []
        for family in family_order:
            members = [i for i, item in enumerate(chosen) if item["family"] == family]
            for rank, item_index in enumerate(members):
                item = chosen[item_index]
                drift = recent_drift[item_index]
                in_market = bool(len(item["position"]) and item["position"][-1] > 0)
                angle = 2 * np.pi * rank / max(1, len(members)) + drift * 2.0 + (0.18 if in_market else 0.0)
                radius = 0.18 + 0.025 * (rank % 3) + min(abs(drift) * 0.25, 0.055)
                base_x, base_y = centers[family]
                family_shift = family_drift[family]
                cx = base_x + family_shift * 0.45
                cy = base_y + family_shift * 0.35
                x = cx + radius * np.cos(angle) + drift * 0.25
                y = cy + radius * np.sin(angle) + drift * 0.20 + (0.035 if in_market else 0.0)
                coords.append((item_index, x, y))
        coords.sort(key=lambda row: row[0])
        edge_count = 0
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if correlations[i, j] >= 0.72:
                    x1, y1 = coords[i][1], coords[i][2]
                    x2, y2 = coords[j][1], coords[j][2]
                    ax.plot([x1, x2], [y1, y2], color="#4b5563", alpha=0.3, linewidth=0.7, zorder=1)
                    edge_count += 1
        for item_index, x, y in coords:
            item = chosen[item_index]
            color = AILE_RENKLERI[item["family"]]
            size = float(np.clip(24 + abs(item["score"]) * 8 + abs(recent_drift[item_index]) * 80, 24, 130))
            ax.scatter(x, y, s=size, color=color, edgecolors=GRID, linewidths=0.5, alpha=0.88, zorder=3)
            ax.text(x, y - 0.06, item["name"], color=TEXT, ha="center", va="top", fontsize=6.5, zorder=4)
        counts = [f"{family} {sum(item['family'] == family for item in chosen)}" for family in family_order]
        ax.text(-1.20, 0.80, "   ·   ".join(counts), color=MUTED, fontsize=7.5, ha="left", va="top")
        ax.text(
            -1.20, -0.99, f"Top {len(chosen)} by score  ·  edges show return correlation ≥ 0.72  ·  {edge_count} links",
            color=MUTED, fontsize=7, ha="left",
        )
        if progress_label:
            ax.text(
                0.98, 0.94, progress_label, transform=ax.transAxes, color=TEXT, fontsize=7, ha="right", va="top",
                bbox={"facecolor": PANEL_ALT, "edgecolor": GRID, "boxstyle": "round,pad=0.25"}, zorder=5,
            )

    def _draw_loop(self, ax, analysis):
        self._axes_stili(ax, "THE LOOP  ·  CLOSED-LOOP SEARCH")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        steps = ["RESEARCH", "CODE", "BACKTEST", "CHECK", "GATE", "PAPER"]
        positions = [(0.18, 0.78), (0.50, 0.78), (0.82, 0.78), (0.82, 0.38), (0.50, 0.38), (0.18, 0.38)]
        for index in range(len(positions) - 1):
            x1, y1 = positions[index]
            x2, y2 = positions[index + 1]
            ax.plot([x1, x2], [y1, y2], color=GRID, linewidth=2, zorder=1)
        for index, ((x, y), label) in enumerate(zip(positions, steps)):
            active = index == 2
            ax.scatter(
                x, y, s=680, color=RED if active else PANEL_ALT, edgecolors=RED if active else GRID,
                linewidths=1.2, zorder=2,
            )
            ax.text(x, y, label, ha="center", va="center", color=TEXT, fontsize=7, fontweight="bold", zorder=3)
        ax.text(0.08, 0.14, f"{len(analysis['candidates'])} configs evaluated", color=TEXT, fontsize=8, ha="left")
        ax.text(0.08, 0.07, f"{len(analysis['survivors'])} passed the historical gate", color=MUTED, fontsize=7.5, ha="left")

    def _draw_equity(self, ax, df, analysis):
        self._axes_stili(ax, "EQUITY CURVE  ·  NET OF 0.1% COST  ·  OOS SHADED")
        leader = analysis["leader"]
        equity = pd.Series(leader["equity"], index=df.index)
        price = df["Close"].astype(float)
        if len(equity) < 2:
            ax.text(0.5, 0.5, "Not enough bars for an equity curve", transform=ax.transAxes,
                    color=MUTED, ha="center", va="center")
            return
        start_equity = float(equity.iloc[0])
        start_price = float(price.iloc[0])
        equity_index = equity / (start_equity if start_equity else 1.0) * 100.0
        benchmark = price / (start_price if start_price else 1.0) * 100.0
        x = np.arange(len(equity_index))
        split = int(len(equity_index) * (1.0 - OOS_ORAN))
        ax.plot(x, equity_index.to_numpy(), color=GREEN, linewidth=1.8, label=f"{leader['name']} · net")
        ax.plot(x, benchmark.to_numpy(), color=BLUE, linewidth=1.1, alpha=0.75, label="Price benchmark")
        ax.axhline(100, color=MUTED, linestyle="--", linewidth=0.7, alpha=0.7)
        running_peak = np.maximum.accumulate(equity_index.to_numpy())
        ax.fill_between(x, equity_index.to_numpy(), running_peak, color=RED, alpha=0.25, label="Drawdown")
        if 0 < split < len(equity_index):
            ax.axvspan(split, len(equity_index), color=BLUE, alpha=0.15, label="OOS · son %30")
            ax.axvline(split, color=MUTED, linestyle=":", linewidth=0.9)
            ax.text(split, ax.get_ylim()[0], " IS | OOS", color=MUTED, fontsize=7, va="bottom", ha="left")
        self._x_etiketleri(ax, equity_index.index)
        ax.set_ylabel("Index (start = 100)")
        ax.legend(loc="upper left", frameon=False, labelcolor=MUTED, fontsize=7.5, ncol=3)

    def _draw_vol_surface(self, ax, df):
        self._axes_stili(ax, "VOL SURFACE  ·  LOOKBACK × HORIZON")
        returns = df["Close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        lookbacks = [10, 20, 40, 80]
        horizons = [5, 10, 20, 40]
        surface = np.zeros((len(horizons), len(lookbacks)))
        for row, horizon in enumerate(horizons):
            for column, lookback in enumerate(lookbacks):
                sample = returns.tail(lookback)
                sigma = float(sample.std()) if len(sample) > 1 else 0.0
                surface[row, column] = sigma * np.sqrt(horizon) * 100.0
        ax.imshow(surface, cmap="Spectral_r", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(lookbacks)), [str(value) for value in lookbacks])
        ax.set_yticks(range(len(horizons)), [str(value) for value in horizons])
        ax.set_ylabel("Horizon bars")
        for row in range(len(horizons)):
            for column in range(len(lookbacks)):
                # Dark theme için yazı rengi açık yapıldı
                ax.text(column, row, f"{surface[row, column]:.1f}%", ha="center", va="center", color="#f8f9fa", fontsize=7)

    @staticmethod
    def _true_range(df):
        previous_close = df["Close"].shift(1)
        parts = pd.concat(
            [df["High"] - df["Low"], (df["High"] - previous_close).abs(), (df["Low"] - previous_close).abs()],
            axis=1,
        )
        return parts.max(axis=1)

    def _draw_grid(self, ax, df, order_rows, step):
        self._axes_stili(ax, "CANDLESTICK GRID ENGINE  ·  LAST 150 HISTORICAL BARS")
        visible = df.tail(150)
        x = np.arange(len(visible))
        opens = visible["Open"].astype(float).to_numpy()
        highs = visible["High"].astype(float).to_numpy()
        lows = visible["Low"].astype(float).to_numpy()
        closes = visible["Close"].astype(float).to_numpy()
        colors = [GREEN if close >= open_ else RED for open_, close in zip(opens, closes)]
        for position, (open_, high, low, close, color) in enumerate(zip(opens, highs, lows, closes, colors)):
            ax.vlines(position, low, high, color=color, linewidth=0.8, zorder=2)
            body = max(abs(close - open_), max(abs(close), abs(open_), 1.0) * 0.00015)
            ax.add_patch(
                Rectangle((position - 0.30, min(open_, close)), 0.60, body, facecolor=color, edgecolor=color, linewidth=0.4, zorder=3)
            )
        all_closes = df["Close"].astype(float)
        ema20 = all_closes.ewm(span=20, adjust=False).mean().tail(len(visible)).to_numpy()
        ema50 = all_closes.ewm(span=50, adjust=False).mean().tail(len(visible)).to_numpy()
        ax.plot(x, ema20, color="#fbbf24", linewidth=1.35, label="EMA 20", zorder=4)
        ax.plot(x, ema50, color="#38bdf8", linewidth=1.45, label="EMA 50", zorder=4)
        current = float(df["Close"].iloc[-1])
        ax.axhline(current, color=TEXT, linewidth=1.0, linestyle="--", alpha=0.8)
        for name, price, side, _touches in order_rows:
            color = GOLD if side == "TP" else (GREEN if price < current else RED)
            ax.axhline(price, color=color, linewidth=0.65, alpha=0.55, linestyle=":" if side != "TP" else "--")
            ax.text(len(visible) - 1, price, f" {name}", color=color, fontsize=6.5, va="bottom")
        self._x_etiketleri(ax, visible.index)
        ax.set_xlim(-1, len(visible) + 2)
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", frameon=True, framealpha=0.2, facecolor=PANEL_ALT, edgecolor=GRID, fontsize=7, labelcolor=TEXT)
        ax.text(
            0.99, 0.98, f"LAST {sayi_yaz(current)}  ·  GRID STEP {sayi_yaz(step)}",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7,
        )

    def _order_rows(self, df):
        current = float(df["Close"].iloc[-1])
        atr = float(self._true_range(df).tail(14).mean())
        if not np.isfinite(atr) or atr <= 0:
            atr = max(current * 0.01, 0.01)
        step = max(atr / 2.0, current * 0.001)
        recent = df.tail(300)
        rows = []
        take_profit = current + 2.0 * step
        tp_touches = int(((recent["Low"] <= take_profit) & (recent["High"] >= take_profit)).sum())
        rows.append(("TP", take_profit, "TP", tp_touches))
        for level in range(1, 7):
            price = current - level * step
            touches = int(((recent["Low"] <= price) & (recent["High"] >= price)).sum())
            rows.append((f"L{level}", price, "BUY", touches))
        return rows, step

    def _draw_orders(self, ax, rows):
        self._axes_stili(ax, "ORDER GRID  ·  LAST 300 BARS")
        ax.axis("off")
        cells = [[name, sayi_yaz(price), str(touches), side] for name, price, side, touches in rows]
        table = ax.table(
            cellText=cells, colLabels=["LEVEL", "PRICE", "TOUCHES", "SIDE"],
            cellLoc="center", colLoc="center", bbox=[0.02, 0.06, 0.96, 0.82],
            colWidths=[0.19, 0.31, 0.25, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_facecolor(PANEL_ALT if row == 0 else PANEL)
            cell.get_text().set_color(MUTED if row == 0 else TEXT)
            if row == 0:
                cell.get_text().set_fontweight("bold")
        ax.text(0.03, 0.01, "Levels are calculated from ATR; no orders are sent.", transform=ax.transAxes, color=MUTED, fontsize=6.5)

    @staticmethod
    def _macd_serileri(df, hizli=12, yavas=26, sinyal=9):
        return macd_serileri(df, hizli, yavas, sinyal)

    def _draw_macd(self, ax, df, bar_sayisi=150):
        self._axes_stili(ax, "MACD  ·  12 / 26 / 9")
        macd, signal, histogram = self._macd_serileri(df)
        if len(macd) < 2:
            ax.text(0.5, 0.5, "Not enough bars for MACD", transform=ax.transAxes,
                    color=MUTED, ha="center", va="center")
            return
        macd = macd.tail(bar_sayisi)
        signal = signal.tail(bar_sayisi)
        histogram = histogram.tail(bar_sayisi)
        x = np.arange(len(macd))
        colors = [GREEN if value >= 0 else RED for value in histogram]
        ax.bar(x, histogram.to_numpy(), width=0.62, color=colors, alpha=0.65, linewidth=0, zorder=2)
        ax.plot(x, macd.to_numpy(), color=GOLD, linewidth=1.5, label="MACD 12/26", zorder=3)
        ax.plot(x, signal.to_numpy(), color=BLUE, linewidth=1.3, label="Signal 9", zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.8, zorder=1)
        self._x_etiketleri(ax, macd.index)
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left", frameon=False, labelcolor=MUTED, fontsize=7.5, ncol=2)
        ax.text(
            0.99, 0.96, f"MACD {sayi_yaz(float(macd.iloc[-1]), 4)}  ·  SIGNAL {sayi_yaz(float(signal.iloc[-1]), 4)}",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7,
        )

    def _renk_seviyeleri(self):
        if Indrenk is None or self.df is None: return None
        key = (id(self.df), len(self.df))
        if self._renk_key == key: return self._renk_verisi
        try:
            veri = Indrenk().hesapla(self.df)
        except Exception:
            return None
        self._renk_key = key
        self._renk_verisi = veri
        return veri

    def _draw_renk(self, ax, df, bar_sayisi=150):
        self._axes_stili(ax, "RENKLER  ·  S2 · TOP/BOTTOM BOX · FIBONACCI")
        veri = self._renk_seviyeleri()
        if veri is None:
            ax.text(0.5, 0.5, "indrenk.py bulunamadı", transform=ax.transAxes,
                    color=MUTED, ha="center", va="center")
            return
        data = veri.reindex(df.index).tail(bar_sayisi)
        if len(data) < 2:
            ax.text(0.5, 0.5, "Not enough bars", transform=ax.transAxes,
                    color=MUTED, ha="center", va="center")
            return
        x = np.arange(len(data))
        close = data["Close"].to_numpy(dtype=float)
        top_series = data["TopBox"].to_numpy(dtype=float)
        bottom_series = data["BottomBox"].to_numpy(dtype=float)
        ax.plot(x, close, color=MUTED, linewidth=0.8, alpha=0.65, label="Close")
        ax.plot(x, data["S2"].to_numpy(dtype=float), color=GOLD, linewidth=1.7, label="S2")
        ax.plot(x, top_series, color=RED, linewidth=1.4, label="TopBox")
        ax.plot(x, bottom_series, color=GREEN, linewidth=1.4, label="BottomBox")
        fark = top_series - bottom_series
        with np.errstate(invalid="ignore"):
            ax.fill_between(x, bottom_series, top_series, color=BLUE, alpha=0.1, zorder=0)
            for ratio, color, style, name in (
                (0.236, MUTED, ":", None), (0.382, MUTED, ":", None), (0.5, GOLD, ":", "Fib retrace"),
                (0.618, MUTED, ":", None), (0.786, MUTED, ":", None), (1.272, BLUE, "--", None),
                (1.618, BLUE, "--", "Fib ext"),
            ):
                ax.plot(x, bottom_series + fark * ratio, color=color, linestyle=style,
                        linewidth=0.85, alpha=0.9, label=name, zorder=1)
        low_y = float(np.nanmin(close))
        high_y = float(np.nanmax(close))
        finite_top = top_series[np.isfinite(top_series)]
        finite_bottom = bottom_series[np.isfinite(bottom_series)]
        if len(finite_top) and len(finite_bottom):
            last_top = float(finite_top[-1])
            last_bottom = float(finite_bottom[-1])
            if last_top > last_bottom:
                low_y = min(low_y, last_bottom)
                high_y = max(high_y, last_top, last_bottom + (last_top - last_bottom) * 1.618)
        pad = (high_y - low_y) * 0.06 if high_y > low_y else max(high_y * 0.02, 0.01)
        ax.set_ylim(low_y - pad, high_y + pad)
        self._x_etiketleri(ax, data.index)
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", frameon=False, labelcolor=MUTED, fontsize=7.5, ncol=4)

    def _draw_plan(self, ax, df):
        self._axes_stili(ax, "ENTRY PLAN  ·  SETUP & TRIGGER")
        ax.axis("off")
        plan = giris_plani(df, self.gunluk_df)
        trigger = plan["trigger"]
        if trigger:
            stop_pct = (plan["stop"] - trigger) / trigger * 100.0
            target_pct = (plan["target"] - trigger) / trigger * 100.0
            stop_text = f"{sayi_yaz(plan['stop'])}  ·  {sayi_yaz(stop_pct, 1)}%"
            target_text = f"{sayi_yaz(plan['target'])}  ·  +{sayi_yaz(target_pct, 1)}%"
        else:
            stop_text = target_text = "—"
        quality = plan["quality"]
        quality_color = GREEN if quality >= 60 else (GOLD if quality >= 40 else RED)
        rows = [
            ("SETUP", plan["setup"], TEXT), ("QUALITY", f"{quality}/100", quality_color),
            ("TRIGGER", plan["trigger_text"], TEXT), ("STOP", stop_text, TEXT),
            ("TARGET · 2R", target_text, TEXT),
            ("SIZE · 1.000 TL", f"{plan['shares']} adet" if plan["shares"] else "—", TEXT),
            ("TREND · EMA50/200", plan["trend"], TEXT), ("STATUS", plan["status"], plan["status_color"]),
        ]
        y = 0.90
        for label, value, color in rows:
            ax.text(0.03, y, label, transform=ax.transAxes, color=MUTED, fontsize=6.8, ha="left", va="center")
            ax.text(0.97, y, value, transform=ax.transAxes, color=color, fontsize=7.3, ha="right", va="center", fontweight="bold")
            ax.plot([0.03, 0.97], [y - 0.05, y - 0.05], transform=ax.transAxes, color=GRID, linewidth=0.6)
            y -= 0.112
        ax.text(
            0.03, 0.015, "1R = trigger − stop; daily bars; 1.000 TL risk per trade.",
            transform=ax.transAxes, color=MUTED, fontsize=6.3,
        )

    def _draw_risk(self, ax, df, analysis):
        self._axes_stili(ax, "MODEL SNAPSHOT  ·  NET OF COST")
        ax.axis("off")
        leader = analysis["leader"]
        macd, signal, histogram = self._macd_serileri(df)
        if len(macd):
            son_macd = float(macd.iloc[-1])
            son_hist = float(histogram.iloc[-1])
        else:
            son_macd = son_hist = float("nan")
        if not np.isfinite(son_hist):
            durum = "—"
        else:
            durum = "BULLISH" if son_hist >= 0 else "BEARISH"
        profit_factor = leader.get("profit_factor", 0.0)
        profit_text = "∞" if not np.isfinite(profit_factor) else sayi_yaz(profit_factor)
        rows = [
            ("TOP RULE", leader["name"]),
            ("IS / OOS RETURN", f"{sayi_yaz(leader.get('is_return', 0.0) * 100, 1)} / {sayi_yaz(leader.get('oos_return', 0.0) * 100, 1)}%"),
            ("MAX DRAWDOWN", f"{sayi_yaz(leader['max_drawdown'] * 100, 1)}%"),
            ("TRADES · WIN RATE", f"{leader['trades']} · {sayi_yaz(leader.get('win_rate', 0.0), 1)}%"),
            ("PROFIT FACTOR", profit_text), ("EXPOSURE", f"{sayi_yaz(leader.get('exposure', 0.0), 1)}%"),
            ("AVG WIN / LOSS", f"{sayi_yaz(leader.get('avg_win', 0.0), 2)} / {sayi_yaz(leader.get('avg_loss', 0.0), 2)}%"),
            ("MACD 12/26/9", f"{sayi_yaz(son_macd, 3)} · {durum}"),
        ]
        y = 0.92
        for label, value in rows:
            ax.text(0.03, y, label, transform=ax.transAxes, color=MUTED, fontsize=6.6, ha="left", va="center")
            ax.text(0.97, y, value, transform=ax.transAxes, color=TEXT, fontsize=7.1, ha="right", va="center", fontweight="bold")
            ax.plot([0.03, 0.97], [y - 0.048, y - 0.048], transform=ax.transAxes, color=GRID, linewidth=0.6)
            y -= 0.108
        ax.text(
            0.03, 0.012, "Net of 0.1% per side; IS = first 70%, OOS = last 30%.",
            transform=ax.transAxes, color=MUTED, fontsize=6.2,
        )

    def _grafikleri_guncelle(self, df, analysis, order_rows, grid_step):
        for ax in self._axes.values():
            ax.clear()
        for key, ax in self._axes.items():
            self._panel_ciz(key, ax, df, analysis, order_rows, grid_step)
        self.figure.subplots_adjust(left=0.065, right=0.985, top=0.955, bottom=0.045)
        self.figure_canvas.draw_idle()

    def _tabloyu_guncelle(self, df):
        for row in self.table.get_children():
            self.table.delete(row)
        for timestamp, row in df.tail(5).iloc[::-1].iterrows():
            self.table.insert(
                "", "end",
                values=(
                    timestamp.strftime("%d.%m.%Y %H:%M"), sayi_yaz(row["Open"]), sayi_yaz(row["High"]),
                    sayi_yaz(row["Low"]), sayi_yaz(row["Close"]), sayi_yaz(row["Volume"], 0),
                ),
            )

    def veri_yukle(self):
        symbol = self.sembol_var.get().strip().upper()
        period = self.periyot_var.get().strip().lower()
        if symbol not in self.semboller:
            messagebox.showwarning("Symbol not found", "Choose a symbol from the list or enter a valid code.", parent=self)
            return
        if period not in PERIYOTLAR:
            messagebox.showwarning("Choose a period", "Select 60dk, 240dk, gunluk, haftalik or aylik.", parent=self)
            return
        self._animasyon_sifirla()
        self.status_var.set(f"Loading {symbol} / {period}…")
        self.load_button.configure(state="disabled")
        self.update_idletasks()
        try:
            df = self.veri.get(symbol, period)
            if df is None or df.empty:
                raise ValueError(f"No {period} data found for {symbol}.")
            df = df.copy()
            for column in ("Open", "High", "Low", "Close", "Volume"):
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) < 2:
                raise ValueError("At least two valid bars are required for the research panels.")
            try:
                gunluk = self.veri.get(symbol, "gunluk")
            except Exception:
                gunluk = None
            self.gunluk_df = gunluk if gunluk is not None and len(gunluk) else None
            candidates, survivors, leader = strateji_adaylarini_hesapla(df, period)
            analysis = {"candidates": candidates, "survivors": survivors, "leader": leader}
            order_rows, grid_step = self._order_rows(df)
            self.df = df
            self.analiz = analysis
            self._aday_tablosu_guncelle(analysis)
            self._kartlari_guncelle(analysis)
            self._sinyal_guncelle(df, analysis, test=True)
            self._grafikleri_guncelle(df, analysis, order_rows, grid_step)
            self._tabloyu_guncelle(df)
            self.animation_button.configure(state="normal", text="▶ PLAY ALL · LAST 100")
            self.animation_reset_button.configure(state="disabled")
            self.animation_progress.configure(value=0)
            self.animation_status_var.set("IDLE · all charts static; play to animate last 100 bars")
            self.status_var.set(
                f"{symbol} · {period} · {df.index.min()} — {df.index.max()}  |  "
                f"{len(df):,} bars  |  Historical simulation from local CSV"
            )
        except Exception as error:
            messagebox.showerror("Data load failed", str(error), parent=self)
            self.status_var.set("Data load failed")
        finally:
            self.load_button.configure(state="normal")

if __name__ == "__main__":
    PiyasaPaneli().mainloop()