# -*- coding: utf-8 -*-
"""
fibo_kanal.py — Sabit çapalı log-Fibonacci kanalı (tek hisse, tek periyot)
==========================================================================

MetaStock/Matriks formülünün Python karşılığı:

    bb := C;
    A  := Log(bb);
    bar := Cum(1) - 1;

    HedefTarih := Year() = 2023 AND Month() = 4 AND Day() = 11;

    GecenBarSayisi     := BarsSince(HedefTarih);
    SabitBaslangicBari := bar - GecenBarSayisi;

    a0 := EXP(DataKanal(A, SabitBaslangicBari, 0, Y, 0.000, D));
    ...
    a9 := EXP(DataKanal(A, SabitBaslangicBari, 0, Y, 1.618, D));

Periyoda göre çapa (formüldeki sabit başlangıç barı):

    günlük / haftalık / aylık : 23.12.2002  (verinin ilk barı)
    60dk / 240dk              : 11.04.2023

Kanal tanımı (mod="kanal", Python API varsayılanı)
---------------------------------------------------
[çapa .. son bar] penceresinde log kapanışa en küçük kareler doğrusu oturtulur:

    fit(t) = a + b·t          r(t) = ln C(t) − fit(t)          W = max(r) − min(r)
    L_k(t) = exp( fit(t) + (k − 0.5)·W )

Çizgiler log grafikte PARALEL ve EĞİMLİDİR (XLOG ekranındaki gibi).
k=0 alt kenar, k=0.5 regresyon (orta), k=1 üst kenar, k>1 genişleme.

Matriks otomatik trend (mod="matriks", CLI varsayılanı)
---------------------------------------------------------
Y=yükselen ve D=dip için alt çizgi, penceredeki en düşük log kapanış m'den
geçen ve tüm kapanışların altında kalan en dik yükselen destektir:

    b = min((ln C(t) - ln C(m)) / (t - m))  (t > m)
    A(t) = ln C(m) + b·(t - m)
    W = max(ln C(t) - A(t))
    L_k(t) = exp(A(t) + k·W)

DataKanal(..., Bitiş Noktası=0, ...) için hesaplama penceresi son mevcut barda
biter. Çizgi bu barda kapanışa sabitlenmez; L_k son değeri doğru üzerindeki
değerdir. `t` değerleri m ile son bar arasındadır; son temas barı eğimi belirler,
çizgi son bara kadar devam eder. Yükselen destek yoksa sonuç sıfırdır.

Alternatif mod="aralik" (yatay merdiven)
----------------------------------------
Çapadan son bara kadar en düşük/en yüksek kapanış; seviyeler tüm pencere
boyunca YATAY çizgi olarak çizilir:

    dip = min C(t) ,  tepe = max C(t)
    L_k(t) = exp( ln(dip) + k·(ln(tepe) − ln(dip)) )

Trendin neredeyse sıfır olduğu hisselerde bu görünüm kanalınkine çok yaklaşır
(YATAS 240dk örneği). Dip ve tepe tarihleri konsolda yazar, grafikte işaretlenir.

Alternatif mod="aralik_asof"
----------------------------
Aynı aralık, ama her bar için yalnızca o güne kadarki dip–tepe kullanılır
(merdiven görünümü; ileriye bakmaz, backtest için doğru olan).

Kullanım
--------
Dosyanın en altındaki HISSE / PERIYOT değerlerini değiştirip çalıştır:

    python fibo_kanal.py
    python fibo_kanal.py THYAO haftalik
    python fibo_kanal.py YUNSA 60dk --son 15 --goster
    python fibo_kanal.py YUNSA gunluk --mod aralik

Kod içinden:

    from fibo_kanal import calistir
    kn = calistir("YUNSA", "haftalik")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if "--goster" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# --------------------------------------------------------------- sabitler ----
FIBS: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.618)

PERIYOT_ETIKET = {
    "gunluk": "Günlük",
    "haftalik": "Haftalık",
    "aylik": "Aylık",
    "240dk": "240 dk",
    "60dk": "60 dk",
}

# periyot -> varsayılan çapa tarihi (formüldeki SabitBaslangicBari)
PERIYOT_CAPA = {
    "gunluk": "2002-12-23",
    "haftalik": "2002-12-23",
    "aylik": "2002-12-23",
    "240dk": "2023-04-11",
    "60dk": "2023-04-11",
}

KLASOR_60DK = Path(r"D:\new\yeniden\metastock-60dk-csv")
KLASOR_GUNLUK = Path(r"D:\new\yeniden\metastock-gun-csv")


def seviye_adi(k: float) -> str:
    """0.5 -> 'L0.500' gibi kolon adı."""
    return f"L{k:.3f}"


def _zaman_str(t: pd.Timestamp) -> str:
    """Günlük barlarda 'dd.mm.yyyy', intraday barlarda 'dd.mm.yyyy HH:MM'."""
    if t.hour or t.minute:
        return f"{t:%d.%m.%Y %H:%M}"
    return f"{t:%d.%m.%Y}"


# -------------------------------------------------------------------- veri ----
def _kolon_haritasi(df: pd.DataFrame) -> dict[str, str]:
    """Kolon adlarını (BOM / büyük-küçük harf farkıyla) standart adlara eşler."""
    istenen = ("date", "time", "open", "high", "low", "close", "volume")
    harita: dict[str, str] = {}
    for kol in df.columns:
        temiz = str(kol).strip().lstrip("\ufeff").lower()
        if temiz in istenen and temiz not in harita:
            harita[temiz] = kol
    return harita


def _saat_duzelt(deger: str) -> str:
    """'10:00', '9:30', '1000' gibi saatleri 'HH:MM' biçimine getirir."""
    s = str(deger).strip().replace(".", ":")
    if ":" in s:
        parca = s.split(":")
        return f"{int(parca[0]):02d}:{int(parca[1]):02d}" if len(parca) > 1 else f"{int(parca[0]):02d}:00"
    s = s.zfill(4)
    return f"{s[:2]}:{s[2:]}"


def _csv_oku(yol: Path) -> pd.DataFrame:
    """CSV okur: Date (+ varsa Time), Open, High, Low, Close, Volume."""
    raw = pd.read_csv(yol)
    h = _kolon_haritasi(raw)
    if "date" not in h or "close" not in h:
        raise ValueError(f"{yol.name}: 'Date' ve 'Close' kolonları bulunamadı")

    # kolon adları geçici olarak küçük harfe indirilir
    df = raw.rename(columns={orijinal: kucuk for kucuk, orijinal in h.items()})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "time" in df.columns:
        saat = df["time"].map(_saat_duzelt)
        zaman = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + saat, errors="coerce")
        df = df.drop(columns=["time"])
    else:
        zaman = df["date"]

    df.index = pd.DatetimeIndex(zaman, name="DATETIME")
    df = df.drop(columns=["date"])

    for kol in ("open", "high", "low", "close", "volume"):
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    # bozuk bar: High < Low ise yer değiştir
    bozuk = df["High"] < df["Low"]
    if bozuk.any():
        df.loc[bozuk, ["High", "Low"]] = df.loc[bozuk, ["Low", "High"]].to_numpy()

    return df[["Open", "High", "Low", "Close", "Volume"]]


def _resample(df: pd.DataFrame, kural: str) -> pd.DataFrame:
    return (
        df.resample(kural)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


def _240dk_yap(df: pd.DataFrame) -> pd.DataFrame:
    """60dk barları günün iki bloğuna indirger: 10:00–13:59 ve 14:00–18:00."""
    g = df.copy()
    g["_gun"] = g.index.normalize()
    g["_blok"] = (g.index.hour >= 14).astype(int)

    out = (
        g.groupby(["_gun", "_blok"], sort=True)
        .agg(Open=("Open", "first"), High=("High", "max"),
             Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"))
        .reset_index()
    )
    saat = np.where(out["_blok"].to_numpy() == 0, 10, 14)
    out.index = pd.DatetimeIndex(out["_gun"] + pd.to_timedelta(saat, unit="h"), name="DATETIME")
    return out.drop(columns=["_gun", "_blok"]).sort_index()


def veri_yukle(
    hisse: str,
    periyot: str,
    csv_yolu: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Hisse + periyot için OHLCV verisini döndürür: (df, kaynak_dosya).

    Kaynak sırası: verilen csv_yolu -> Matriks klasörü -> çalışma dizinindeki <hisse>.csv
    """
    hisse = hisse.upper()
    periyot = periyot.lower()
    if periyot not in PERIYOT_CAPA:
        raise ValueError(f"Geçersiz periyot: {periyot} (seçenekler: {', '.join(PERIYOT_CAPA)})")

    if csv_yolu is not None:
        yol = Path(csv_yolu)
        if not yol.exists():
            raise FileNotFoundError(f"Veri dosyası yok: {yol}")
    else:
        klasor = KLASOR_60DK if periyot in ("60dk", "240dk") else KLASOR_GUNLUK
        yol = klasor / f"{hisse}.csv"
        if not yol.exists():
            yerel = Path(f"{hisse.lower()}.csv")
            if yerel.exists():
                yol = yerel
            else:
                raise FileNotFoundError(
                    f"{hisse}.csv bulunamadı. Aranan: {klasor} ve {Path.cwd()}"
                )

    df = _csv_oku(yol)

    if periyot == "haftalik":
        df = _resample(df, "W-FRI")
    elif periyot == "aylik":
        df = _resample(df, "ME")
    elif periyot == "240dk":
        df = _240dk_yap(df)

    if len(df) < 5:
        raise ValueError(f"{hisse} {periyot}: yetersiz veri ({len(df)} bar)")
    return df, yol


# ------------------------------------------------------------------- kanal ----
def capa_bul(index: pd.DatetimeIndex, tarih: str | pd.Timestamp) -> tuple[int, pd.Timestamp, bool]:
    """Çapa tarihinin bar indeksini bulur.

    Tam o gün yoksa ondan sonraki ilk bara oturur; tarih verinin başından
    önceyse (hisse daha sonra işlem görmeye başlamışsa) ilk bara düşer.
    Dönen: (pozisyon, gerçek_tarih, kaydirildi_mi)
    """
    hedef = pd.Timestamp(tarih).normalize()
    if hedef <= index[0].normalize():
        return 0, index[0], hedef != index[0].normalize()
    pos = int(index.searchsorted(hedef))
    if pos >= len(index):
        raise ValueError(
            f"Çapa tarihi verinin sonundan sonra: {hedef.date()} > {index[-1].date()}"
        )
    return pos, index[pos], index[pos].normalize() != hedef


@dataclass
class Kanal:
    """Hesaplanmış kanal + teşhis bilgileri."""

    tablo: pd.DataFrame                 # index=tarih, kolonlar: Close, L0.000 ... L1.618
    capa_pos: int
    capa_tarih: pd.Timestamp
    istenen_tarih: pd.Timestamp
    kaydirildi: bool
    mod: str
    fibs: tuple[float, ...]
    matriks_temas_pos: int | None = None
    sigma: float = 0.0
    egim_yil: float = 0.0
    r2: float = 0.0
    genislik: float = 0.0
    dip: float = float("nan")
    tepe: float = float("nan")
    dip_t: pd.Timestamp | None = None
    tepe_t: pd.Timestamp | None = None
    notlar: list[str] = field(default_factory=list)

    @property
    def seviyeler(self) -> list[str]:
        return [seviye_adi(k) for k in self.fibs]

    @property
    def bars_since(self) -> int:
        return len(self.tablo) - 1 - self.capa_pos


def kanal_hesapla(
    df: pd.DataFrame,
    capa: str | pd.Timestamp,
    fibs: tuple[float, ...] = FIBS,
    mod: str = "kanal",
    a_ref: float | None = None,
    w_ref: float | None = None,
) -> Kanal:
    """Formüldeki kanalı hesaplar.

    mod="kanal"       : log-OLS doğrusu ± W/2 (W = artıkların min-max aralığı)
                        — eğimli, paralel çizgiler (XLOG ekranındaki gibi)
    mod="matriks"     : en düşük kapanıştan geçen en dik yükselen destek; son
                        temas barı eğimi belirler, çizgi son mevcut bara uzanır.
                        Bitiş Noktası=0 bu son barı ifade eder. Yükselen destek
                        yoksa sonuç sıfırdır.
    mod="aralik"      : çapadan bugüne dip–tepe aralığı, yatay sabit çizgiler
    mod="aralik_asof" : aynı aralık ama her barda o güne kadarki dip–tepe
                        (merdiven görünümü, ileriye bakmayan sürüm)

    a_ref / w_ref verilirse (XLOG ekranındaki k=0 değeri ve kanal genişliği)
    seviyeler doğrudan bunlara kalibre edilir: son barda k=0 tam a_ref olur,
    doğru çapa kapanışından geçecek şekilde eğim alınır.
    """
    if mod not in ("aralik", "aralik_asof", "kanal", "matriks"):
        raise ValueError("mod 'matriks', 'kanal', 'aralik' veya 'aralik_asof' olmalı")
    if len(df) < 5:
        raise ValueError("Kanal için en az 5 bar gerekli")

    index = df.index
    pos, gercek, kaydirildi = capa_bul(index, capa)

    ln = np.log(df["Close"].to_numpy(dtype=float))
    n = len(ln)
    k_arr = np.asarray(fibs, dtype=float)
    seviye = np.full((len(fibs), n), np.nan)

    sigma = r2 = egim_yil = genislik = 0.0
    dip = tepe = float("nan")
    dip_t = tepe_t = None
    matriks_temas_pos = None

    if a_ref is not None:
        # XLOG ekranına kalibrasyon: alt çizgi min kapanıştan geçer, son barda a_ref'e ulaşır.
        # W verilmezse üst çizgi en yüksek kapanışa değecek şekilde (max sapma) hesaplanır.
        pencere = ln[pos:]
        m = int(np.argmin(pencere))
        gun = max((index[-1] - index[pos + m]).days, 1)
        b = (np.log(a_ref) - pencere[m]) / gun
        ofs = np.asarray((index[pos:] - index[-1]).days, dtype=float)
        A = pencere[m] + b * (ofs - ofs[m])
        if w_ref is None:
            w_ref = float((pencere - A).max())
        for i, k in enumerate(k_arr):
            seviye[i, pos:] = np.exp(A + k * w_ref)
        genislik = float(w_ref)
        egim_yil = b * 365.25
    elif mod == "matriks":
        # DataKanal(..., Bitiş Noktası=0, Y, k, D): pencere son mevcut barda biter.
        # Destek eğimi, minimum kapanıştan sonraki en küçük log-fiyat/bar oranıdır.
        y = ln[pos:]
        m0 = int(np.argmin(y))
        idx = np.arange(len(y), dtype=float)
        if m0 < len(y) - 1:
            oranlar = (y[m0 + 1:] - y[m0]) / (idx[m0 + 1:] - m0)
            temas_rel = m0 + 1 + int(np.argmin(oranlar))
            b = float(oranlar[temas_rel - m0 - 1])
            if b > 0:
                A = y[m0] + b * (idx - m0)
                genislik = float((y - A).max())
                for i, k in enumerate(k_arr):
                    seviye[i, pos:] = np.exp(A + k * genislik)
                matriks_temas_pos = pos + temas_rel
                yil = (index[-1] - index[pos]).days / 365.25
                bar_yil = (len(y) - 1) / yil if yil > 0 else 252.0
                egim_yil = b * bar_yil
            else:
                # Y yönü yükselen trend ister; sıfır/yatay destek sinyali üretmez.
                seviye[:, pos:] = 0.0
        else:
            # En düşük kapanış bitiş barındaysa sonrasına göre yükselen destek kurulamaz.
            seviye[:, pos:] = 0.0
    elif mod == "kanal":
        y = ln[pos:]
        m = len(y)
        x = np.arange(m, dtype=float)
        b, a = np.polyfit(x, y, 1)
        fit = a + b * x
        resid = y - fit
        ddof = 2 if m > 2 else 0
        sigma = float(resid.std(ddof=ddof)) if m > 2 else 0.0
        genislik = float(resid.max() - resid.min())
        for i, k in enumerate(k_arr):
            seviye[i, pos:] = np.exp(fit + (k - 0.5) * genislik)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
        yil = (index[-1] - index[pos]).days / 365.25
        bar_yil = (m - 1) / yil if yil > 0 else 252.0
        egim_yil = b * bar_yil
    else:
        pencere = ln[pos:]
        i_dip = int(np.argmin(pencere))
        i_tepe = int(np.argmax(pencere))
        dip, tepe = float(np.exp(pencere[i_dip])), float(np.exp(pencere[i_tepe]))
        dip_t, tepe_t = index[pos + i_dip], index[pos + i_tepe]

        if mod == "aralik":            # sabit çizgiler (platform görünümü)
            lo, hi = float(pencere.min()), float(pencere.max())
            genislik = hi - lo
            for i, k in enumerate(k_arr):
                seviye[i, pos:] = np.exp(lo + k * genislik)
        else:                          # her bar için o güne kadarki aralık
            alt = np.minimum.accumulate(pencere)
            ust = np.maximum.accumulate(pencere)
            for i, k in enumerate(k_arr):
                seviye[i, pos:] = np.exp(alt + k * (ust - alt))
            genislik = float(np.log(tepe) - np.log(dip))

    tablo = pd.DataFrame({seviye_adi(k): seviye[i] for i, k in enumerate(k_arr)}, index=index)
    tablo.insert(0, "Close", df["Close"].astype(float))

    kn = Kanal(
        tablo=tablo, capa_pos=pos, capa_tarih=gercek,
        istenen_tarih=pd.Timestamp(capa).normalize(), kaydirildi=kaydirildi,
        mod=mod, fibs=tuple(fibs), sigma=sigma, egim_yil=egim_yil,
        r2=r2, genislik=genislik, dip=dip, tepe=tepe, dip_t=dip_t, tepe_t=tepe_t,
        matriks_temas_pos=matriks_temas_pos,
    )
    if kaydirildi:
        if kn.istenen_tarih <= index[0].normalize():
            kn.notlar.append(
                f"Veri {index[0]:%d.%m.%Y} başlıyor; istenen {kn.istenen_tarih:%d.%m.%Y} "
                f"öncesi olduğu için ilk bar kullanıldı"
            )
        else:
            kn.notlar.append(
                f"İstenen {kn.istenen_tarih:%d.%m.%Y} işlem günü değil; "
                f"sonraki bara oturuldu ({gercek:%d.%m.%Y})"
            )
    return kn


def konum_k(kn: Kanal) -> pd.Series:
    """Kapanışın kanal içindeki Fibonacci karşılığı (0 = alt, 1 = üst)."""
    t = kn.tablo
    alt = t[seviye_adi(0.0)]
    ust = t[seviye_adi(1.0)]
    fark = (np.log(ust) - np.log(alt)).replace(0, np.nan)
    return (np.log(t["Close"]) - np.log(alt)) / fark


def son_bar_tablosu(kn: Kanal, adet: int = 10, ondalik: int = 4) -> pd.DataFrame:
    """Son `adet` barın kapanış + 10 seviye + k* tablosunu döndürür."""
    t = kn.tablo.iloc[-adet:].copy()
    t.insert(0, "bar", np.arange(len(kn.tablo) - len(t), len(kn.tablo)))
    t["k*"] = konum_k(kn).iloc[-adet:]
    return t[["bar", "Close", "k*"] + kn.seviyeler].round(ondalik)


# ------------------------------------------------------------------ konsol ----
def _baslik(metin: str) -> None:
    print()
    print(metin)
    print("-" * min(118, max(60, len(metin))))


def konsola_yaz(kn: Kanal, hisse: str, periyot: str, kaynak: Path, adet: int = 10) -> None:
    t = kn.tablo
    n = len(t)
    print()
    print("=" * 112)
    print(f"  {hisse.upper()}  |  {PERIYOT_ETIKET[periyot]:<9}  |  "
          f"sabit çapalı log-Fibonacci kanalı ({kn.mod})")
    print("=" * 112)
    print(f"  Veri      : {kaynak}")
    print(f"              {n} bar   {_zaman_str(t.index[0])} -> {_zaman_str(t.index[-1])}")
    print(f"  Çapa      : {_zaman_str(kn.capa_tarih)}  (mutlak bar no {kn.capa_pos})"
          f"   |   bugün bar {n - 1}   |   BarsSince = {kn.bars_since}")
    print(f"  Kural     : {periyot} varsayılan çapası {PERIYOT_CAPA[periyot]}"
          f"   |   SabitBaslangicBari = {kn.capa_pos} (sabit)")
    for not_ in kn.notlar:
        print(f"  Not       : {not_}")

    _baslik("  KANAL TANIMI")
    if kn.mod == "kanal":
        print("  mod       : log-OLS kanalı  (log kapanış üzerine en küçük kareler doğrusu)")
        print(f"  eğim      : {(np.exp(kn.egim_yil) - 1) * 100:+.1f}% / yıl   "
              f"(log eğim {kn.egim_yil:+.4f})")
        print("  kenarlar  : fit(t) - W/2 ... fit(t) + W/2   (W = artık min-max aralığı)")
        print(f"  genişlik  : W = {kn.genislik:.4f} log birim  "
              f"(alt/üst oranı {np.exp(kn.genislik):.2f}x)")
        print(f"  sigma     : {kn.sigma:.4f}   |   R2: {kn.r2:.4f}")
        print(f"  seviye_k  : EXP( fit(t) + (k - 0.5)*{kn.genislik:.4f} )")
        if kn.r2 < 0.10:
            print("  dikkat    : R2 çok düşük — bu pencerede belirgin trend yok, kanal yatay")
    elif kn.mod == "matriks":
        print("  mod       : Matriks DataKanal Y=yükselen, D=dip (destek zarfı)")
        print(f"  eğim      : {(np.exp(kn.egim_yil) - 1) * 100:+.1f}% / yıl   "
              f"(log eğim {kn.egim_yil:+.4f})")
        print("  alt çizgi : min kapanıştan geçen, bütün kapanışların altında kalan en dik doğru")
        if kn.matriks_temas_pos is None:
            print("  destek    : yükselen temas yok; DataKanal Y sonucu 0")
        else:
            temas_tarihi = t.index[kn.matriks_temas_pos]
            print(f"  temas     : bar {kn.matriks_temas_pos}  {_zaman_str(temas_tarihi)} "
                  "(eğimi belirleyen ikinci destek teması)")
        print(f"  bitiş     : Bitiş Noktası=0, son mevcut bar {n - 1}  "
              f"{_zaman_str(t.index[-1])}; L0={t['L0.000'].iloc[-1]:.4f}, "
              f"kapanış={t['Close'].iloc[-1]:.4f}")
        print(f"  genişlik  : W = max(ln(C) - A(t)) = {kn.genislik:.4f} log birim  "
              f"(alt/üst oranı {np.exp(kn.genislik):.2f}x)")
        print(f"  seviye_k  : EXP(A(t) + k*{kn.genislik:.4f})")
    else:
        if kn.mod == "aralik":
            print("  mod       : çapadan bugüne dip-tepe aralığı, seviyeler SABİT")
            print("              (Matriks'teki yatay çizgiler; aralık tüm pencere için bir kez)")
        else:
            print("  mod       : çapadan bugüne dip-tepe aralığı, her bar için o güne kadar")
            print("              (merdiven görünümü, ileriye bakmaz)")
        print(f"  dip       : {kn.dip:.4f}   {_zaman_str(kn.dip_t)}")
        print(f"  tepe      : {kn.tepe:.4f}   {_zaman_str(kn.tepe_t)}")
        print(f"  genişlik  : {kn.genislik:.4f} log birim  "
              f"(tepe/dip oranı {kn.tepe / kn.dip:.2f}x)")
        print("  seviye_k  : EXP( ln(dip) + k*(ln(tepe) - ln(dip)) )   (log uzayında)")

    son = t.iloc[-1]
    kstar = konum_k(kn).iloc[-1]
    _baslik("  SON BAR")
    print(f"  Tarih {_zaman_str(t.index[-1])}   Kapanış {son['Close']:.4f}   "
          f"kanal içi konum k* = {kstar:.3f}")
    for k in kn.fibs:
        deger = son[seviye_adi(k)]
        fark = (deger / son["Close"] - 1.0) * 100.0
        print(f"    k={k:<5.3f}  {deger:10.4f}   ({fark:+7.2f}% kapanışa göre)")

    _baslik(f"  SON {adet} BAR")
    tablo = son_bar_tablosu(kn, adet)
    kolonlar = kn.seviyeler
    bas = f"{'bar':>5} {'tarih':<17} {'Close':>8} {'k*':>6}"
    print(bas + "".join(f"{c:>9}" for c in kolonlar))
    print("-" * (len(bas) + 9 * len(kolonlar)))
    for ts, satir in tablo.iterrows():
        cizgi = (f"{int(satir['bar']):5d} {_zaman_str(ts):<17} "
                 f"{satir['Close']:8.4f} {satir['k*']:6.3f}")
        cizgi += "".join(f"{satir[c]:9.4f}" for c in kolonlar)
        print(cizgi)

    en_yakin = min(kn.fibs, key=lambda k: abs(k - kstar))
    _baslik("  ÖZET")
    print(f"  Kapanış kanal içinde k* = {kstar:.3f} -> en yakın seviye k={en_yakin:g} "
          f"({son[seviye_adi(en_yakin)]:.4f})")
    print(f"  Son {adet} barda k* aralığı: {tablo['k*'].min():.3f} ... {tablo['k*'].max():.3f}")
    print()


# ------------------------------------------------------------------- grafik ----
def _renkler(adet: int) -> list:
    cmap = plt.get_cmap("turbo")
    return [cmap(x) for x in np.linspace(0.03, 0.97, adet)]


def safe_png_path(yol: str | Path | None) -> Path | None:
    """Windows için güvenli çıktı yolu üretir; geçersiz karakterleri sterilize eder."""
    if yol is None:
        return None
    p = Path(str(yol)).expanduser()
    if not str(p.name):
        return None

    name = p.name
    for ch in '<>:"/\\|?*\x00':
        name = name.replace(ch, "_")
    name = name.strip().strip(".")
    if not name:
        name = "fibo_kanal.png"
    if not name.lower().endswith(".png"):
        name += ".png"

    base = p.parent if p.parent != Path("") else Path.cwd()
    safe = base / name
    safe.parent.mkdir(parents=True, exist_ok=True)
    return safe


def grafik_ciz(
    kn: Kanal,
    hisse: str,
    periyot: str,
    png: str | Path | None = None,
) -> None:
    t = kn.tablo
    renk = _renkler(len(kn.fibs))
    kstar = konum_k(kn).iloc[-1]
    intraday = periyot in ("60dk", "240dk")
    hisse = hisse.upper()

    fig = plt.figure(figsize=(16.2, 10.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 2.45], hspace=0.22,
        left=0.06, right=0.845, top=0.93, bottom=0.07,
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # --- üst panel: tüm geçmiş ------------------------------------------
    ax0.plot(t.index, t["Close"], color="#1f1f1f", lw=1.0)
    ax0.axvline(kn.capa_tarih, color="#c0392b", lw=1.2, ls="--", alpha=0.85)
    ax0.axvspan(kn.capa_tarih, t.index[-1], color="#f2c14e", alpha=0.12, lw=0)
    ax0.set_yscale("log")
    ax0.set_title(
        f"{hisse} · {PERIYOT_ETIKET[periyot]} · tüm geçmiş (log)  —  "
        f"çapa {_zaman_str(kn.capa_tarih)} · bar {kn.capa_pos}",
        loc="left", fontsize=12, fontweight="bold",
    )
    ax0.set_ylabel("Fiyat (log)")
    ax0.grid(alpha=0.22, which="major")
    ax0.grid(alpha=0.10, which="minor")
    ax0.annotate(f"çapa  {_zaman_str(kn.capa_tarih)}",
                 xy=(0.022, 0.94), xycoords="axes fraction", va="top",
                 fontsize=8, color="#8e2b20")

    # --- alt panel: çapa sonrası ----------------------------------------
    bas = max(0, kn.capa_pos - 40)
    d = t.iloc[bas:]
    x = d.index

    alt, ust = d["L0.000"], d["L1.000"]
    kanal_gecerli = alt.notna() & ust.notna() & (alt > 0) & (ust > 0)
    ax1.fill_between(x[kanal_gecerli], alt[kanal_gecerli], ust[kanal_gecerli],
                     color="#f2c14e", alpha=0.10, lw=0, zorder=1)
    ax1.plot(x, d["Close"], color="#111111", lw=1.5, zorder=6, label="Kapanış")

    for i, k in enumerate(kn.fibs):
        kol = seviye_adi(k)
        gecerliler = d[kol].notna() & (d[kol] > 0)
        genisletme = k > 1.0
        ax1.plot(
            x[gecerliler], d[kol][gecerliler],
            color=renk[i],
            lw=0.85 if genisletme else 1.25,
            ls=(0, (4, 2)) if genisletme else "-",
            alpha=0.9,
            label=f"{k:.3f}  →  {d[kol].iloc[-1]:.4f}",
            zorder=3,
        )

    ax1.axvline(kn.capa_tarih, color="#c0392b", lw=1.2, ls="--", alpha=0.85, zorder=4)
    ax1.scatter([kn.capa_tarih], [t["Close"].iloc[kn.capa_pos]], s=36,
                color="#c0392b", zorder=7, edgecolor="white", linewidth=0.9)

    # dip / tepe işaretleri (aralık modlarında)
    if kn.dip_t is not None and kn.dip_t >= d.index[0]:
        ax1.scatter([kn.dip_t], [kn.dip], marker="v", s=46, color="#1b7f4b",
                    zorder=8, edgecolor="white", linewidth=0.7)
    if kn.tepe_t is not None and kn.tepe_t >= d.index[0]:
        ax1.scatter([kn.tepe_t], [kn.tepe], marker="^", s=46, color="#b3261e",
                    zorder=8, edgecolor="white", linewidth=0.7)

    ax1.set_yscale("log")
    kalibre = any("KALİBRE" in n for n in kn.notlar)
    mod_adi = {
        "kanal": "log-OLS kanalı (W = artık aralığı)",
        "matriks": "Matriks otomatik trend (destek zarfı)",
        "aralik": "log aralık (dip → tepe, sabit)",
        "aralik_asof": "log aralık (dip → tepe, o güne kadar)",
    }[kn.mod]
    if kalibre:
        mod_adi = "XLOG değerlerine kalibre (log eksen)"
    ax1.set_title(
        f"Kanal · çapadan bugüne ({kn.bars_since + 1} bar)   —   {mod_adi}",
        loc="left", fontsize=12, fontweight="bold", pad=27,
    )
    ax1.set_ylabel("Fiyat (log)")
    ax1.grid(alpha=0.22, which="major")
    ax1.grid(alpha=0.10, which="minor")

    if kalibre:
        bilgi = (f"eğim {(np.exp(kn.egim_yil) - 1) * 100:+.1f}%/yıl   |   "
                 f"W = {kn.genislik:.4f} log ({np.exp(kn.genislik):.2f}x)   |   "
                 f"son kapanış {t['Close'].iloc[-1]:.4f} (k*={kstar:.3f})   |   "
                 f"XLOG ekran değerlerine kalibre")
    elif kn.mod in ("kanal", "matriks"):
        bilgi = (f"eğim {(np.exp(kn.egim_yil) - 1) * 100:+.1f}%/yıl   |   "
                 f"W = {kn.genislik:.4f} log ({np.exp(kn.genislik):.2f}x)   |   "
                 f"σ={kn.sigma:.4f}   |   R²={kn.r2:.3f}   |   "
                 f"son kapanış {t['Close'].iloc[-1]:.4f} (k*={kstar:.3f})")
    else:
        bilgi = (f"dip {kn.dip:.4f} ({_zaman_str(kn.dip_t)})   →   "
                 f"tepe {kn.tepe:.4f} ({_zaman_str(kn.tepe_t)})   |   "
                 f"son kapanış {t['Close'].iloc[-1]:.4f}  (k*={kstar:.3f})")
    ax1.text(0.0, 1.006, bilgi, transform=ax1.transAxes, va="bottom",
             fontsize=8.3, color="#444444", family="monospace")

    ax1.legend(
        loc="upper left", bbox_to_anchor=(1.012, 1.0), frameon=False,
        fontsize=8.5, title="Fibonacci (son değer)", title_fontsize=8.5,
        labelspacing=0.62, borderaxespad=0.0, handlelength=2.4,
    )

    for ax in (ax0, ax1):
        ax.margins(x=0.012)
        x0, x1 = ax.get_xlim()
        gun = x1 - x0
        if intraday and gun <= 45:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=11))
            fmt = "%d.%m %H:%M"
        elif gun > 1500:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=2 if gun > 5500 else 1))
            fmt = "%Y"
        elif gun > 400:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fmt = "%m.%y"
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=11))
            fmt = "%d.%m.%y"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        for yan in ("top", "right"):
            ax.spines[yan].set_visible(False)

    fig.text(0.012, 0.012,
             "seviyeler log uzayında: 0 = dip/alt kenar, 1 = tepe/üst kenar, >1 genişleme   ·   "
             "formül: EXP(DataKanal(Log(C), SabitBaslangicBari, 0, Y, k, D))",
             fontsize=7.6, color="#555555")

    if png:
        hedef = safe_png_path(png)
        if hedef is not None:
            fig.savefig(str(hedef), dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Grafik kaydedildi: {hedef.resolve()}")

    if "--goster" in sys.argv:
        plt.show()
    plt.close(fig)


# -------------------------------------------------------------------- ana ----
def calistir(
    hisse: str,
    periyot: str,
    tarih: str | None = None,
    mod: str = "kanal",
    adet: int = 10,
    csv_yolu: str | Path | None = None,
    png: str | Path | None = "",
    a_ref: float | None = None,
    w_ref: float | None = None,
) -> Kanal:
    """Tek hisse + tek periyot: kanalı hesaplar, konsola yazar, grafiği çizer.

    tarih verilmezse periyodun varsayılan çapası kullanılır
    (günlük/haftalık/aylık 2002-12-23, 60dk/240dk 2023-04-11).
    a_ref/w_ref verilirse kanal XLOG ekranındaki değerlere kalibre edilir.
    png="" ise grafik <HISSE>_<periyot>_fibo.png adıyla kaydedilir.
    """
    hisse = hisse.upper()
    periyot = periyot.lower()
    capa = tarih or PERIYOT_CAPA[periyot]

    df, kaynak = veri_yukle(hisse, periyot, csv_yolu=csv_yolu)
    kn = kanal_hesapla(df, capa=capa, mod=mod, a_ref=a_ref, w_ref=w_ref)
    if a_ref is not None and w_ref is not None:
        kn.notlar.append(
            f"KALİBRE: k=0 son bar {a_ref:.4f}, genişlik {w_ref:.4f} "
            f"(ekrandaki değerlere sabitlendi)"
        )

    konsola_yaz(kn, hisse, periyot, kaynak, adet=adet)

    png_yolu = f"{hisse}_{periyot}_fibo.png" if png == "" else png
    grafik_ciz(kn, hisse, periyot, png=safe_png_path(png_yolu))
    return kn


def _cli(hisse: str, periyot: str) -> tuple[str, str, dict]:
    """İsteğe bağlı komut satırı: hisse, periyot, --tarih, --mod, --son, --csv."""
    import argparse

    ap = argparse.ArgumentParser(description="Sabit çapalı log-Fibonacci kanalı")
    ap.add_argument("hisse", nargs="?", default=hisse)
    ap.add_argument("periyot", nargs="?", default=periyot, choices=list(PERIYOT_CAPA))
    ap.add_argument("--tarih", default=None, help="Çapa tarihi (YYYY-MM-DD)")
    ap.add_argument("--mod", default="matriks",
                    choices=["matriks", "kanal", "aralik", "aralik_asof"],
                    help="matriks: otomatik trend | kanal: log-OLS | aralik: yatay dip-tepe")
    ap.add_argument("--son", type=int, default=10, help="Konsolda gösterilecek bar sayısı")
    ap.add_argument("--a", dest="a_ref", type=float, default=None,
                    help="Kalibrasyon: XLOG ekranındaki k=0 (son bar) değeri")
    ap.add_argument("--w", dest="w_ref", type=float, default=None,
                    help="Kalibrasyon: kanal genişliği (log birim)")
    ap.add_argument("--csv", default=None, help="Özel CSV yolu")
    ap.add_argument("--png", default="", help="PNG adı ('yok' = kaydetme)")
    ap.add_argument("--goster", action="store_true", help="Grafik penceresini aç")
    a = ap.parse_args()

    png = None if str(a.png).lower() == "yok" else (a.png or "")
    return a.hisse, a.periyot, dict(
        tarih=a.tarih, mod=a.mod, adet=a.son, csv_yolu=a.csv, png=png,
        a_ref=a.a_ref, w_ref=a.w_ref,
    )


if __name__ == "__main__":
    # ================== BURAYI DEĞİŞTİR ==================
    HISSE = "paseu"        # hisse kodu (CSV dosya adı)
    PERIYOT = "gunluk"     # gunluk | haftalik | aylik | 240dk | 60dk
    # =====================================================

    _h, _p, _secenek = _cli(HISSE, PERIYOT)
    calistir(_h, _p, **_secenek)

