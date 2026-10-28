# 🤖 Self-Improving Trading Machine · BIST

> **Yerel OHLCV araştırma paneli** — Kapalı döngü strateji üretimi, backtest ve paper simulation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Tkinter](https://img.shields.io/badge/UI-Tkinter-green)](https://docs.python.org/3/library/tkinter.html)
[![Matplotlib](https://img.shields.io/badge/Charts-Matplotlib-orange)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Genel Bakış

Bu proje, **BIST hisseleri** için tasarlanmış tamamen yerel çalışan bir trading araştırma panelidir. Sistem şu döngüyü otomatik olarak uygular:

```
RESEARCH → CODE → BACKTEST → SCORE → GATE → PAPER DEPLOY
```

Yüzlerce strateji adayı üretilir, geçmiş veri üzerinde test edilir, başarısız olanlar elenir ve **hayatta kalan en iyi strateji** kullanıcıya önerilir. Hiçbir veri buluta gönderilmez — tüm CSV'ler yerel diskinizde kalır.

---

## ✨ Özellikler

### 🎯 Strateji Üretimi
- **4 strateji ailesi**: Trend (MA), Mean Reversion (Z-score), Breakout, Volume Edge
- **36+ otomatik konfigürasyon** (farklı periyot ve eşik değerleri)
- **Kill-rate sistemi**: başarısız stratejileri otomatik eleme
- **IS / OOS ayrımı**: %70 eğitim / %30 test verisi

### 📊 Görselleştirme
- 🕯️ **Candlestick Grid Engine** (EMA20/EMA50 ile)
- 📐 **Fibonacci Kanal** entegrasyonu (`fibo_kanal.py`)
- 🎨 **Renk Seviyeleri** (S2, TopBox, BottomBox) via `indrenk.py`
- 📈 **MACD** (12/26/9) histogram ve sinyal
- 💰 **Equity Curve** (drawdown ve OOS gölgelemesi ile)
- 🌐 **Strategy Mesh** (korelasyon ağı görselleştirmesi)
- 🔄 **Closed-Loop** diyagramı
- 📉 **Volatilite Yüzeyi** (lookback × horizon)

### 🧠 Sinyal ve Entry Planı
- 4 oylamalı sinyal sistemi: Trend + Momentum + MACD + Model
- Otomatik **trigger / stop / target / position size** hesabı
- ATR tabanlı **order grid** (L1–L6 seviyeleri)
- **Kalite skoru** (0–100) ile setup değerlendirmesi

### 🎬 Animasyonlu Backtest
- Son 100 barı **adım adım** oynatma
- Tüm paneller senkron güncellenir
- `Ctrl+Enter` kısayolu ile başlat/durdur

### 🎨 Modern Arayüz
- Dark theme (göz yormayan antrasit palet)
- Responsive pencere boyutlandırma
- Scrollable canvas ile tüm panellere erişim

---

## 🚀 Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install numpy pandas matplotlib
```

> Tkinter Python ile birlikte gelir, ayrıca kurulmasına gerek yoktur.

### 2. Proje Yapısı

```
SMOM/
├── panel.py                 # Ana panel (bu repo)
├── veri_periyot.py          # CSV veri yükleyici
├── fibo_kanal.py            # Fibonacci kanal hesaplayıcı
├── indrenk.py               # Renk seviyeleri modülü
├── panel_tvdata_ekleri.py   # TVDataFeed entegrasyonu (opsiyonel)
└── GLM/                     # Opsiyonel GLM dizini
```

### 3. CSV Veri Klasörleri

`veri_periyot.py` dosyasında tanımlı klasörlere CSV dosyalarınızı yerleştirin:

```
📁 60dk/   → THYAO.csv, GARAN.csv, ...
📁 gunluk/ → THYAO.csv, GARAN.csv, ...
```

CSV formatı standart OHLCV olmalıdır:
```
Date,Open,High,Low,Close,Volume
2024-01-02,100.5,102.3,99.8,101.2,1500000
```

---

## 🎮 Kullanım

### Paneli Başlatma

```bash
python panel.py
```

### Arayüz Kontrolleri

| Kontrol | Açıklama |
|---------|----------|
| **SYMBOL** | Hisse kodu (THYAO, GARAN, vb.) |
| **PERIOD** | Zaman dilimi (60dk, 240dk, gunluk, haftalik, aylik) |
| **LOAD DATA** | Veriyi yükle ve analizi başlat |
| **◱ RESEARCH** | Mesh/Loop/Vol panellerini aç/kapat |
| **▶ PLAY ALL** | Son 100 barı animasyonlu oynat |
| **↺ RESET** | Animasyonu sıfırla |
| **Ctrl+Enter** | Animasyon kısayolu |

### Stage (Aşama) Geçişi

Üstteki 6 buton ile farklı görünümlere geçiş yapın:
1. **RESEARCH** — Strateji mesh ve volatilite yüzeyi
2. **CODE** — Aday konfigürasyonlar tablosu
3. **BACKTEST** — Equity eğrisi ve drawdown
4. **LIVE** — Paper grid ve ATR seviyeleri
5. **POST-MORTEM** — MACD analizi
6. **FINE-TUNE** — Başka bir aday seç

---

## 🧮 Hesaplama Detayları

| Metrik | Açıklama |
|--------|----------|
| **İşlem Maliyeti** | %0.1 her yön (komisyon + slippage) |
| **OOS Oranı** | %30 (son %30 test verisi) |
| **Sinyal Eşiği** | AL ≥ +2, SAT ≤ -2 (4 oylu sistem) |
| **Risk/Trade** | 1.000 TL sabit risk |
| **Position Size** | `1000 / (trigger − stop)` adet |

---

## 📂 Modüller

### `fibo_kanal.py` Entegrasyonu
Panel, `fibo_kanal.py`'nin **kendi pipeline'ını** kullanır:
- CLI ile birebir aynı sonuç
- 10 Fibonacci seviyesi (0.0 – 1.618+)
- Matriks / kanal / aralık modları
- CSV değişikliklerinde otomatik yeniden hesaplama

### `indrenk.py` Entegrasyonu
- S2 seviyesi
- TopBox / BottomBox
- Fibonacci retracement (0.236 – 0.786) ve extension (1.272, 1.618)

### `panel_tvdata_ekleri.py` (Opsiyonel)
- TVDataFeed ile BIST ve kripto verisi çekme
- Yerel CSV'ye kaydetme

---

## ⚙️ Teknik Detaylar

- **Dil**: Python 3.9+
- **UI**: Tkinter + ttk (Clam theme)
- **Grafik**: Matplotlib (TkAgg backend)
- **Veri İşleme**: NumPy + Pandas
- **Önbellekleme**: Fibo kanal ve renk seviyeleri mtime/boyut bazlı cache

---

## 🗺️ Yol Haritası

- [ ] Web arayüzü (Streamlit / Dash)
- [ ] Daha fazla strateji ailesi (ML tabanlı)
- [ ] Portföy seviyesi backtest
- [ ] Walk-forward optimizasyonu
- [ ] Canlı veri feed entegrasyonu

---

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için önce bir issue açarak tartışalım.

---

## 📜 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır — detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## ⚠️ Yasal Uyarı

> Bu proje **yalnızca araştırma ve eğitim amaçlıdır**. Yatırım tavsiyesi değildir. Geçmiş performans gelecek sonuçların garantisi değildir. Tüm trading kararlarınız kendi sorumluluğunuzdadır.

---

<div align="center">

**⭐ Bu projeyi faydalı bulduysanız yıldız vermeyi unutmayın!**

Made with ❤️ for BIST traders

</div>
