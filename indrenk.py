# indrenk.py
import pandas as pd
import numpy as np

class Indrenk:
    """
    MetaStock Formülleri (Sadece S2 + Box)
    - S2: Trailing Destek/Direnç (yuzde, per1)
    - Box: Konsolidasyon Kutusu (TopBox/BottomBox, boxp)
    """
    def __init__(self, yuzde=2.0, per1=1, boxp=17):
        self.yuzde = yuzde
        self.per1  = per1
        self.boxp  = boxp

    def _hesapla_s2(self, close):
        """
        S2: Trailing Destek/Direnç (MetaStock mantığının numpy-optimize çevirisi).
        - L1: trailing destek  (EMA'nın yuzde/100 altı, sadece yukarı yürür)
        - L2: trailing direnç  (EMA'nın yuzde/100 üstü, sadece aşağı yürür)
        - k1: EMA, önceki L2'yi yukarı keser  → Cross(EMA, Ref(L2,-1))
        - k2: EMA, önceki L1'in altına düşer  → Cross(Ref(L1,-1), EMA)
        - S2 = son kesimden beri hangisi daha eskiyse o (d1<d2 → L1, aksi L2)
        NOT: girilen seride NaN varsa karşılaştırmalar False döner (sessiz donma);
        çağıran taraf veriyi dropna etmiş olmalı (veri_periyot zaten yapıyor).
        """
        close = pd.Series(close).astype(float)
        n = len(close)
        if n == 0:
            return pd.Series(dtype=float, name='S2')
        if n == 1:
            return pd.Series([close.iloc[0] * (1 + self.yuzde / 100)],
                             index=close.index, name='S2')

        p    = self.yuzde / 100.0
        ema  = close.ewm(span=self.per1, adjust=False).mean().to_numpy()
        ortp = ema * (1 - p)          # numpy — döngüde iloc yok
        ortm = ema * (1 + p)

        l1, l2, s2 = np.zeros(n), np.zeros(n), np.zeros(n)
        l1[0], l2[0], s2[0] = ortp[0], ortm[0], ortm[0]

        k1_bar, k2_bar = -1, -1

        for i in range(1, n):
            pl1, pl2 = l1[i-1], l2[i-1]

            # --- L1: trailing destek (asla aşağı inmez) ---
            if ema[i] < pl1:      l1[i] = ortp[i]
            elif ortp[i] > pl1:   l1[i] = ortp[i]
            else:                 l1[i] = pl1

            # --- L2: trailing direnç (asla yukarı çıkmaz) ---
            if ema[i] > pl2:      l2[i] = ortm[i]
            elif ortm[i] < pl2:   l2[i] = ortm[i]
            else:                 l2[i] = pl2

            # --- MetaStock cross mantığı (Ref zinciri doğru çevrilmiş) ---
            k1 = i >= 2 and ema[i] > pl2 and ema[i-1] <= l2[i-2]
            k2 = i >= 2 and pl1 > ema[i] and l1[i-2] <= ema[i-1]

            if k1: k1_bar = i
            if k2: k2_bar = i

            d1 = i - k1_bar if k1_bar != -1 else -1
            d2 = i - k2_bar if k2_bar != -1 else -1

            if d1 == -1 and d2 == -1: s2[i] = l2[i]
            elif d1 == -1:            s2[i] = l2[i]
            elif d2 == -1:            s2[i] = l1[i]
            elif d1 < d2:             s2[i] = l1[i]
            else:                     s2[i] = l2[i]

        return pd.Series(s2, index=close.index, name='S2')

    def _hesapla_box(self, df):
        df = df.copy()
        df['LL'] = df['Low'].rolling(window=self.boxp, min_periods=1).min()
        df['k1'] = df['High'].rolling(window=self.boxp, min_periods=1).max()
        df['k2'] = df['High'].rolling(window=self.boxp-1, min_periods=1).max()
        df['k3'] = df['High'].rolling(window=self.boxp-2, min_periods=1).max()
        
        breakout = df['High'] > df['k1'].shift(1)
        df['NH'] = pd.Series(np.where(breakout, df['High'], np.nan), index=df.index).ffill()
        df['box1'] = df['k3'] < df['k2']
        
        df['grp'] = breakout.cumsum()
        df['bars_since'] = df.groupby('grp').cumcount()
        onay = (df['bars_since'] == self.boxp-2) & df['box1']
        
        top    = pd.Series(np.where(onay, df['NH'], np.nan), index=df.index).ffill()
        bottom = pd.Series(np.where(onay, df['LL'], np.nan), index=df.index).ffill()
        
        return top, bottom

    def hesapla(self, df):
        """Sadece S2 ve Box seviyelerini döndürür"""
        res = df[['Open', 'High', 'Low', 'Close']].copy()
        res['S2'] = self._hesapla_s2(res['Close'])
        res['TopBox'], res['BottomBox'] = self._hesapla_box(res)
        return res[['Close', 'S2', 'TopBox', 'BottomBox']]

# 🧪 HIZLI TEST BLOĞU
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from veri_periyot import VeriPeriyot
    except ImportError:
        print("❌ 'veri_periyot.py' aynı dizinde bulunamadı.")
        sys.exit(1)

    vp = VeriPeriyot(
        csv_60dk_dizini=r'D:\new\yeniden\metastock-60dk-csv',
        csv_gunluk_dizini=r'D:\new\yeniden\metastock-gun-csv'
    )
    
    df_raw = vp.get('tuprs', '60dk')
    if df_raw is None:
        print("❌ Veri çekilemedi")
    else:
        ind = Indrenk(yuzde=2.0, per1=1, boxp=17)
        sonuc = ind.hesapla(df_raw)
        
        print("📊 INDRENK TEST ÇIKTISI (Son 10 Bar)")
        print("="*80)
        print(sonuc.tail(10).round(3))
        print("="*80)
        
        