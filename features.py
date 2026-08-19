import logging
import numpy as np
import pandas as pd
from typing import Optional

from config import (
    FEATURES_CONT,
    FEATURES_CAT,
    FEATURES_FUND,
    HORIZONS_MONTHS,
    TARGET_COL,
    STATE_LABELS,
    STATE_LABELS_FUND,
)

logger = logging.getLogger("FII.features")

def get_quantile_edges(series: pd.Series, q: int = 4) -> np.ndarray:
    """
    Obtém os limites (edges) de quantis para discretizar uma série contínua.
    """
    try:
        jitter = series + np.random.normal(0, 1e-8, size=len(series))
        edges = pd.qcut(jitter, q=q, duplicates="drop", retbins=True)[1]
        if len(edges) < q + 1:
            edges = np.histogram_bin_edges(series, bins=q)
        return edges
    except:
        return np.histogram_bin_edges(series, bins=q)

def classify_regime(selic_val: float, selic_change_6m: float, ipca12_val: float) -> str:
    """
    Classifica o regime macroeconômico atual com base na Selic e IPCA.
    """
    ipca_high = ipca12_val > 6.0
    ipca_low = ipca12_val < 4.5
    selic_rising = selic_change_6m > 0.5
    if ipca_high and selic_rising:
        return "estagflacao"
    elif ipca_low and not selic_rising:
        return "crescimento"
    else:
        return "neutro"

def classify_market(ret_3m: float, vol: float, vol_median: float) -> str:
    """
    Classifica o momento de mercado (bull, bear, neutro) do IFIX/Mercado.
    """
    if ret_3m > 0.03 and vol < vol_median:
        return "bull"
    elif ret_3m < -0.03 or vol > 1.25 * vol_median:
        return "bear"
    return "neutro"

def discretize_dataframe_for_horizon(raw_df: pd.DataFrame, edges: dict, horizon_months: int) -> pd.DataFrame:
    """
    Aplica discretização nas features contínuas para o modelo BN num horizonte específico.
    Inclui features de fundamentos se estiverem presentes no dataframe.
    """
    target_col = f"Target_{horizon_months}M"
    if target_col not in raw_df.columns:
        return pd.DataFrame()
        
    df = raw_df.copy()
    df = df.dropna(subset=[target_col])
    
    # Discretiza contínuas
    for col in FEATURES_CONT:
        if col in df.columns and col in edges:
            labels = STATE_LABELS[col]
            df[col] = pd.cut(df[col], bins=edges[col], labels=labels, include_lowest=True)
            
    # As categóricas (macro, mercado) já estão em string
    for col in FEATURES_CAT:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    # Verifica fundamentos e garante formato
    for col in FEATURES_FUND:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    df[TARGET_COL] = df[target_col].map({1: "sim", 0: "nao"})
    
    cols = ["Date", "Ticker"] + FEATURES_CONT + FEATURES_CAT
    for col in FEATURES_FUND:
        if col in df.columns:
            cols.append(col)
    cols.append(TARGET_COL)
    
    # Verifica se todas as colunas existem
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"Colunas ausentes no discretize: {missing}")
        return pd.DataFrame()
        
    return df[cols].dropna()

def build_historical_dataset(price_hist: dict, selic: pd.Series, ipca12: pd.Series, ipca_mensal: pd.Series, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Constrói o dataset histórico tabular para todos os fundos.
    Opcionalmente recebe fundamentals_df para integrar P/VP, DY e Vacância ao modelo.
    """
    close_pivot = pd.DataFrame({t: df.set_index("Date")["Close"] for t, df in price_hist.items()})
    volume_pivot = pd.DataFrame({t: df.set_index("Date")["Volume"] for t, df in price_hist.items()})
    
    close_pivot = close_pivot.sort_index().ffill()
    volume_pivot = volume_pivot.sort_index().ffill()
    
    close_pivot = close_pivot.dropna(thresh=max(10, int(len(price_hist)*0.5)))
    volume_pivot = volume_pivot.loc[close_pivot.index]

    daily_ret = close_pivot.pct_change(fill_method=None)
    count_valid = daily_ret.notna().sum(axis=1)
    market_daily_ret = daily_ret.mean(axis=1)
    market_daily_ret[count_valid < 5] = np.nan
    market_index = (1 + market_daily_ret.fillna(0)).cumprod()
    
    vol_daily = daily_ret.rolling(63).std() * np.sqrt(252)
    market_vol_daily = market_daily_ret.rolling(63).std() * np.sqrt(252)
    
    turnover_daily = close_pivot * volume_pivot
    liq_daily = turnover_daily.rolling(63).mean()

    try:
        monthly_close = close_pivot.resample("ME").last()
        monthly_market_index = market_index.resample("ME").last()
        monthly_vol = vol_daily.resample("ME").last()
        monthly_market_vol = market_vol_daily.resample("ME").last()
        monthly_liq = liq_daily.resample("ME").last()
    except:
        monthly_close = close_pivot.resample("M").last()
        monthly_market_index = market_index.resample("M").last()
        monthly_vol = vol_daily.resample("M").last()
        monthly_market_vol = market_vol_daily.resample("M").last()
        monthly_liq = liq_daily.resample("M").last()

    selic_monthly = selic.reindex(monthly_close.index, method="ffill")
    ipca12_monthly = ipca12.reindex(monthly_close.index, method="ffill")
    ipca_mensal_monthly = ipca_mensal.reindex(monthly_close.index, method="ffill")
    selic_change_6m = selic_monthly - selic_monthly.shift(6)

    rows = []
    for ticker in monthly_close.columns:
        for date in monthly_close.index:
            price = monthly_close.loc[date, ticker]
            if pd.isna(price):
                continue
                
            mom_3m = price / monthly_close[ticker].shift(3).loc[date] - 1
            vol = monthly_vol.loc[date, ticker]
            liq = monthly_liq.loc[date, ticker]
            mkt_ret_3m = monthly_market_index.loc[date] / monthly_market_index.shift(3).loc[date] - 1
            mkt_vol = monthly_market_vol.loc[date]
            selic_val = selic_monthly.loc[date]
            ipca12_val = ipca12_monthly.loc[date]
            ipca_m_val = ipca_mensal_monthly.loc[date]
            selic_chg = selic_change_6m.loc[date]
            
            if pd.isna(mom_3m) or pd.isna(vol) or pd.isna(liq) or pd.isna(mkt_ret_3m) or pd.isna(mkt_vol):
                continue
            if pd.isna(selic_val) or pd.isna(ipca12_val):
                continue
                
            regime = classify_regime(selic_val, selic_chg, ipca12_val)
            
            mkt_vol_median = market_vol_daily.rolling(252).median().resample("ME" if hasattr(market_vol_daily.resample("ME"), "last") else "M").last()
            mkt_vol_med_val = mkt_vol_median.loc[date] if date in mkt_vol_median.index else 0.20
            if pd.isna(mkt_vol_med_val): mkt_vol_med_val = 0.20
                
            mkt_cond = classify_market(mkt_ret_3m, mkt_vol, mkt_vol_med_val)

            row = {
                "Date": date, "Ticker": ticker, "Momentum": mom_3m,
                "Volatilidade": vol, "Liquidez": liq, "Retorno_Mercado": mkt_ret_3m,
                "Volatilidade_Mercado": mkt_vol, "Selic": selic_val, "IPCA_12M": ipca12_val,
                "IPCA_Mensal": ipca_m_val, "Regime": regime, "Mercado": mkt_cond,
            }
            
            for h in HORIZONS_MONTHS:
                future_price = monthly_close[ticker].shift(-h).loc[date]
                if pd.isna(future_price):
                    row[f"Future_Return_{h}M"] = np.nan
                    row[f"Risk_Free_{h}M"] = np.nan
                    row[f"Target_{h}M"] = np.nan
                else:
                    future_ret = future_price / price - 1
                    risk_free = (1 + selic_val / 100) ** (h / 12) - 1
                    row[f"Future_Return_{h}M"] = future_ret
                    row[f"Risk_Free_{h}M"] = risk_free
                    row[f"Target_{h}M"] = 1 if future_ret > risk_free else 0
                    
            rows.append(row)

    df = pd.DataFrame(rows)
    
    if fundamentals_df is not None and not df.empty:
        # Drop Liquidez from fundamentals to avoid collision with historical Liquidez
        fund_merge = fundamentals_df.copy()
        if "Liquidez" in fund_merge.columns:
            fund_merge = fund_merge.drop(columns=["Liquidez"])
            
        # Merge de fundamentos
        if "Date" in fund_merge.columns:
            df = pd.merge(df, fund_merge, on=["Ticker", "Date"], how="left")
        else:
            df = pd.merge(df, fund_merge, on=["Ticker"], how="left")
            
        def pvp_cat(val):
            if pd.isna(val): return np.nan
            if val < 0.9: return "desconto"
            if val <= 1.1: return "justo"
            return "premio"
            
        def dy_cat(val):
            if pd.isna(val): return np.nan
            if val < 6.0: return "baixo"
            if val <= 10.0: return "medio"
            return "alto"
            
        def vac_cat(val):
            if pd.isna(val): return np.nan
            if val < 5.0: return "baixa"
            if val <= 15.0: return "media"
            return "alta"
            
        # P/VP
        pvp_col = next((c for c in ["P/VP", "PVP", "p/vp", "pvp"] if c in df.columns), None)
        if pvp_col:
            df["PVP_Cat"] = pd.to_numeric(df[pvp_col], errors="coerce").apply(pvp_cat)
            
        # DY
        dy_col = next((c for c in ["DY", "Dividend Yield", "dy", "dividend_yield"] if c in df.columns), None)
        if dy_col:
            df["DY_Cat"] = pd.to_numeric(df[dy_col], errors="coerce").apply(dy_cat)
            
        # Vacância
        vac_col = next((c for c in ["Vacância", "Vacancia", "Vacância Média", "vacancia"] if c in df.columns), None)
        if vac_col:
            df["Vacancia_Cat"] = pd.to_numeric(df[vac_col], errors="coerce").apply(vac_cat)

    logger.info(f"Dataset histórico gerado: {len(df)} linhas")
    return df
