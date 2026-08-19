import pandas as pd
import numpy as np
import logging
import config

logger = logging.getLogger('FII.portfolio')

def adjust_score(row, current_df, profile, objective, horizon) -> float:
    """
    Ajusta o score base do modelo considerando perfil, objetivo e horizonte do investidor.
    """
    base = row["Base_Prob"]
    pct_vol = current_df["Volatilidade"].rank(pct=True).loc[row.name]
    pct_mom = current_df["Momentum"].rank(pct=True).loc[row.name]
    pct_liq = current_df["Liquidez"].rank(pct=True).loc[row.name]
    pct_dy = current_df["DY"].rank(pct=True).loc[row.name]
    pct_pvp = current_df["P/VP"].rank(pct=True).loc[row.name]
    pct_vac = current_df["Vacancia"].rank(pct=True).loc[row.name]

    adj = 0.0
    if objective == "renda":
        adj += 0.15 * pct_dy + 0.05 * (1 - pct_vac)
    elif objective == "valorização":
        adj += 0.10 * pct_mom + 0.10 * (1 - pct_pvp)
    elif objective == "equilíbrio":
        adj += 0.05 * pct_dy + 0.05 * pct_mom

    if profile == "conservador":
        adj -= 0.15 * pct_vol
    elif profile == "moderado":
        adj -= 0.05 * pct_vol
    elif profile == "agressivo":
        adj += 0.10 * pct_vol

    if horizon == "curto":
        adj -= 0.10 * pct_vol + 0.10 * (1 - pct_liq)
    elif horizon == "longo":
        adj += 0.05 * pct_vol

    # Bônus para fundos tradicionais e altamente líquidos (ex: MXRF11)
    adj += 0.15 * pct_liq

    return float(np.clip(base + adj, 0, 1))

def project_price_monte_carlo(current_price, base_prob, vol_annual, months, regime='neutro', n_sim=None, seed=None) -> dict:
    """
    Projeta o preço através de simulação de Monte Carlo.
    """
    if n_sim is None:
        n_sim = getattr(config, 'MC_N_SIMULATIONS', 3000)
    if seed is None:
        seed = getattr(config, 'MC_SEED', 42)
        
    rng = np.random.default_rng(seed)
    sigma_m = vol_annual / np.sqrt(12)
    annual_drift = (base_prob - 0.5) * 0.30
    drift_m = annual_drift / 12
    
    scenarios = getattr(config, 'MC_SCENARIOS_BY_REGIME', {}).get(regime, getattr(config, 'MC_SCENARIOS_DEFAULT', {
        "bull": {"prob": 0.35, "drift": 0.010},
        "neutral": {"prob": 0.40, "drift": 0.000},
        "bear": {"prob": 0.25, "drift": -0.015},
    }))
    
    bull_scen = scenarios.get("bull", {"prob": 0.33, "drift": 0.010})
    neutral_scen = scenarios.get("neutral", {"prob": 0.34, "drift": 0.00})
    bear_scen = scenarios.get("bear", {"prob": 0.33, "drift": -0.015})
    
    # Gera números aleatórios fixando a seed para consistência (determinismo local)
    scenario_draws = rng.random(n_sim)
    bull_mask = scenario_draws < bull_scen["prob"]
    bear_mask = scenario_draws > (1 - bear_scen["prob"])
    neutral_mask = ~(bull_mask | bear_mask)
    
    scenario_drifts = np.zeros(n_sim)
    scenario_drifts[bull_mask] = bull_scen["drift"]
    scenario_drifts[neutral_mask] = neutral_scen["drift"]
    scenario_drifts[bear_mask] = bear_scen["drift"]
    
    all_final_prices = []
    for i in range(n_sim):
        # Determina o drift do cenário
        drift = drift_m + scenario_drifts[i]
        shocks = rng.normal(drift, sigma_m, months)
        path = current_price * np.exp(np.cumsum(shocks))
        all_final_prices.append(path[-1])
        
    expected_price = np.mean(all_final_prices)
    prob_queda = np.mean(np.array(all_final_prices) < current_price)
    ret_esperado = (expected_price / current_price) - 1
    percentile_10 = np.percentile(all_final_prices, 10)
    percentile_90 = np.percentile(all_final_prices, 90)

    return {
        'expected_price': expected_price,
        'ret_esperado': ret_esperado,
        'prob_queda': prob_queda,
        'percentile_10': percentile_10,
        'percentile_90': percentile_90,
        'all_final_prices': np.array(all_final_prices)
    }

def allocate_capital(top_df, capital, weight_column, segment_cap=None) -> pd.DataFrame:
    """
    Aloca o capital nos FIIs escolhidos, respeitando opcionalmente um limite por segmento.
    Garante compra de cotas inteiras e alocação gulosa do troco.
    """
    df = top_df.copy()
    peso = df[weight_column].clip(lower=0)
    if peso.sum() > 0:
        peso = peso / peso.sum()
    else:
        peso = pd.Series(1 / len(df), index=df.index)
        
    df['Peso_Inicial'] = peso
    
    if segment_cap is not None:
        while True:
            seg_weights = df.groupby('Segmento')['Peso_Inicial'].sum()
            over_cap_segs = seg_weights[seg_weights > segment_cap].index
            
            if len(over_cap_segs) == 0:
                break
                
            excess = 0
            for seg in over_cap_segs:
                seg_total = seg_weights[seg]
                ratio = segment_cap / seg_total
                df.loc[df['Segmento'] == seg, 'Peso_Inicial'] *= ratio
                excess += (seg_total - segment_cap)
            
            under_cap_mask = ~df['Segmento'].isin(over_cap_segs)
            if under_cap_mask.sum() == 0:
                break
                
            under_cap_sum = df.loc[under_cap_mask, 'Peso_Inicial'].sum()
            if under_cap_sum > 0:
                df.loc[under_cap_mask, 'Peso_Inicial'] += (df.loc[under_cap_mask, 'Peso_Inicial'] / under_cap_sum) * excess
            else:
                break

    target_value = df['Peso_Inicial'] * capital
    df['Cotas_Estimadas'] = np.floor(target_value / df['Preco']).astype(int)
    df['Valor_Investido'] = df['Cotas_Estimadas'] * df['Preco']
    
    remaining_cash = capital - df['Valor_Investido'].sum()
    
    # Alocação gulosa do restante para aproveitar o máximo do capital
    df_sorted = df.sort_values(by='Peso_Inicial', ascending=False)
    for ticker, row in df_sorted.iterrows():
        price = row['Preco']
        if price <= remaining_cash:
            add_cotas = int(remaining_cash // price)
            df.loc[ticker, 'Cotas_Estimadas'] += add_cotas
            df.loc[ticker, 'Valor_Investido'] += add_cotas * price
            remaining_cash -= add_cotas * price

    # O Valor_Alocado final que será mostrado na UI é o quanto realmente foi investido
    df['Valor_Alocado'] = df['Valor_Investido']
    
    return df

def compute_correlation_penalty(price_hist, tickers, lookback_days=252) -> pd.Series:
    """
    Calcula a correlação média com outros tickers do mesmo segmento. Retorna penalidade [0, 0.15].
    """
    # Assuming price_hist is a dict {ticker: DataFrame with 'Close'} or a wide DataFrame
    # If dict of dataframes
    if isinstance(price_hist, dict):
        closes = {}
        for t in tickers:
            if t in price_hist and 'Close' in price_hist[t]:
                closes[t] = price_hist[t]['Close'].tail(lookback_days)
        price_df = pd.DataFrame(closes)
    else:
        price_df = price_hist[tickers].tail(lookback_days)
        
    ret_df = price_df.pct_change().dropna()
    corr_matrix = ret_df.corr()
    
    penalties = pd.Series(0.0, index=tickers)
    for t in tickers:
        corrs = corr_matrix[t].drop(t, errors='ignore')
        if len(corrs) > 0:
            avg_corr = corrs.mean()
            # Escala correlação [-1, 1] para penalidade [0, 0.15] (penaliza corr positiva)
            penalties[t] = float(np.clip(avg_corr * 0.15, 0, 0.15))
            
    return penalties

def suggest_rebalancing(current_positions: dict, target_allocation: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """
    Compara as posições atuais com a alocação alvo e sugere rebalanceamento.
    """
    suggestions = []
    
    for t in target_allocation.index:
        target_cotas = target_allocation.loc[t, 'Cotas_Estimadas']
        current_cotas = current_positions.get(t, 0)
        diff = target_cotas - current_cotas
        price = prices.get(t, target_allocation.loc[t, 'Preco'] if 'Preco' in target_allocation.columns else 0.0)
        
        if diff > 0:
            action = "Comprar"
        elif diff < 0:
            action = "Vender"
        else:
            action = "Manter"
            
        suggestions.append({
            'Ticker': t,
            'Ação': action,
            'Cotas': int(abs(diff)),
            'Valor': abs(diff) * price
        })
        
    for t, current_cotas in current_positions.items():
        if t not in target_allocation.index and current_cotas > 0:
            price = prices.get(t, 0.0)
            suggestions.append({
                'Ticker': t,
                'Ação': "Vender",
                'Cotas': int(current_cotas),
                'Valor': current_cotas * price
            })
            
    return pd.DataFrame(suggestions)
