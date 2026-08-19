# -*- coding: utf-8 -*-
"""Recomendador Inteligente de Fundos Imobiliários — Entry Point

Ponto de entrada do app Streamlit. Importa e orquestra todos os módulos:
  - config: constantes e parâmetros centralizados
  - data_sources: coleta de dados multi-fonte com fallback
  - features: construção do dataset histórico
  - model: rede bayesiana + métricas + benchmark
  - portfolio: score de adequação, Monte Carlo, alocação
  - persistence: armazenamento SQLite
  - chat: assistente IA (Gemini/Claude)
  - export: PDF/Excel
  - ui: componentes de interface
"""

import os
import sys
import logging
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Suprime warnings e configura logging ─────────────────────────────────
warnings.filterwarnings("ignore")

import config

# Garante que o diretório data/ existe
os.makedirs(config.DATA_DIR, exist_ok=True)

# Logging em arquivo + console
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("FII")

for lib in ["yfinance", "urllib3", "requests", "pandas_datareader", "investpy", "pgmpy"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Importações dos módulos internos ─────────────────────────────────────
from data_sources.macro import fetch_selic, fetch_ipca12, fetch_ipca_mensal
from data_sources.fii_universe import fetch_fii_universe
from data_sources.prices import fetch_price_history
try:
    from data_sources.cvm import fetch_cvm_recent_reports, fetch_cvm_relevant_facts
except ImportError:
    fetch_cvm_recent_reports = None
    fetch_cvm_relevant_facts = None

from features import build_historical_dataset, get_quantile_edges, discretize_dataframe_for_horizon
from model import (
    build_bn_model, get_bn_probs, get_current_predictions,
    train_models_walk_forward, compute_metrics, compute_calibration_curve,
    detect_drift, compute_training_stats, get_bn_dag_edges, get_cpd_table,
)
try:
    from model import train_benchmark_lgbm
except ImportError:
    train_benchmark_lgbm = None

from portfolio import adjust_score, project_price_monte_carlo, allocate_capital
from persistence import (
    load_interactions, save_interaction, load_raw_dataset, save_raw_dataset,
    load_model_meta, save_model_meta, evaluate_matured_interactions,
    merge_feedback_into_raw, save_portfolio,
)
from chat import get_ai_client, build_chat_context, get_system_prompt, stream_response, check_rate_limit

from ui.theme import apply_custom_css
from ui.disclaimer import show_disclaimer, show_privacy_notice
from ui.overview import show_overview
from ui.sidebar import show_sidebar
from ui.ranking import show_ranking
from ui.details import show_details
from ui.simulation import show_simulation
from ui.model_performance import show_model_performance
from ui.reports import show_reports
from ui.how_it_works import show_how_it_works
from ui.portfolio_tracker import show_portfolio_tracker

# ── Configuração da página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Recomendador Inteligente de FIIs",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

# ============================================================
# CARREGAMENTO DE DADOS DE MERCADO
# ============================================================
@st.cache_data(ttl=config.TTL_UNIVERSE, show_spinner=False)
def load_market_data(max_tickers: int = 400):
    """Coleta séries macro, universo de FIIs e preços."""
    logger.info("Coletando séries macroeconômicas...")
    selic = fetch_selic()
    ipca12 = fetch_ipca12()
    ipca_mensal = fetch_ipca_mensal()

    tickers, fundamentals_universo = fetch_fii_universe(max_tickers=max_tickers)

    logger.info(f"Coletando preços de {len(tickers)} FIIs...")
    price_hist = {}
    erros = []
    hoje = pd.Timestamp.today().normalize()
    limite_defasagem = hoje - pd.Timedelta(days=config.MAX_STALE_DAYS)
    progress_bar = st.progress(0, text=f"Buscando preços de {len(tickers)} fundos... (0/{len(tickers)})")
    concluidos = 0

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_PRICE) as executor:
        future_to_ticker = {executor.submit(fetch_price_history, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and len(df) > 10 and pd.to_datetime(df["Date"]).max() >= limite_defasagem:
                    price_hist[ticker] = df
                else:
                    erros.append(ticker)
            except Exception:
                erros.append(ticker)
            concluidos += 1
            progress_bar.progress(
                concluidos / len(tickers),
                text=f"Buscando preços de fundos imobiliários... ({concluidos}/{len(tickers)})"
            )
    progress_bar.empty()

    if erros:
        logger.warning(f"Falha na coleta de preços para {len(erros)} FIIs: {erros[:10]}...")

    tickers_validos = list(price_hist.keys())
    logger.info(f"FIIs com dados reais e recentes: {len(tickers_validos)}")
    return selic, ipca12, ipca_mensal, price_hist, tickers_validos, fundamentals_universo


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def main():
    # ── Sidebar ──────────────────────────────────────────────
    params = show_sidebar()
    capital = params["capital"]
    patrimonio_total = params["patrimonio_total"]
    profile = params["profile"]
    objective = params["objective"]
    horizon_months = params["horizon_months"]
    n_fiis = params["n_fiis"]
    criterio_alocacao = params["criterio_alocacao"]
    segment_cap = params.get("segment_cap", config.MAX_SEGMENT_WEIGHT)
    excluded_segments = params.get("excluded_segments", [])
    excluded_tickers = params.get("excluded_tickers", [])
    run_button = params["run_button"]
    retrain_clicked = params["retrain_clicked"]

    # ── Retrain: limpa caches ────────────────────────────────
    if retrain_clicked:
        fetch_fii_universe.clear()
        load_market_data.clear()
        for key in ["raw_all", "models", "calibrations", "metrics_by_horizon",
                     "training_stats", "benchmark_metrics"]:
            st.session_state.pop(key, None)

    # ── Carrega dados de mercado ─────────────────────────────
    with st.status("🔎 Estamos buscando o melhor para você...", expanded=True) as status:
        status.write("Carregando dados de mercado (Selic, IPCA, universo de FIIs e preços)...")
        selic, ipca12, ipca_mensal, price_hist, tickers_validos, fundamentals_universo = load_market_data(max_tickers=400)
        status.write(f"✅ Dados de mercado obtidos para {len(tickers_validos)} fundos ativos e líquidos.")

        if "raw_all" not in st.session_state:
            raw_file = load_raw_dataset()
            if raw_file.empty:
                status.write("Construindo base histórica (isso só acontece na primeira vez ou após retreinar)...")
                raw_all = build_historical_dataset(price_hist, selic, ipca12, ipca_mensal,
                                                   fundamentals_df=fundamentals_universo)
                save_raw_dataset(raw_all)
            else:
                raw_all = raw_file
            st.session_state.raw_all = raw_all
        else:
            raw_all = st.session_state.raw_all
        status.update(label="✅ Dados de mercado prontos", state="complete", expanded=False)

    # ── Dashboard inicial (sempre visível) ───────────────────
    show_overview(selic, ipca12, len(tickers_validos))

    # ── Avalia interações passadas ───────────────────────────
    interactions = load_interactions()
    if not interactions.empty:
        interactions = evaluate_matured_interactions(interactions, price_hist, selic)
        raw_all = merge_feedback_into_raw(raw_all, interactions, price_hist)
        save_raw_dataset(raw_all)
        st.session_state.raw_all = raw_all

    # ── Edges e treinamento ──────────────────────────────────
    dates_sorted = sorted(raw_all["Date"].unique())
    cutoff_date = dates_sorted[int(0.8 * len(dates_sorted))]
    train_for_bins = raw_all[raw_all["Date"] <= cutoff_date]
    edges = {col: get_quantile_edges(train_for_bins[col], q=4) for col in config.FEATURES_CONT}

    if "models" not in st.session_state:
        with st.status("🧠 Treinando as redes bayesianas...", expanded=True) as status:
            status.write("Analisando o histórico de todos os fundos para aprender padrões de risco e retorno...")

            # Verifica se temos features de fundamentos no dataset
            has_fund = all(f in raw_all.columns for f in config.FEATURES_FUND)

            models, calibrations, metrics_by_horizon = train_models_walk_forward(
                raw_all, edges, include_fund_features=has_fund
            )
            st.session_state.models = models
            st.session_state.calibrations = calibrations
            st.session_state.metrics_by_horizon = metrics_by_horizon

            # Estatísticas de treino para drift detection
            training_stats = compute_training_stats(train_for_bins)
            st.session_state.training_stats = training_stats

            # Benchmark LightGBM (se disponível)
            benchmark_metrics = {}
            if train_benchmark_lgbm is not None:
                for h in config.HORIZONS_MONTHS:
                    try:
                        _, bm = train_benchmark_lgbm(raw_all, edges, h)
                        if bm:
                            benchmark_metrics[h] = bm
                    except Exception as e:
                        logger.warning(f"Falha no benchmark LightGBM para {h}M: {e}")
            st.session_state.benchmark_metrics = benchmark_metrics

            agora_str = pd.Timestamp.now().strftime("%d/%m/%Y às %H:%M")
            save_model_meta({"last_trained": agora_str, "n_fiis_universo": len(tickers_validos)})
            status.update(label="✅ Modelos treinados", state="complete", expanded=False)

    models = st.session_state.models
    calibrations = st.session_state.calibrations
    metrics_by_horizon = st.session_state.get("metrics_by_horizon", {})
    training_stats = st.session_state.get("training_stats", {})
    benchmark_metrics = st.session_state.get("benchmark_metrics", {})

    if not models:
        st.error("Não foi possível treinar nenhum modelo. Verifique os dados.")
        return

    # ── Fundamentos ──────────────────────────────────────────
    faltantes = [t for t in tickers_validos if t not in fundamentals_universo.index]
    fundamental_df = fundamentals_universo.copy()
    if faltantes:
        for t in faltantes:
            fundamental_df.loc[t] = {"Preco": 100.0, "P/VP": 1.0, "DY": 8.0,
                                     "Vacancia": 5.0, "Liquidez": 1.0, "Segmento": "Desconhecido"}
    fundamental_df = fundamental_df.loc[[t for t in tickers_validos if t in fundamental_df.index]]

    # ── Features atuais de cada fundo ────────────────────────
    current_raw = raw_all[raw_all["Ticker"].isin(tickers_validos)].sort_values("Date").groupby("Ticker").tail(1).copy()
    current_raw = current_raw.set_index("Ticker")

    missing = set(tickers_validos) - set(current_raw.index)
    if missing:
        median_vals = current_raw[config.FEATURES_CONT].median()
        mode_vals = current_raw[config.FEATURES_CAT].mode().iloc[0] if not current_raw[config.FEATURES_CAT].mode().empty else pd.Series({"Regime": "neutro", "Mercado": "neutro"})
        for t in missing:
            current_raw.loc[t] = {
                "Date": current_raw["Date"].max() if "Date" in current_raw.columns else pd.Timestamp.today(),
                "Momentum": median_vals.get("Momentum", 0),
                "Volatilidade": median_vals.get("Volatilidade", 0.2),
                "Liquidez": median_vals.get("Liquidez", 1.0),
                "Retorno_Mercado": current_raw.get("Retorno_Mercado", pd.Series([0])).median(),
                "Volatilidade_Mercado": current_raw.get("Volatilidade_Mercado", pd.Series([0.2])).median(),
                "Selic": current_raw.get("Selic", pd.Series([10])).median(),
                "IPCA_12M": current_raw.get("IPCA_12M", pd.Series([5])).median(),
                "IPCA_Mensal": current_raw.get("IPCA_Mensal", pd.Series([0.5])).median(),
                "Regime": mode_vals.get("Regime", "neutro"),
                "Mercado": mode_vals.get("Mercado", "neutro"),
            }

    # Discretização para a BN
    current_discrete = pd.DataFrame(index=current_raw.index)
    for col in config.FEATURES_CONT:
        if col in edges:
            current_discrete[col] = pd.cut(
                current_raw[col].clip(edges[col][0], edges[col][-1]),
                bins=edges[col], labels=config.STATE_LABELS[col], include_lowest=True
            )
    for col in config.FEATURES_CAT:
        current_discrete[col] = current_raw[col].apply(
            lambda x: x if x in config.STATE_LABELS[col] else config.STATE_LABELS[col][-1]
        )
    # Features fundamentalistas discretizadas (se BN expandida)
    has_fund_model = any(f in config.FEATURES_FUND for f in (list(models.values())[0].nodes() if models else []))
    if has_fund_model:
        for t in current_discrete.index:
            if t in fundamental_df.index:
                pvp = fundamental_df.loc[t, "P/VP"] if "P/VP" in fundamental_df.columns else 1.0
                dy = fundamental_df.loc[t, "DY"] if "DY" in fundamental_df.columns else 8.0
                vac = fundamental_df.loc[t, "Vacancia"] if "Vacancia" in fundamental_df.columns else 5.0
                current_discrete.loc[t, "PVP_Cat"] = "desconto" if pvp < 0.9 else ("premio" if pvp > 1.1 else "justo")
                current_discrete.loc[t, "DY_Cat"] = "baixo" if dy < 6 else ("alto" if dy > 10 else "medio")
                current_discrete.loc[t, "Vacancia_Cat"] = "baixa" if vac < 5 else ("alta" if vac > 15 else "media")

    # ── Predições da BN ──────────────────────────────────────
    bn_horizon_months = min(config.HORIZONS_MONTHS, key=lambda h: abs(h - horizon_months))
    bn_probs = get_current_predictions(models, calibrations, current_discrete, bn_horizon_months)

    # Bucket textual do horizonte
    if horizon_months <= 18:
        horizon = "curto"
    elif horizon_months <= 72:
        horizon = "médio"
    else:
        horizon = "longo"

    # Score fundamentalista
    fund_percentiles = pd.DataFrame({
        "pct_pvp": fundamental_df["P/VP"].rank(pct=True),
        "pct_dy": fundamental_df["DY"].rank(pct=True),
        "pct_vac": fundamental_df["Vacancia"].rank(pct=True),
        "pct_liq": fundamental_df["Liquidez"].rank(pct=True),
    })
    score_fund = (0.25 * (1 - fund_percentiles["pct_pvp"]) +
                  0.35 * fund_percentiles["pct_dy"] +
                  0.20 * (1 - fund_percentiles["pct_vac"]) +
                  0.20 * fund_percentiles["pct_liq"]).clip(0, 1)

    base_prob_series = pd.Series(0.7 * bn_probs + 0.3 * score_fund.values[:len(bn_probs)],
                                 index=current_discrete.index[:len(bn_probs)]).clip(0, 1)

    preco_mercado = {t: price_hist[t]["Close"].iloc[-1] for t in tickers_validos if t in price_hist}

    # Monta DataFrame consolidado
    valid_tickers = [t for t in tickers_validos if t in preco_mercado and t in fundamental_df.index and t in base_prob_series.index]
    current_df = pd.DataFrame(index=valid_tickers)
    current_df["Preco"] = [preco_mercado[t] for t in valid_tickers]
    current_df["P/VP"] = fundamental_df.loc[valid_tickers, "P/VP"].values
    current_df["DY"] = fundamental_df.loc[valid_tickers, "DY"].values
    current_df["Vacancia"] = fundamental_df.loc[valid_tickers, "Vacancia"].values
    current_df["Liquidez"] = fundamental_df.loc[valid_tickers, "Liquidez"].values
    current_df["Segmento"] = fundamental_df.loc[valid_tickers, "Segmento"].values
    current_df["Momentum"] = current_raw.loc[valid_tickers, "Momentum"].values if "Momentum" in current_raw.columns else 0
    current_df["Volatilidade"] = current_raw.loc[valid_tickers, "Volatilidade"].values if "Volatilidade" in current_raw.columns else 0.2
    current_df["BN_Prob"] = bn_probs[:len(valid_tickers)]
    current_df["Score_Fund"] = score_fund.loc[valid_tickers].values if hasattr(score_fund, 'loc') else score_fund.values[:len(valid_tickers)]
    current_df["Base_Prob"] = base_prob_series.loc[valid_tickers].values

    # Aplica exclusões do usuário
    if excluded_segments:
        current_df = current_df[~current_df["Segmento"].isin(excluded_segments)]
    if excluded_tickers:
        current_df = current_df[~current_df.index.isin(excluded_tickers)]

    # Score de adequação
    scores = current_df.apply(lambda r: adjust_score(r, current_df, profile, objective, horizon), axis=1)
    current_df["Adequação"] = scores
    current_df = current_df.sort_values("Adequação", ascending=False)

    top_df = current_df.head(n_fiis).copy()

    # Detecta regime atual para Monte Carlo condicionado
    current_regime = current_raw.iloc[0].get("Regime", "neutro") if not current_raw.empty else "neutro"

    # ── Projeções Monte Carlo ────────────────────────────────
    projections = []
    for t in top_df.index:
        price = top_df.loc[t, "Preco"]
        vol = top_df.loc[t, "Volatilidade"]
        bp = top_df.loc[t, "Base_Prob"]
        mc_result = project_price_monte_carlo(price, bp, vol, months=horizon_months, regime=current_regime)
        projections.append({
            "Ticker": t,
            "Preco_Projetado": mc_result["expected_price"],
            "Retorno_Esperado": mc_result["ret_esperado"],
            "Prob_Queda": mc_result["prob_queda"],
            "P10": mc_result.get("percentile_10", price * 0.8),
            "P90": mc_result.get("percentile_90", price * 1.2),
        })
    proj_df = pd.DataFrame(projections).set_index("Ticker")
    top_df = top_df.join(proj_df)
    top_df["Div_Anual_Estimado"] = top_df["Preco"] * top_df["DY"] / 100
    top_df["Chance_Sucesso"] = 1 - top_df["Prob_Queda"]

    # ── Alocação de capital ──────────────────────────────────
    coluna_peso = "Adequação" if criterio_alocacao == "Adequação ao perfil" else "Chance_Sucesso"
    top_df = allocate_capital(top_df, capital, coluna_peso, segment_cap=segment_cap)

    # ── Drift detection ──────────────────────────────────────
    drift_warnings = []
    if training_stats:
        drift_warnings = detect_drift(current_raw[config.FEATURES_CONT], training_stats)

    # ── Formata display ──────────────────────────────────────
    display_df = top_df[[
        "Segmento", "Preco", "P/VP", "DY", "Vacancia", "Liquidez",
        "Div_Anual_Estimado", "Adequação", "Valor_Alocado", "Cotas_Estimadas",
        "Chance_Sucesso", "Preco_Projetado", "Retorno_Esperado", "Prob_Queda"
    ]].copy()
    display_df["Liquidez"] = display_df["Liquidez"].apply(lambda x: f"R$ {x:.2f} mi")
    display_df["DY"] = display_df["DY"].apply(lambda x: f"{x:.1f}%")
    display_df["Vacancia"] = display_df["Vacancia"].apply(lambda x: f"{x:.1f}%")
    display_df["Valor_Alocado"] = display_df["Valor_Alocado"].apply(lambda x: f"R$ {x:,.2f}")
    display_df["Chance_Sucesso"] = display_df["Chance_Sucesso"].apply(lambda x: f"{x:.1%}")
    display_df["Div_Anual_Estimado"] = display_df["Div_Anual_Estimado"].apply(lambda x: f"R$ {x:.2f}")
    display_df["Adequação"] = display_df["Adequação"].apply(lambda x: f"{x:.2%}")
    display_df["Retorno_Esperado"] = display_df["Retorno_Esperado"].apply(lambda x: f"{x:+.2%}")
    display_df["Prob_Queda"] = display_df["Prob_Queda"].apply(lambda x: f"{x:.2%}")
    display_df["Preco_Projetado"] = display_df["Preco_Projetado"].apply(lambda x: f"R$ {x:.2f}")
    display_df = display_df.rename(columns={
        "Preco": "Preço Atual", "Vacancia": "Vacância", "Liquidez": "Liquidez Diária",
        "Div_Anual_Estimado": "Div. Anual Estimado", "Adequação": "Adequação ao Perfil",
        "Valor_Alocado": "Valor a Investir", "Cotas_Estimadas": "Cotas",
        "Chance_Sucesso": "Chance de Sucesso", "Preco_Projetado": "Preço Projetado",
        "Retorno_Esperado": "Retorno Esperado", "Prob_Queda": "Prob. de Queda",
    })

    valor_investido_total = top_df["Valor_Alocado"].sum()
    sobra = capital - valor_investido_total

    # ── Salva interações ─────────────────────────────────────
    if run_button:
        st.success(f"Análise concluída para **{profile}**, **{objective}**, horizonte **{horizon_months} meses**.")

        now = pd.Timestamp.now()
        target_date = now + pd.DateOffset(months=horizon_months)
        for t in top_df.index:
            save_interaction({
                "Timestamp": now.isoformat(),
                "Ticker": t,
                "Capital": capital,
                "Profile": profile,
                "Objective": objective,
                "Horizon_Months": horizon_months,
                "Price_At_Rec": top_df.loc[t, "Preco"],
                "Date_At_Rec": now.isoformat(),
                "Target_Date": target_date.isoformat(),
                "Predicted_Prob": top_df.loc[t, "Base_Prob"],
                "Recommended": 1,
                "Actual_Return": None,
                "Risk_Free_Return": None,
                "Outcome": None,
                "Evaluated": 0,
            })
        st.info("Suas recomendações foram armazenadas e serão avaliadas após o prazo do horizonte.")

    # ── Abas ─────────────────────────────────────────────────
    tab_names = ["📊 Ranking", "🔍 Detalhes", "📈 Simulação",
                 "🧠 Desempenho do Modelo", "📄 Relatórios",
                 "📁 Meu Portfólio", "❓ Como Funciona"]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Ranking
        if run_button:
            show_ranking(display_df, top_df, fund_percentiles, params)
            st.caption(
                f"💰 Total alocado: R$ {valor_investido_total:,.2f} de R$ {capital:,.2f} "
                f"(R$ {sobra:,.2f} não alocado por arredondamento de cotas inteiras). "
                f"A divisão entre fundos é proporcional à **{criterio_alocacao}** de cada um."
            )
        else:
            st.info("Clique em **🚀 Executar Análise** na barra lateral para gerar o ranking.")

    with tabs[1]:  # Detalhes
        show_details(current_df, price_hist)

    with tabs[2]:  # Simulação
        show_simulation(current_df, horizon_months, current_regime)

    with tabs[3]:  # Desempenho do Modelo
        show_model_performance(
            metrics_by_horizon=metrics_by_horizon,
            models=models,
            benchmark_metrics=benchmark_metrics,
            drift_warnings=drift_warnings,
        )

    with tabs[4]:  # Relatórios
        show_reports(tickers_validos)

    with tabs[5]:  # Meu Portfólio
        show_portfolio_tracker(price_hist, selic)

    with tabs[6]:  # Como Funciona
        show_how_it_works()

    # ── Chat com IA ──────────────────────────────────────────
    st.divider()
    st.markdown("### 💬 Converse com a IA sobre a recomendação")
    st.caption(
        "Pergunte sobre os fundos do ranking, o que significam os indicadores, "
        "ou peça para comparar dois FIIs. A IA responde com base apenas nos "
        "dados calculados acima — não é uma recomendação de investimento."
    )

    # Sugestões rápidas
    suggestion_cols = st.columns(len(config.CHAT_QUICK_SUGGESTIONS))
    for i, suggestion in enumerate(config.CHAT_QUICK_SUGGESTIONS):
        if suggestion_cols[i].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.setdefault("chat_history", []).append({"role": "user", "content": suggestion})
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pergunta = st.chat_input("Pergunte algo sobre os fundos recomendados...")
    if pergunta:
        # Rate limit check
        allowed, limit_msg = check_rate_limit(st.session_state)
        if not allowed:
            st.warning(limit_msg)
        else:
            st.session_state.chat_history.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.markdown(pergunta)

            with st.chat_message("assistant"):
                provider, client = get_ai_client()
                if client is None:
                    resposta = (
                        "⚠️ Nenhuma chave de API configurada. Adicione `GOOGLE_API_KEY` "
                        "ou `ANTHROPIC_API_KEY` em Settings → Secrets para habilitar o chat."
                    )
                    st.markdown(resposta)
                else:
                    contexto = build_chat_context(top_df, profile, objective, horizon_months, criterio_alocacao)
                    system_prompt = get_system_prompt(contexto)
                    historico_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]

                    try:
                        resposta_container = st.empty()
                        resposta_acumulada = ""
                        for chunk in stream_response(provider, client, historico_api, system_prompt):
                            resposta_acumulada += chunk
                            resposta_container.markdown(resposta_acumulada + "▌")
                        resposta_container.markdown(resposta_acumulada)
                        resposta = resposta_acumulada
                    except Exception as e:
                        resposta = f"⚠️ Não foi possível obter resposta da IA agora ({e})."
                        st.markdown(resposta)

                st.session_state.chat_history.append({"role": "assistant", "content": resposta})

    # ── Disclaimer permanente ────────────────────────────────
    show_disclaimer()
    show_privacy_notice()


if __name__ == "__main__":
    main()
