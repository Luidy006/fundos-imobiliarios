# -*- coding: utf-8 -*-
"""Configuração centralizada do Recomendador de FIIs.

Todas as constantes que antes estavam espalhadas pelo código monolítico
vivem aqui.  Importar com `from config import CFG` e acessar como atributos.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ── URLs de fontes de dados ──────────────────────────────────────────────
FUNDAMENTUS_URL = "https://www.fundamentus.com.br/fii_resultado.php"
BCB_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato={fmt}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.SA?period1={p1}&period2={p2}&interval=1d"
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}.sa&i=d"
STATUS_INVEST_URL = "https://statusinvest.com.br/fundos-imobiliarios/{ticker}"
CVM_OPEN_DATA_BASE = "https://dados.cvm.gov.br/dados/FI/DOC"
CVM_MONTHLY_URL = CVM_OPEN_DATA_BASE + "/INF_MENSAL/DADOS/inf_mensal_fii_{year}{month:02d}.csv"
CVM_CADASTRO_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
FUNDOS_NET_BASE = "https://fnet.bmfbovespa.com.br/fnet/publico"

# ── Headers HTTP ─────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

# ── Cache TTLs (segundos) ────────────────────────────────────────────────
TTL_PRICES       = 4 * 3600       # 4 horas
TTL_FUNDAMENTALS = 24 * 3600      # 24 horas
TTL_MACRO        = 12 * 3600      # 12 horas
TTL_CVM_DOCS     = 7 * 24 * 3600  # 7 dias
TTL_UNIVERSE     = 24 * 3600      # 24 horas

# ── Retry / HTTP ─────────────────────────────────────────────────────────
MAX_RETRIES = 4
BASE_BACKOFF_S = 0.5     # backoff exponencial: 0.5, 1.0, 2.0, 4.0 + jitter
REQUEST_TIMEOUT = 20
MAX_WORKERS_PRICE = 15   # ThreadPoolExecutor para busca de preços

# ── Modelo Bayesiano ─────────────────────────────────────────────────────
HORIZONS_MONTHS: List[int] = [3, 12, 36, 120]
FEATURES_CONT: List[str] = ["Momentum", "Volatilidade", "Liquidez"]
FEATURES_CAT: List[str] = ["Regime", "Mercado"]
TARGET_COL = "Recomendacao"

STATE_LABELS: Dict[str, List[str]] = {
    "Momentum":      ["muito_baixo", "baixo", "alto", "muito_alto"],
    "Volatilidade":  ["muito_baixa", "baixa", "alta", "muito_alta"],
    "Liquidez":      ["muito_baixa", "baixa", "alta", "muito_alta"],
    "Regime":        ["crescimento", "estagflacao", "neutro"],
    "Mercado":       ["bull", "bear", "neutro"],
    "Recomendacao":  ["nao", "sim"],
}

# Nós extras (fundamentos) que podem ser incluídos na BN expandida
FEATURES_FUND: List[str] = ["PVP_Cat", "DY_Cat", "Vacancia_Cat"]
STATE_LABELS_FUND: Dict[str, List[str]] = {
    "PVP_Cat":      ["desconto", "justo", "premio"],
    "DY_Cat":       ["baixo", "medio", "alto"],
    "Vacancia_Cat": ["baixa", "media", "alta"],
}

# Walk-forward validation
WALK_FORWARD_MIN_TRAIN_MONTHS = 24
WALK_FORWARD_STEP_MONTHS = 6

# ── Monte Carlo ──────────────────────────────────────────────────────────
MC_N_SIMULATIONS = 3000
MC_SEED = 42

# Cenários macro padrão (sobrescritos pelo regime detectado)
MC_SCENARIOS_DEFAULT = {
    "bull":    {"prob": 0.35, "drift": 0.010},
    "neutral": {"prob": 0.40, "drift": 0.000},
    "bear":    {"prob": 0.25, "drift": -0.015},
}

# Cenários condicionados ao regime macro
MC_SCENARIOS_BY_REGIME = {
    "crescimento": {
        "bull":    {"prob": 0.50, "drift": 0.015},
        "neutral": {"prob": 0.35, "drift": 0.002},
        "bear":    {"prob": 0.15, "drift": -0.010},
    },
    "estagflacao": {
        "bull":    {"prob": 0.15, "drift": 0.005},
        "neutral": {"prob": 0.35, "drift": -0.003},
        "bear":    {"prob": 0.50, "drift": -0.020},
    },
    "neutro": MC_SCENARIOS_DEFAULT,  # referência ao default
}

# ── Alocação e diversificação ────────────────────────────────────────────
MAX_SEGMENT_WEIGHT = 0.35   # teto de 35% do capital em um único segmento
PATRIMONIO_RULE_PCT = 0.25  # regra dos 25% do patrimônio total

# ── Validação de dados ───────────────────────────────────────────────────
PRICE_MIN = 0.01
PRICE_MAX = 50_000.0
PVP_MIN = 0.05
PVP_MAX = 5.0
DY_MIN = 0.0
DY_MAX = 40.0    # DY > 40% quase certamente é erro
VACANCIA_MIN = 0.0
VACANCIA_MAX = 100.0
LIQUIDEZ_MIN = 0.0

# Defasagem máxima (dias) para considerar um fundo ativo
MAX_STALE_DAYS = 25

# ── Chat com IA ──────────────────────────────────────────────────────────
CHAT_MAX_MSGS_PER_SESSION = 30
CHAT_MODEL_GEMINI = "gemini-2.5-flash"
CHAT_MODEL_CLAUDE = "claude-sonnet-4-6"
CHAT_MAX_TOKENS = 1200

CHAT_QUICK_SUGGESTIONS = [
    "Compare os dois primeiros colocados do ranking",
    "Por que esse fundo tem mais risco que os outros?",
    "Qual fundo paga mais dividendos?",
    "Qual a diferença entre fundos de papel e de tijolo?",
]

# ── Persistência (SQLite) ────────────────────────────────────────────────
import os as _os
DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
SQLITE_DB_PATH = _os.path.join(DATA_DIR, "fii_app.db")
SNAPSHOT_PATH = _os.path.join(DATA_DIR, "snapshot_universe.json")
LOG_FILE_PATH = _os.path.join(DATA_DIR, "fii_app.log")

# ── Série BCB (códigos) ─────────────────────────────────────────────────
BCB_SELIC_CODES = [432, 4189, 1178]
BCB_IPCA12_CODES = [13522]
BCB_IPCA_MENSAL_CODES = [433]
BCB_USD_BRL_CODES = [1]          # PTAX venda
BCB_IFIX_CODES = [12364]         # IFIX (se disponível)

# ── Logging ──────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = "INFO"

# ── Segmentos conhecidos ─────────────────────────────────────────────────
KNOWN_SEGMENTS = [
    "Logístico", "Shopping", "Híbrido", "Recebíveis", "Lajes Corporativas",
    "Fundo de Fundos", "Agro", "Residencial", "Hotel", "Hospitalar",
    "Educacional", "Varejo", "Outros", "Desconhecido",
]

# ── Disclaimer ───────────────────────────────────────────────────────────
DISCLAIMER_TEXT = (
    "⚠️ **Aviso Legal:** Este aplicativo é uma ferramenta analítica educacional. "
    "Não constitui recomendação de investimento, consultoria financeira, nem "
    "substitui análise de profissional credenciado pela CVM. Investimentos em "
    "fundos imobiliários envolvem riscos, incluindo perda do capital investido. "
    "Rentabilidade passada não garante resultados futuros. Consulte um assessor "
    "de investimentos antes de tomar qualquer decisão."
)

PRIVACY_NOTICE = (
    "🔒 **Privacidade:** Os dados informados (capital, perfil, objetivo) são "
    "armazenados localmente apenas para melhorar as recomendações futuras. "
    "Nenhum dado pessoal é compartilhado com terceiros."
)

# ── Dados estáticos de fallback (último recurso) ────────────────────────
import pandas as _pd

STATIC_TICKERS = [
    "HGLG11","KNRI11","XPML11","VISC11","BTLG11","MXRF11","CPTS11","RBRR11",
    "RBRF11","HFOF11","KNCR11","IRDM11","BRCO11","RECR11","SNCI11","XPSF11",
    "VRTA11","TGAR11","VSLH11","BCRI11","RZAK11","VINO11","RZTR11","SARE11",
    "ALZR11","BARI11","BCFF11","BMLC11","BRCR11","BTAL11","CARE11","CVBI11",
    "DEVA11","GARE11","HABT11","HCTR11","HSLG11","HGRU11","HGRE11","HGBS11",
    "JSRE11","KISU11","KNIP11","MALL11","MFII11","MGFF11","NSLU11","PATL11",
    "RBED11","RBFF11","RBRP11","RECT11","RFOF11","RMAI11","SDIL11","SHPH11",
    "TEPP11","URPR11","VCJR11","VGHF11","VGIP11","VGIR11","VILG11","XPCI11",
    "XPLG11","XPIN11",
]

_STATIC_FUND_DICT = {
    "HGLG11": {"Preco": 145.11, "P/VP": 1.02, "DY": 7.8, "Vacancia": 5.2, "Liquidez": 8.5, "Segmento": "Logístico"},
    "KNRI11": {"Preco": 149.12, "P/VP": 1.10, "DY": 7.1, "Vacancia": 3.5, "Liquidez": 6.0, "Segmento": "Híbrido"},
    "XPML11": {"Preco": 100.84, "P/VP": 1.00, "DY": 8.5, "Vacancia": 6.0, "Liquidez": 12.0, "Segmento": "Shopping"},
    "VISC11": {"Preco": 101.07, "P/VP": 1.05, "DY": 7.5, "Vacancia": 5.0, "Liquidez": 5.2, "Segmento": "Shopping"},
    "BTLG11": {"Preco": 98.31, "P/VP": 0.98, "DY": 8.0, "Vacancia": 4.8, "Liquidez": 4.5, "Segmento": "Logístico"},
    "MXRF11": {"Preco": 9.24, "P/VP": 0.90, "DY": 11.2, "Vacancia": 7.5, "Liquidez": 9.8, "Segmento": "Recebíveis"},
    "CPTS11": {"Preco": 7.30, "P/VP": 0.85, "DY": 10.5, "Vacancia": 0.0, "Liquidez": 3.2, "Segmento": "Recebíveis"},
    "KNCR11": {"Preco": 105.24, "P/VP": 0.98, "DY": 8.2, "Vacancia": 0.0, "Liquidez": 7.0, "Segmento": "Recebíveis"},
    "IRDM11": {"Preco": 62.40, "P/VP": 0.92, "DY": 10.0, "Vacancia": 4.5, "Liquidez": 4.0, "Segmento": "Recebíveis"},
    "BRCO11": {"Preco": 110.00, "P/VP": 1.02, "DY": 7.6, "Vacancia": 7.0, "Liquidez": 5.5, "Segmento": "Logístico"},
}
STATIC_FUNDAMENTALS = _pd.DataFrame(_STATIC_FUND_DICT).T
