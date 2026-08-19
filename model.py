import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from config import (
    FEATURES_CONT,
    FEATURES_CAT,
    FEATURES_FUND,
    TARGET_COL,
    HORIZONS_MONTHS,
    WALK_FORWARD_MIN_TRAIN_MONTHS,
    WALK_FORWARD_STEP_MONTHS,
)
from features import discretize_dataframe_for_horizon

logger = logging.getLogger("FII.model")

def build_bn_model(data: pd.DataFrame, include_fund_features: bool = False) -> DiscreteBayesianNetwork:
    """
    Constrói e treina a Rede Bayesiana com base nos dados fornecidos.
    Se include_fund_features for True e os dados possuírem as colunas, 
    adiciona arestas dos fundamentos (PVP_Cat, DY_Cat, Vacancia_Cat) para Recomendacao.
    """
    edges = [
        ("Regime", "Mercado"),
        ("Regime", "Recomendacao"),
        ("Mercado", "Recomendacao"),
        ("Momentum", "Recomendacao"),
        ("Volatilidade", "Recomendacao"),
        ("Liquidez", "Recomendacao"),
    ]
    
    nodes = FEATURES_CONT + FEATURES_CAT + [TARGET_COL]
    
    if include_fund_features:
        for feat in FEATURES_FUND:
            if feat in data.columns:
                edges.append((feat, "Recomendacao"))
                nodes.append(feat)
                
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(nodes)
    
    training_data = data[nodes].copy()
    for col in nodes:
        training_data[col] = training_data[col].astype("category")
        
    estimator = BayesianEstimator(model, training_data)
    for node in model.nodes():
        cpd = estimator.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        model.add_cpds(cpd)
        
    if not model.check_model():
        raise ValueError("Modelo inconsistente após estimação das CPDs")
        
    return model

def get_bn_probs(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> np.ndarray:
    """
    Realiza inferência no modelo BN e retorna as probabilidades da classe positiva.
    """
    infer = VariableElimination(model)
    probs = []
    
    # Identificar quais features de evidência usar (apenas as que o modelo conhece)
    evidence_cols = [n for n in model.nodes() if n != TARGET_COL]
    
    for _, row in df.iterrows():
        evidence = {col: row[col] for col in evidence_cols if col in df.columns}
        try:
            q = infer.query(variables=[TARGET_COL], evidence=evidence, show_progress=False)
            state_names = list(q.state_names[TARGET_COL])
            idx = state_names.index("sim")
            probs.append(q.values[idx])
        except Exception:
            probs.append(0.5)
            
    return np.array(probs)

def get_current_predictions(models: dict, calibrations: dict, current_discrete: pd.DataFrame, horizon_months: int) -> np.ndarray:
    """
    Obtém predições para o momento atual usando o modelo correspondente ao horizonte.
    """
    if horizon_months not in models:
        return np.full(len(current_discrete), 0.5)
        
    probs = get_bn_probs(models[horizon_months], current_discrete)
    if horizon_months in calibrations and calibrations[horizon_months] is not None:
        probs = calibrations[horizon_months].predict(probs)
        
    return probs

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calcula diversas métricas de performance do modelo.
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "brier_score_loss": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    try:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Erro ao computar métricas avançadas: {e}")
        
    return metrics

def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computa os dados para o reliability diagram (curva de calibração).
    """
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return mean_predicted_value, fraction_of_positives
    except Exception:
        return np.array([]), np.array([])

def train_models_walk_forward(raw_all: pd.DataFrame, edges: dict, include_fund_features: bool = False) -> Tuple[dict, dict, dict]:
    """
    Treina modelos com validação walk-forward expansiva e calibração via regressão isotônica.
    Retorna os modelos finais treinados em todos os dados, calibradores finais, e métricas por horizonte.
    """
    models = {}
    calibrations = {}
    metrics_by_horizon = {}
    
    dates_sorted = sorted(raw_all["Date"].unique())
    n = len(dates_sorted)
    
    if n < WALK_FORWARD_MIN_TRAIN_MONTHS + WALK_FORWARD_STEP_MONTHS:
        logger.warning("Dados históricos insuficientes para validação walk-forward.")
        # Fallback: sem validação cruzada, treina e calibra de forma simplificada
        return _fallback_train(raw_all, edges, include_fund_features)

    for h in HORIZONS_MONTHS:
        discrete_h = discretize_dataframe_for_horizon(raw_all, edges, h)
        if discrete_h.empty:
            continue
            
        all_y_true = []
        all_y_prob_uncalib = []
        all_y_prob_calib = []
        
        train_end_idx = WALK_FORWARD_MIN_TRAIN_MONTHS
        while train_end_idx < n:
            val_end_idx = min(train_end_idx + WALK_FORWARD_STEP_MONTHS, n)
            
            train_dates = dates_sorted[:train_end_idx]
            val_dates = dates_sorted[train_end_idx:val_end_idx]
            
            train_df = discrete_h[discrete_h["Date"].isin(train_dates)]
            val_df = discrete_h[discrete_h["Date"].isin(val_dates)]
            
            if not train_df.empty and not val_df.empty:
                try:
                    fold_model = build_bn_model(train_df, include_fund_features)
                    val_probs = get_bn_probs(fold_model, val_df)
                    y_val = val_df[TARGET_COL].map({"sim": 1, "nao": 0}).values
                    
                    calib_model = IsotonicRegression(out_of_bounds="clip")
                    calib_model.fit(val_probs, y_val)
                    val_probs_calib = calib_model.predict(val_probs)
                    
                    all_y_true.extend(y_val)
                    all_y_prob_uncalib.extend(val_probs)
                    all_y_prob_calib.extend(val_probs_calib)
                except Exception as e:
                    logger.warning(f"Erro fold horizonte {h}M (idx {train_end_idx}): {e}")
            
            train_end_idx = val_end_idx
            
        if all_y_true:
            metrics_by_horizon[h] = compute_metrics(np.array(all_y_true), np.array(all_y_prob_calib))
        else:
            metrics_by_horizon[h] = {}
            
        # Treinamento final com todo o dataset (com calibração via último fold ou overall)
        try:
            final_model = build_bn_model(discrete_h, include_fund_features)
            models[h] = final_model
            
            # Calibrador global
            if all_y_true:
                final_calib = IsotonicRegression(out_of_bounds="clip")
                final_calib.fit(all_y_prob_uncalib, all_y_true)
                calibrations[h] = final_calib
            else:
                calibrations[h] = None
        except Exception as e:
            logger.warning(f"Falha ao treinar modelo final para {h}M: {e}")
            
    return models, calibrations, metrics_by_horizon

def _fallback_train(raw_all: pd.DataFrame, edges: dict, include_fund_features: bool) -> Tuple[dict, dict, dict]:
    """
    Fallback quando não há dados suficientes para walk-forward.
    """
    models = {}
    calibrations = {}
    metrics_by_horizon = {}
    
    for h in HORIZONS_MONTHS:
        discrete_h = discretize_dataframe_for_horizon(raw_all, edges, h)
        if discrete_h.empty:
            continue
        try:
            model = build_bn_model(discrete_h, include_fund_features)
            probs = get_bn_probs(model, discrete_h)
            y_true = discrete_h[TARGET_COL].map({"sim": 1, "nao": 0}).values
            
            calib = IsotonicRegression(out_of_bounds="clip")
            calib.fit(probs, y_true)
            probs_calib = calib.predict(probs)
            
            models[h] = model
            calibrations[h] = calib
            metrics_by_horizon[h] = compute_metrics(y_true, probs_calib)
        except Exception as e:
            logger.warning(f"Erro fallback treinamento {h}M: {e}")
            
    return models, calibrations, metrics_by_horizon

def compute_training_stats(train_df: pd.DataFrame) -> dict:
    """
    Calcula média e desvio padrão das features contínuas do dataset de treino.
    """
    stats = {}
    for col in FEATURES_CONT:
        if col in train_df.columns:
            stats[col] = {
                "mean": train_df[col].mean(),
                "std": train_df[col].std()
            }
    return stats

def detect_drift(current_features: pd.DataFrame, training_stats: dict, threshold: float = 2.0) -> List[str]:
    """
    Compara features atuais com as estatísticas de treino para detectar drift.
    """
    warnings = []
    for col, stat in training_stats.items():
        if col in current_features.columns:
            curr_mean = current_features[col].mean()
            std = stat["std"]
            if std > 0:
                z_score = abs(curr_mean - stat["mean"]) / std
                if z_score > threshold:
                    warnings.append(f"Drift detectado em {col}: z-score de {z_score:.2f} excedeu {threshold}.")
    return warnings

def train_benchmark_lgbm(raw_all: pd.DataFrame, edges: dict, horizon_months: int) -> Tuple[Optional[object], dict]:
    """
    Treina um modelo LightGBM base (benchmark) com os mesmos dados.
    """
    if lgb is None:
        logger.warning("LightGBM não instalado. Benchmark ignorado.")
        return None, {}
        
    discrete_h = discretize_dataframe_for_horizon(raw_all, edges, horizon_months)
    if discrete_h.empty:
        return None, {}
        
    features = [c for c in discrete_h.columns if c not in ["Date", "Ticker", TARGET_COL]]
    
    # Prepara categóricas para o LightGBM
    X = discrete_h[features].copy()
    for col in X.columns:
        X[col] = X[col].astype("category")
        
    y = discrete_h[TARGET_COL].map({"sim": 1, "nao": 0}).values
    
    dates_sorted = sorted(discrete_h["Date"].unique())
    n = len(dates_sorted)
    
    if n < WALK_FORWARD_MIN_TRAIN_MONTHS + WALK_FORWARD_STEP_MONTHS:
        # Fallback train on all
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, probs)
        return model, metrics
        
    all_y_true = []
    all_y_prob = []
    
    train_end_idx = WALK_FORWARD_MIN_TRAIN_MONTHS
    while train_end_idx < n:
        val_end_idx = min(train_end_idx + WALK_FORWARD_STEP_MONTHS, n)
        
        train_dates = dates_sorted[:train_end_idx]
        val_dates = dates_sorted[train_end_idx:val_end_idx]
        
        train_idx = discrete_h["Date"].isin(train_dates)
        val_idx = discrete_h["Date"].isin(val_dates)
        
        if train_idx.sum() > 0 and val_idx.sum() > 0:
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            fold_model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
            fold_model.fit(X_train, y_train)
            
            val_probs = fold_model.predict_proba(X_val)[:, 1]
            all_y_true.extend(y_val)
            all_y_prob.extend(val_probs)
            
        train_end_idx = val_end_idx
        
    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob)) if all_y_true else {}
    
    final_model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
    final_model.fit(X, y)
    
    return final_model, metrics

def get_bn_dag_edges(model: DiscreteBayesianNetwork) -> List[Tuple[str, str]]:
    """
    Retorna as arestas (edges) do modelo BN para visualização do DAG.
    """
    return list(model.edges())

def get_cpd_table(model: DiscreteBayesianNetwork, node: str) -> pd.DataFrame:
    """
    Retorna a tabela de Probabilidades Condicionais (CPD) de um nó como um DataFrame legível.
    """
    cpd = model.get_cpds(node)
    if cpd is None:
        return pd.DataFrame()
        
    # Extrai state names
    state_names = cpd.state_names
    
    df = pd.DataFrame(
        cpd.get_values(),
        index=state_names[node] if node in state_names else range(cpd.variable_card)
    )
    
    # Se houver variáveis condicionantes, definir as colunas
    if cpd.variables[1:]:
        import itertools
        cols = []
        for var in cpd.variables[1:]:
            cols.append(state_names[var] if var in state_names else range(cpd.get_cardinality([var])[var]))
        
        col_tuples = list(itertools.product(*cols))
        df.columns = pd.MultiIndex.from_tuples(col_tuples, names=cpd.variables[1:])
    else:
        df.columns = ["Probabilidade"]
        
    return df
