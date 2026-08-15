"""
ml_pipeline.py  -  FinPulse ML Pipeline
========================================
Precomputes all ML artifacts offline. Run with: python src/ml_pipeline.py

Pipeline steps:
  1. Feature engineering from raw daily balance data
  2. Risk classification (Logistic Regression baseline + Random Forest)
  3. K-Means clustering validation of predefined segments
  4. Forecasting model comparison (Naive vs 7-day SMA vs Holt-Winters)
  5. Statistical testing (ANOVA + Spearman correlation)

All outputs saved to model_results/ as static JSON/CSV.
The API serves these files  -  nothing here runs live on the server.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from fhs_calculator import compute_fhs, get_risk_label
from customer_generator import SEGMENTS

MODEL_DIR = Path("model_results")
DATA_DIR = Path("data")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Feature Engineering
# ═════════════════════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 730 days of daily balance data into per-customer behavioral
    features. Each feature captures a different dimension of financial health.

    Features (11 total, excluding fhs_mean to avoid data leakage):
      - bal_mean:           Average daily balance
      - bal_std:            Balance standard deviation (volatility)
      - bal_trend:          Linear regression slope (direction)
      - bal_cv:             Coefficient of variation (normalized volatility)
      - fhs_std:            FHS score volatility over rolling windows
      - income_regularity:  CV of monthly balance deltas (income predictability)
      - vel_mean:           Mean absolute daily balance change
      - vel_max:            Maximum absolute daily balance change
      - pct_negative_days:  Fraction of days with balance < 0
      - max_drawdown:       Largest peak-to-trough decline
      - runway_mean:        Estimated days of expenses balance can cover

    Target:
      - risk_tier:          High (<60 FHS) / Medium (60-75) / Low (>=75)

    NOTE: fhs_mean is deliberately EXCLUDED from features. risk_tier is
    defined as bucket(fhs_mean, [60, 75]), so including fhs_mean would be
    data leakage  -  the model could trivially re-derive the threshold rule
    instead of learning meaningful patterns.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Feature Engineering")
    print("=" * 70)

    df["date"] = pd.to_datetime(df["date"])
    features = []

    customers = df.groupby(["customer_id", "segment"])
    total = len(customers)
    done = 0

    for (cid, segment), cust_df in customers:
        cust_df = cust_df.sort_values("date")
        balances = cust_df["balance"].values
        dates = cust_df["date"].values

        # Balance features
        bal_mean = float(np.mean(balances))
        bal_std = float(np.std(balances))
        bal_cv = bal_std / (abs(bal_mean) + 1e-9)

        # Trend: slope of linear regression
        x = np.arange(len(balances))
        bal_trend = float(np.polyfit(x, balances, 1)[0])

        # Daily changes (velocity)
        daily_changes = np.abs(np.diff(balances))
        vel_mean = float(np.mean(daily_changes))
        vel_max = float(np.max(daily_changes))

        # Negative balance exposure
        pct_negative_days = float(np.mean(balances < 0))

        # Maximum drawdown
        cummax = np.maximum.accumulate(balances)
        drawdowns = cummax - balances
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Income regularity: CV of monthly balance changes
        bal_series = pd.Series(balances, index=pd.to_datetime(dates))
        monthly_deltas = bal_series.resample("ME").last().diff().dropna()
        if len(monthly_deltas) > 1:
            income_regularity = float(monthly_deltas.std() / (abs(monthly_deltas.mean()) + 1e-9))
        else:
            income_regularity = 1.0

        # Runway: how many days of avg daily expenses the balance covers
        neg_diffs = np.diff(balances)[np.diff(balances) < 0]
        avg_daily_expense = float(np.mean(np.abs(neg_diffs))) if len(neg_diffs) > 0 else 1.0
        runway_mean = bal_mean / (avg_daily_expense + 1e-9)
        runway_mean = float(np.clip(runway_mean, 0, 365))

        # FHS computation for target variable and fhs_std
        # Compute FHS over multiple rolling windows to get volatility
        window_size = 180
        step_size = 30
        fhs_values = []
        for start_idx in range(0, max(1, len(balances) - window_size), step_size):
            window = bal_series.iloc[start_idx:start_idx + window_size]
            if len(window) >= 90:
                fhs_values.append(compute_fhs(window))

        if len(fhs_values) == 0:
            fhs_values = [compute_fhs(bal_series)]

        fhs_mean_val = float(np.mean(fhs_values))
        fhs_std = float(np.std(fhs_values)) if len(fhs_values) > 1 else 0.0

        # Risk tier (classification target)
        # These thresholds (<60/60-75/>=75) differ from the dashboard display
        # thresholds (<35/35-59/>=60) intentionally. The classifier predicts
        # financial distress severity using stricter bands, catching at-risk
        # customers before they cross the operational RED line.
        if fhs_mean_val < 60:
            risk_tier = "High"
        elif fhs_mean_val < 75:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        features.append({
            "customer_id": cid,
            "segment": segment,
            "bal_mean": round(bal_mean, 2),
            "bal_std": round(bal_std, 2),
            "bal_trend": round(bal_trend, 4),
            "bal_cv": round(bal_cv, 4),
            "fhs_std": round(fhs_std, 4),
            "income_regularity": round(income_regularity, 4),
            "vel_mean": round(vel_mean, 2),
            "vel_max": round(vel_max, 2),
            "pct_negative_days": round(pct_negative_days, 4),
            "max_drawdown": round(max_drawdown, 2),
            "runway_mean": round(runway_mean, 2),
            "risk_tier": risk_tier,
            # Store fhs_mean for reference/validation  -  NOT used as model input
            "_fhs_mean": round(fhs_mean_val, 2),
        })

        done += 1
        if done % 200 == 0:
            print(f"    Processed {done}/{total} customers...")

    features_df = pd.DataFrame(features)
    features_df.to_csv(MODEL_DIR / "features.csv", index=False)
    print(f"  [OK] features.csv  -  {len(features_df)} customers  x  {len(features_df.columns)} columns")
    print(f"    Risk tier distribution: {features_df['risk_tier'].value_counts().to_dict()}")
    print(f"    fhs_mean excluded from model inputs (data leakage prevention)")

    return features_df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Risk Classification
# ═════════════════════════════════════════════════════════════════════════════

def _train_classifiers(features_df: pd.DataFrame) -> dict:
    """
    Train Logistic Regression (baseline) and Random Forest classifiers
    for risk tier prediction. Uses stratified 80/20 split.

    fhs_mean is excluded from inputs  -  see Step 1 docstring for why.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, auc
    )

    print("\n" + "=" * 70)
    print("STEP 2: Risk Classification (LogReg + Random Forest)")
    print("=" * 70)

    # Feature columns  -  deliberately excluding fhs_mean and metadata
    feature_cols = [
        "bal_mean", "bal_std", "bal_trend", "bal_cv",
        "fhs_std", "income_regularity",
        "vel_mean", "vel_max",
        "pct_negative_days", "max_drawdown", "runway_mean"
    ]

    X = features_df[feature_cols].values
    y = features_df["risk_tier"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features for LogReg (RF doesn't need it but doesn't hurt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression (Baseline) ---
    print("\n  Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_report = classification_report(y_test, y_pred_lr, target_names=class_names, output_dict=True)
    lr_accuracy = float(lr_report["accuracy"])
    print(f"    Accuracy: {lr_accuracy:.3f}")
    results["logistic_regression"] = {
        "accuracy": round(lr_accuracy, 4),
        "report": {k: {m: round(v, 4) for m, v in vals.items()} if isinstance(vals, dict) else round(vals, 4) for k, vals in lr_report.items()},
    }

    # --- Random Forest ---
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF uses unscaled features
    y_pred_rf = rf.predict(X_test)
    rf_report = classification_report(y_test, y_pred_rf, target_names=class_names, output_dict=True)
    rf_accuracy = float(rf_report["accuracy"])
    print(f"    Accuracy: {rf_accuracy:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_rf).tolist()

    # Feature importance
    importances = rf.feature_importances_
    importance_dict = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(feature_cols, importances), key=lambda x: x[1], reverse=True
        )
    }
    print(f"    Top 3 features: {list(importance_dict.items())[:3]}")

    results["random_forest"] = {
        "accuracy": round(rf_accuracy, 4),
        "report": {k: {m: round(v, 4) for m, v in vals.items()} if isinstance(vals, dict) else round(vals, 4) for k, vals in rf_report.items()},
    }

    # ROC curves (one-vs-rest for each class)
    roc_data = {}
    if hasattr(rf, "predict_proba"):
        y_proba = rf.predict_proba(X_test)
        for i, cls in enumerate(class_names):
            y_binary = (y_test == i).astype(int)
            if y_proba.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                roc_auc = float(auc(fpr, tpr))
                roc_data[cls] = {
                    "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//50)]],
                    "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//50)]],
                    "auc": round(roc_auc, 4),
                }

    # Save all classification results
    classification_results = {
        "feature_columns": feature_cols,
        "class_names": class_names,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "models": results,
        "confusion_matrix": cm,
        "feature_importance": importance_dict,
        "roc_data": roc_data,
        "notes": {
            "leakage_prevention": "fhs_mean excluded from inputs because risk_tier = bucket(fhs_mean, [35, 60]). "
                                  "Including it would let the model trivially re-derive the threshold rule.",
            "structural_note": "The remaining features (bal_trend, vel_mean, etc.) are decomposed components "
                              "of the signal that built the FHS composite score. The model is largely learning "
                              "to reconstruct a known weighted formula from its components, which is expected "
                              "for a synthetic-data proof of concept."
        }
    }

    with open(MODEL_DIR / "classification_report.json", "w") as f:
        json.dump(classification_results, f, indent=2)

    print(f"\n  [OK] classification_report.json saved")
    print(f"    LogReg accuracy: {lr_accuracy:.3f} | RF accuracy: {rf_accuracy:.3f}")

    if rf_accuracy > 0.97:
        print(f"    [!] WARNING: RF accuracy ({rf_accuracy:.3f}) is suspiciously high.")
        print(f"      Verify fhs_mean is not in feature_cols: {feature_cols}")

    return classification_results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Clustering Validation
# ═════════════════════════════════════════════════════════════════════════════

def _run_clustering(features_df: pd.DataFrame) -> dict:
    """
    K-Means clustering to validate whether predefined segments correspond
    to natural behavioral clusters in the feature space.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    print("\n" + "=" * 70)
    print("STEP 3: Clustering Validation (K-Means)")
    print("=" * 70)

    feature_cols = [
        "bal_mean", "bal_std", "bal_trend", "bal_cv",
        "fhs_std", "income_regularity",
        "vel_mean", "vel_max",
        "pct_negative_days", "max_drawdown", "runway_mean"
    ]

    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method + silhouette scores
    k_range = range(2, 11)
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_scaled, labels))
        silhouette_scores_list.append(sil)
        print(f"    k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}")

    best_k = list(k_range)[np.argmax(silhouette_scores_list)]
    print(f"\n  Best k by silhouette: {best_k} (score={max(silhouette_scores_list):.3f})")

    # Final clustering with best k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Cross-tabulation: cluster vs actual segment
    crosstab = pd.crosstab(
        features_df["segment"], cluster_labels, margins=False
    )

    clustering_results = {
        "k_range": list(k_range),
        "inertias": [round(x, 2) for x in inertias],
        "silhouette_scores": [round(x, 4) for x in silhouette_scores_list],
        "best_k": int(best_k),
        "best_silhouette": round(float(max(silhouette_scores_list)), 4),
        "pca_variance_explained": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "pca_data": [
            {
                "x": round(float(X_pca[i, 0]), 4),
                "y": round(float(X_pca[i, 1]), 4),
                "cluster": int(cluster_labels[i]),
                "segment": features_df["segment"].iloc[i],
            }
            for i in range(0, len(X_pca), max(1, len(X_pca) // 200))  # Subsample for JSON size
        ],
        "crosstab": {
            "segments": crosstab.index.tolist(),
            "clusters": [int(c) for c in crosstab.columns.tolist()],
            "values": crosstab.values.tolist(),
        },
    }

    with open(MODEL_DIR / "clustering.json", "w") as f:
        json.dump(clustering_results, f, indent=2)

    print(f"  [OK] clustering.json saved")
    print(f"    PCA variance explained: {[round(float(x), 4) for x in pca.explained_variance_ratio_]}")

    return clustering_results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Forecasting Model Comparison
# ═════════════════════════════════════════════════════════════════════════════

def _compare_forecasters(df: pd.DataFrame) -> dict:
    """
    Compare 3 forecasting models on each segment:
      - Naive (carry forward last value)
      - 7-day SMA (rolling average extrapolation)
      - Holt-Winters (exponential smoothing with trend)

    Train on first 80% of historical FHS series, test on last 20%.
    Reports RMSE, MAE, MAPE per model x segment.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from forecaster import build_daily_fhs_series

    print("\n" + "=" * 70)
    print("STEP 4: Forecasting Model Comparison")
    print("=" * 70)

    segments = df["segment"].unique()
    all_metrics = {}
    summary = {"segments": [], "naive": [], "sma": [], "holt_winters": []}

    for seg in segments:
        print(f"\n  Segment: {seg}")
        series = build_daily_fhs_series(df, seg)
        if len(series) < 10:
            print(f"    Skipped  -  insufficient data ({len(series)} points)")
            continue

        # 80/20 train/test split
        split_idx = int(len(series) * 0.8)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        n_test = len(test)

        if n_test < 3:
            print(f"    Skipped  -  test set too small ({n_test} points)")
            continue

        actuals = test.values
        metrics = {}

        # --- Naive: carry forward last training value ---
        naive_pred = np.full(n_test, train.iloc[-1])
        metrics["naive"] = _compute_metrics(actuals, naive_pred)

        # --- 7-day SMA: rolling average of last 7 days projected forward ---
        sma_window = min(7, len(train))
        sma_pred = np.full(n_test, train.iloc[-sma_window:].mean())
        metrics["sma"] = _compute_metrics(actuals, sma_pred)

        # --- Holt-Winters (exponential smoothing with additive trend) ---
        try:
            hw_model = ExponentialSmoothing(
                train, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            hw_pred = hw_model.forecast(n_test).values
        except Exception:
            try:
                ses_model = SimpleExpSmoothing(
                    train, initialization_method="estimated"
                ).fit(optimized=True)
                hw_pred = ses_model.forecast(n_test).values
            except Exception:
                hw_pred = naive_pred  # Worst case fallback

        metrics["holt_winters"] = _compute_metrics(actuals, hw_pred)

        all_metrics[seg] = metrics
        summary["segments"].append(seg)
        summary["naive"].append(metrics["naive"]["rmse"])
        summary["sma"].append(metrics["sma"]["rmse"])
        summary["holt_winters"].append(metrics["holt_winters"]["rmse"])

        # Print comparison
        for model_name, m in metrics.items():
            print(f"    {model_name:15s}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  MAPE={m['mape']:.1f}%")

    # Compute how often Holt-Winters beats baselines
    hw_wins = sum(
        1 for seg in all_metrics
        if all_metrics[seg]["holt_winters"]["rmse"] < all_metrics[seg]["naive"]["rmse"]
    )
    total = len(all_metrics)

    # Average improvement
    avg_improvement = 0
    if total > 0:
        improvements = []
        for seg in all_metrics:
            naive_rmse = all_metrics[seg]["naive"]["rmse"]
            hw_rmse = all_metrics[seg]["holt_winters"]["rmse"]
            if naive_rmse > 0:
                improvements.append((naive_rmse - hw_rmse) / naive_rmse * 100)
        avg_improvement = float(np.mean(improvements)) if improvements else 0

    forecast_results = {
        "per_segment": all_metrics,
        "summary": {
            "hw_wins_vs_naive": f"{hw_wins}/{total}",
            "avg_rmse_improvement_pct": round(avg_improvement, 1),
            "models_compared": ["naive", "sma", "holt_winters"],
            "evaluation_method": "80/20 train/test split on historical FHS series",
        },
    }

    with open(MODEL_DIR / "forecast_metrics.json", "w") as f:
        json.dump(forecast_results, f, indent=2)

    print(f"\n  [OK] forecast_metrics.json saved")
    print(f"    Holt-Winters beats naive on {hw_wins}/{total} segments")
    print(f"    Average RMSE improvement: {avg_improvement:.1f}%")

    return forecast_results


def _compute_metrics(actuals: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE for a forecast vs actuals."""
    residuals = actuals - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    # MAPE  -  avoid division by zero
    mask = actuals != 0
    if mask.any():
        mape = float(np.mean(np.abs(residuals[mask] / actuals[mask])) * 100)
    else:
        mape = 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Statistical Testing
# ═════════════════════════════════════════════════════════════════════════════

def _run_statistical_tests(features_df: pd.DataFrame) -> dict:
    """
    Test 1: One-way ANOVA  -  does mean FHS differ across segments?
            Includes eta-squared (effect size) because with n=1000,
            p < 0.05 is nearly guaranteed regardless of actual difference size.

    Test 2: Spearman correlation  -  runway_mean vs _fhs_mean
            Uses runway_mean (not balance_trend) because balance_trend is
            40% of the FHS formula by weight  -  correlating them would be
            circular (validating your own arithmetic).
    """
    from scipy import stats

    print("\n" + "=" * 70)
    print("STEP 5: Statistical Testing")
    print("=" * 70)

    results = {}

    # --- Test 1: ANOVA ---
    print("\n  Test 1: One-way ANOVA (FHS across segments)")
    groups = [
        group["_fhs_mean"].values
        for _, group in features_df.groupby("segment")
    ]
    f_stat, p_value = stats.f_oneway(*groups)

    # Eta-squared: SS-between / SS-total
    grand_mean = features_df["_fhs_mean"].mean()
    ss_total = float(np.sum((features_df["_fhs_mean"] - grand_mean) ** 2))
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
    )
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    results["anova"] = {
        "test": "One-way ANOVA",
        "question": "Does mean FHS differ significantly across the 8 segments?",
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(f"{p_value:.2e}") if p_value < 0.01 else round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "eta_squared": round(eta_squared, 4),
        "eta_squared_interpretation": (
            f"Segment membership explains {eta_squared*100:.1f}% of the variance in FHS. "
            f"{'This is a large effect.' if eta_squared > 0.14 else 'This is a medium effect.' if eta_squared > 0.06 else 'This is a small effect.'}"
        ),
        "note": "With n=1000 across 8 designed-to-differ segments, p < 0.05 is expected. "
                "Eta-squared shows the actual magnitude of the difference."
    }

    print(f"    F={f_stat:.2f}, p={p_value:.2e}, eta_sq={eta_squared:.4f}")
    print(f"    {results['anova']['eta_squared_interpretation']}")

    # --- Test 2: Spearman Correlation ---
    print("\n  Test 2: Spearman Correlation (runway_mean vs FHS)")
    rho, p_val_spearman = stats.spearmanr(
        features_df["runway_mean"], features_df["_fhs_mean"]
    )

    results["spearman"] = {
        "test": "Spearman Rank Correlation",
        "question": "Is there a monotonic relationship between liquidity runway and FHS?",
        "variables": ["runway_mean", "fhs_mean"],
        "rho": round(float(rho), 4),
        "p_value": float(f"{p_val_spearman:.2e}") if p_val_spearman < 0.01 else round(float(p_val_spearman), 6),
        "significant": bool(p_val_spearman < 0.05),
        "interpretation": (
            f"{'Strong' if abs(rho) > 0.7 else 'Moderate' if abs(rho) > 0.4 else 'Weak'} "
            f"{'positive' if rho > 0 else 'negative'} monotonic relationship (rho={rho:.3f}). "
            f"{'FHS captures liquidity risk as intended.' if abs(rho) > 0.3 else 'Weak association  -  FHS may not fully capture liquidity risk.'}"
        ),
        "note": "runway_mean is computed independently of the FHS formula (not one of the 4 weighted sub-scores), "
                "so this is a genuine validation, not circular."
    }

    print(f"    rho={rho:.4f}, p={p_val_spearman:.2e}")
    print(f"    {results['spearman']['interpretation']}")

    # --- Per-segment FHS summary (for context) ---
    seg_summary = {}
    for seg, group in features_df.groupby("segment"):
        seg_summary[seg] = {
            "mean_fhs": round(float(group["_fhs_mean"].mean()), 2),
            "std_fhs": round(float(group["_fhs_mean"].std()), 2),
            "n_customers": int(len(group)),
            "pct_high_risk": round(float((group["risk_tier"] == "High").mean() * 100), 1),
            "pct_medium_risk": round(float((group["risk_tier"] == "Medium").mean() * 100), 1),
            "pct_low_risk": round(float((group["risk_tier"] == "Low").mean() * 100), 1),
        }
    results["segment_summary"] = seg_summary

    with open(MODEL_DIR / "statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [OK] statistical_tests.json saved")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Execute the full ML pipeline."""
    print("=" * 70)
    print("FinPulse ML Pipeline")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = DATA_DIR / "historical.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run customer_generator.py first.")
        sys.exit(1)

    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows, {df['customer_id'].nunique()} customers, {df['segment'].nunique()} segments")

    # Execute pipeline steps
    features_df = _engineer_features(df)
    classification = _train_classifiers(features_df)
    clustering = _run_clustering(features_df)
    forecasting = _compare_forecasters(df)
    statistics = _run_statistical_tests(features_df)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE  -  All results saved to model_results/")
    print("=" * 70)

    rf_acc = classification["models"]["random_forest"]["accuracy"]
    lr_acc = classification["models"]["logistic_regression"]["accuracy"]
    best_sil = clustering["best_silhouette"]
    hw_wins = forecasting["summary"]["hw_wins_vs_naive"]
    eta_sq = statistics["anova"]["eta_squared"]

    print(f"\n  Classification:  RF={rf_acc:.3f}  LR={lr_acc:.3f}")
    print(f"  Clustering:      best_k={clustering['best_k']}, silhouette={best_sil:.3f}")
    print(f"  Forecasting:     HW beats naive on {hw_wins} segments")
    print(f"  ANOVA:           eta_sq={eta_sq:.4f} ({statistics['anova']['eta_squared_interpretation']})")
    print(f"  Spearman:        rho={statistics['spearman']['rho']:.3f}")

    if rf_acc > 0.97:
        print(f"\n  [!] RF accuracy is {rf_acc:.3f}  -  review feature list for possible leakage")


if __name__ == "__main__":
    run_pipeline()

