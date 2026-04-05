from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import skew
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


@dataclass
class ColumnRoles:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    text: List[str]
    id_like: List[str]
    dropped: List[str]


class AutomatedML:
    """
    A more robust tabular ML engine for uploaded CSV datasets.
    Supports both classification and regression with smarter preprocessing,
    cross-validated model comparison, and frontend-friendly outputs.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target: Optional[str] = None
        self.task_type: Optional[str] = None
        self.pipeline: Optional[ColumnTransformer] = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.column_roles: Optional[ColumnRoles] = None
        self.dataset_warnings: List[str] = []
        self.recommendations: List[str] = []
        self.target_summary: Dict[str, Any] = {}
        self.training_profile: Dict[str, Any] = {}
        self.use_log_target: bool = False

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_transformed = None
        self.X_test_transformed = None

    # -----------------------------
    # Public API
    # -----------------------------
    def set_target(self, target_column: str):
        if target_column not in self.df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset.")
        self.target = target_column
        self.task_type = self._infer_task_type(self.df[target_column])
        self.target_summary = self._build_target_summary(self.df[target_column])
        self.use_log_target = False

    def set_training_options(self, use_log_target: bool = False):
        should_use = bool(use_log_target)
        if self.task_type != "regression":
            self.use_log_target = False
            return
        if should_use and not self.target_summary.get("log_transform_supported", False):
            raise ValueError("Log target transform is only available for non-negative regression targets.")
        self.use_log_target = should_use

    def get_dataset_summary(self) -> Dict[str, Any]:
        return {
            "rows": int(len(self.df)),
            "columns": int(self.df.shape[1]),
            "column_names": self.df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in self.df.dtypes.items()},
            "missing_values": self.df.isna().sum().to_dict(),
            "target": self.target,
            "task_type": self.task_type,
        }

    def run_eda(self) -> Dict[str, Any]:
        if self.target is None:
            raise ValueError("Set target before running EDA.")

        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        target_series = self.df[self.target]

        correlations = {}
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr(numeric_only=True).fillna(0)
            correlations["matrix"] = corr_matrix.round(4).to_dict()
            if self.target in corr_matrix.columns:
                target_corr = corr_matrix[self.target].drop(labels=[self.target], errors="ignore")
                correlations["target_relationships"] = (
                    target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
                    .head(15)
                    .round(4)
                    .to_dict()
                )

        missing_summary = (
            self.df.isna().mean().sort_values(ascending=False).mul(100).round(2).to_dict()
        )

        eda = {
            "correlations": correlations,
            "missing_summary_pct": missing_summary,
            "numeric_columns": numeric_df.columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
            "target_distribution": {},
            "target_histogram": {},
            "suggestions": [],
            "warnings": [],
        }

        if self.task_type == "classification":
            eda["target_distribution"] = target_series.astype(str).value_counts(dropna=False).to_dict()
            class_balance = target_series.astype(str).value_counts(normalize=True, dropna=False)
            if len(class_balance) >= 2 and class_balance.max() > 0.85:
                eda["suggestions"].append(
                    "Target appears imbalanced. Consider stratified validation, class weights, or threshold tuning."
                )
                eda["warnings"].append({"type": "imbalance", "severity": "warning", "message": "Target variable is imbalanced — the majority class represents over 85% of the data."})
        else:
            non_null = target_series.dropna()
            if not non_null.empty:
                hist_counts, hist_bins = np.histogram(non_null, bins=min(20, max(5, int(np.sqrt(len(non_null))))))
                eda["target_histogram"] = {
                    "bins": hist_bins.tolist(),
                    "counts": hist_counts.tolist(),
                }
                p99 = float(non_null.quantile(0.99))
                display_vals = non_null[non_null <= p99]
                if len(display_vals) >= 10 and len(display_vals) < len(non_null):
                    display_counts, display_bins = np.histogram(
                        display_vals,
                        bins=min(20, max(5, int(np.sqrt(len(display_vals))))),
                    )
                    eda["target_histogram_display"] = {
                        "bins": display_bins.tolist(),
                        "counts": display_counts.tolist(),
                        "clip_percentile": 99,
                        "clip_value": p99,
                        "excluded_count": int((non_null > p99).sum()),
                    }
                # Log-transformed histogram for skewed targets
                try:
                    sk_val = float(skew(non_null, nan_policy="omit"))
                    if abs(sk_val) > 1.5 and float(non_null.min()) >= 0:
                        log_vals = np.log1p(non_null)
                        if len(log_vals) > 10:
                            log_counts, log_bins = np.histogram(log_vals, bins=min(25, max(5, int(np.sqrt(len(log_vals))))))
                            eda["target_histogram_log"] = {
                                "bins": log_bins.tolist(),
                                "counts": log_counts.tolist(),
                            }
                except Exception:
                    pass
                q1 = float(non_null.quantile(0.25))
                median = float(non_null.median())
                q3 = float(non_null.quantile(0.75))
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                within_fence = non_null[(non_null >= lower_fence) & (non_null <= upper_fence)]
                eda["target_boxplot"] = {
                    "min": float(non_null.min()),
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": float(non_null.max()),
                    "p99": p99,
                    "lower_whisker": float(within_fence.min()) if not within_fence.empty else float(non_null.min()),
                    "upper_whisker": float(within_fence.max()) if not within_fence.empty else float(non_null.max()),
                    "outlier_count": int(((non_null < lower_fence) | (non_null > upper_fence)).sum()),
                }
                eda["suggestions"].append(
                    "For regression, review residuals and predicted-vs-actual plots in addition to RMSE and R²."
                )
                # Skewness warning
                try:
                    sk = float(skew(non_null, nan_policy="omit"))
                    if abs(sk) > 2.0 and float(non_null.min()) >= 0:
                        eda["warnings"].append({"type": "skewness", "severity": "warning", "message": f"Target is highly skewed (skewness = {sk:.2f}). A log transform may improve model performance."})
                except Exception:
                    pass
                # Suspicious low-value warning
                target_min = float(non_null.min())
                target_median = float(non_null.median())
                if target_min >= 0 and target_median > 0 and target_min < target_median * 0.001:
                    eda["warnings"].append({"type": "low_min", "severity": "warning", "message": f"Very low target values detected (min = {target_min:,.0f}). Review for invalid, placeholder, or anomalous records."})

        # Data quality warnings
        high_miss_cols = [c for c, pct in missing_summary.items() if pct > 30]
        if high_miss_cols:
            eda["warnings"].append({"type": "missingness", "severity": "warning", "message": f"{len(high_miss_cols)} column(s) have more than 30% missing data: {', '.join(high_miss_cols[:5])}"})

        dup_count = int(self.df.duplicated().sum())
        if dup_count > 0:
            eda["warnings"].append({"type": "duplicates", "severity": "info", "message": f"{dup_count} duplicate rows detected in the dataset."})

        # Check for ID-like columns
        id_cols = [c for c in self.df.columns if self._looks_like_id(self.df[c], c) and c != self.target]
        if id_cols:
            eda["warnings"].append({"type": "id_columns", "severity": "info", "message": f"Likely ID columns detected and will be excluded: {', '.join(id_cols[:5])}"})

        if self.column_roles is not None:
            if self.column_roles.text:
                eda["suggestions"].append(
                    f"High-cardinality text columns were excluded by default: {', '.join(self.column_roles.text[:5])}."
                )
            if self.column_roles.id_like:
                eda["suggestions"].append(
                    f"Likely ID columns were excluded: {', '.join(self.column_roles.id_like[:5])}."
                )

        return eda

    def train_all_models(self):
        if self.target is None:
            raise ValueError("Set target before training models.")

        self.preprocess_data()
        cv = self._build_cv()
        model_defs = self._get_candidate_models()

        self.results = {}
        self.models = {}

        y_train_eval = self._encode_target_if_needed(self.y_train, fit_encoder=True)

        scoring = self._get_scoring_dict()

        for name, model in model_defs.items():
            try:
                model_for_cv = self._wrap_model_for_training(clone(model))

                cv_result = cross_validate(
                    model_for_cv,
                    self.X_train_transformed,
                    y_train_eval,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=min(4, os.cpu_count() or 1),
                    error_score=np.nan,
                    return_train_score=False,
                )

                summary = self._summarize_cv_metrics(cv_result)

                fitted_model = self._wrap_model_for_training(clone(model))
                fitted_model.fit(self.X_train_transformed, y_train_eval)
                self.models[name] = fitted_model

                test_metrics, extras = self._evaluate_on_holdout(fitted_model, self.X_test_transformed, self.y_test)

                self.results[name] = {
                    "cv_metrics": summary,
                    "test_metrics": test_metrics,
                    "extras": extras,
                }
            except Exception as e:
                self.results[name] = {
                    "cv_metrics": {},
                    "test_metrics": {},
                    "extras": {},
                    "error": str(e),
                }

        self.best_model_name = self._pick_best_model()
        if self.best_model_name is not None:
            self.best_model = self.models.get(self.best_model_name)

        self.recommendations = self._build_recommendations()

    def get_model_comparison(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name, payload in self.results.items():
            row = {"model": name}
            row.update(payload.get("cv_metrics", {}))
            row.update({f"holdout_{k}": v for k, v in payload.get("test_metrics", {}).items()})
            if payload.get("error"):
                row["error"] = payload["error"]
            rows.append(row)

        if self.task_type == "classification":
            rows.sort(key=lambda x: (x.get("cv_accuracy_mean", -np.inf), x.get("cv_roc_auc_mean", -np.inf)), reverse=True)
        else:
            rows.sort(
                key=lambda x: (
                    x.get("cv_r2_mean", -np.inf),
                    -x.get("cv_rmse_mean", np.inf),
                    x.get("holdout_r2", -np.inf),
                ),
                reverse=True,
            )
        return rows

    def get_best_model_info(self) -> Dict[str, Any]:
        if not self.best_model_name:
            return {}

        result = self.results[self.best_model_name]
        return {
            "best_model": self.best_model_name,
            "task_type": self.task_type,
            "cv_metrics": result.get("cv_metrics", {}),
            "holdout_metrics": result.get("test_metrics", {}),
            "target_transform": "log1p" if self.use_log_target else "none",
            "reason": self._build_best_model_reason(),
            "warnings": self.dataset_warnings,
            "recommendations": self.recommendations,
        }

    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        if self.best_model is None:
            return {}

        model = self._unwrap_model(self.best_model)
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importance = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importance is None or len(importance) != len(self.feature_names):
            return {}

        ranked = sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)[:top_n]
        return {k: float(v) for k, v in ranked}

    def get_source_feature_importance(self) -> Dict[str, float]:
        if self.best_model is None or self.X_train is None:
            return {}

        model = self._unwrap_model(self.best_model)
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importance = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importance is None or len(importance) != len(self.feature_names):
            return {}

        source_columns = sorted(self.X_train.columns.tolist(), key=len, reverse=True)
        grouped: Dict[str, float] = {col: 0.0 for col in self.X_train.columns.tolist()}

        for feature_name, score in zip(self.feature_names, importance):
            source = next(
                (col for col in source_columns if feature_name == col or feature_name.startswith(f"{col}_")),
                feature_name,
            )
            grouped[source] = grouped.get(source, 0.0) + float(score)

        return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))

    def get_prediction_schema(self, top_n: int = 10, max_dropdown_options: int = 12) -> Dict[str, Any]:
        if self.best_model is None or self.X_train is None or self.column_roles is None:
            return {}

        used_columns = self.X_train.columns.tolist()
        source_importance = self.get_source_feature_importance()
        ranked_columns = sorted(
            used_columns,
            key=lambda col: (source_importance.get(col, 0.0), -used_columns.index(col)),
            reverse=True,
        )
        key_inputs = ranked_columns[: min(top_n, len(ranked_columns))]

        fields: List[Dict[str, Any]] = []
        for rank, col in enumerate(ranked_columns, start=1):
            series = self.X_train[col]
            non_null = series.dropna()
            is_numeric = col in self.column_roles.numeric
            default_value = None if non_null.empty else non_null.median() if is_numeric else non_null.mode().iloc[0]
            value_type = "number" if pd.api.types.is_numeric_dtype(series) else "boolean" if pd.api.types.is_bool_dtype(series) else "string"
            unique_count = int(non_null.nunique(dropna=True))

            if is_numeric:
                kind = "number"
                options: List[Dict[str, Any]] = []
            else:
                option_values = non_null.value_counts(dropna=True).head(max_dropdown_options).index.tolist()
                if 0 < unique_count <= max_dropdown_options:
                    kind = "select"
                    options = [
                        {"label": str(value), "value": self._to_json_scalar(value)}
                        for value in option_values
                    ]
                else:
                    kind = "text"
                    options = []

            fields.append({
                "name": col,
                "label": col.replace("_", " "),
                "kind": kind,
                "value_type": value_type,
                "default_value": self._to_json_scalar(default_value),
                "default_label": "median" if is_numeric else "most frequent",
                "options": options,
                "is_key": col in key_inputs,
                "importance": float(source_importance.get(col, 0.0)),
                "rank": rank,
                "is_optional": True,
            })

        return {
            "used_columns": used_columns,
            "key_inputs": key_inputs,
            "excluded_columns": self.column_roles.dropped,
            "fields": fields,
            "supports_partial_input": True,
            "notes": [
                "Only model-used inputs are shown.",
                "Fields left blank will fall back to the training pipeline defaults or imputers.",
            ],
        }

    def get_evaluation_charts(self) -> Dict[str, Any]:
        charts: Dict[str, Any] = {}
        if self.best_model is None:
            return charts

        y_true_raw = self.y_test
        y_true = self._encode_target_if_needed(y_true_raw, fit_encoder=False) if self.task_type == "classification" else y_true_raw
        y_pred = self.best_model.predict(self.X_test_transformed)

        if self.task_type == "classification":
            try:
                if hasattr(self.best_model, "predict_proba"):
                    prob = self.best_model.predict_proba(self.X_test_transformed)
                    if prob.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_true, prob[:, 1])
                        charts["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                cm = confusion_matrix(y_true, y_pred)
                charts["confusion_matrix"] = {"matrix": cm.tolist()}
            except Exception:
                pass
        else:
            y_true_arr = np.asarray(y_true_raw)
            y_pred_arr = np.asarray(y_pred)
            residuals = y_true_arr - y_pred_arr

            charts["predicted_vs_actual"] = {
                "true": y_true_arr[:300].tolist(),
                "pred": y_pred_arr[:300].tolist(),
            }
            charts["residuals"] = {
                "pred": y_pred_arr[:300].tolist(),
                "residual": residuals[:300].tolist(),
            }
            hist_counts, hist_bins = np.histogram(residuals[~pd.isna(residuals)], bins=min(20, max(5, int(np.sqrt(len(residuals))))))
            charts["residual_histogram"] = {
                "bins": hist_bins.tolist(),
                "counts": hist_counts.tolist(),
            }

        return charts

    def predict(self, feature_dict: Dict[str, Any]) -> Any:
        if self.best_model is None:
            raise ValueError("No trained best model available. Run train_all_models() first.")

        input_df = pd.DataFrame([feature_dict])
        input_df = self._align_prediction_frame(input_df)
        transformed = self.pipeline.transform(input_df)
        pred = self.best_model.predict(transformed)

        if self.task_type == "classification" and self.label_encoder is not None:
            pred = self.label_encoder.inverse_transform(np.asarray(pred, dtype=int))

        val = pred[0]
        if isinstance(val, (np.integer, np.floating, int, float)):
            return float(val)
        return str(val)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _infer_task_type(self, y: pd.Series) -> str:
        non_null = y.dropna()
        if non_null.empty:
            return "classification"
        if pd.api.types.is_bool_dtype(non_null):
            return "classification"
        if pd.api.types.is_numeric_dtype(non_null):
            nunique = non_null.nunique(dropna=True)
            ratio = nunique / max(len(non_null), 1)
            if nunique <= 12 and ratio < 0.2:
                return "classification"
            return "regression"
        return "classification"

    def _build_target_summary(self, y: pd.Series) -> Dict[str, Any]:
        clean = y.dropna()
        summary = {
            "n_missing": int(y.isna().sum()),
            "n_unique": int(clean.nunique()),
            "usable_rows": int(len(clean)),
            "total_rows": int(len(y)),
        }
        if self.task_type == "classification":
            summary["class_counts"] = clean.astype(str).value_counts().to_dict()
        else:
            if len(clean) > 0:
                summary.update({
                    "mean": float(clean.mean()),
                    "median": float(clean.median()),
                    "std": float(clean.std()),
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "q1": float(clean.quantile(0.25)),
                    "q3": float(clean.quantile(0.75)),
                    "p01": float(clean.quantile(0.01)),
                    "p99": float(clean.quantile(0.99)),
                })
                try:
                    summary["skewness"] = float(skew(clean, nan_policy="omit"))
                except Exception:
                    summary["skewness"] = None
                summary["log_transform_supported"] = bool(float(clean.min()) >= 0)
                # Outlier detection via IQR
                q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
                iqr = q3 - q1
                outlier_count = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                summary["outlier_count"] = outlier_count
                summary["log_transform_suggested"] = (
                    summary["log_transform_supported"] and abs(summary.get("skewness", 0) or 0) > 2.0
                )
            else:
                summary.update({
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "q1": None,
                    "q3": None,
                    "p01": None,
                    "p99": None,
                    "skewness": None,
                    "outlier_count": 0,
                    "log_transform_supported": False,
                    "log_transform_suggested": False,
                })
        return summary

    def get_preprocessing_plan(self) -> Dict[str, Any]:
        """Describe what the engine will do during preprocessing."""
        total = len(self.df)
        
        # Build contextual plan
        plan = {}
        
        # Target transform
        if self.task_type == "regression":
            if self.use_log_target:
                plan["target_transform"] = "Applying log1p transform to the target during training; metrics are reported on the original scale"
            elif self.target_summary.get("log_transform_suggested"):
                plan["target_transform"] = "Log transform recommended due to high skewness"
        
        # Downsampling
        if total > 15000:
            plan["downsampling"] = f"Dataset will be downsampled to 15,000 rows (from {total:,})"
        
        plan["numeric_imputation"] = "Median imputation for missing numeric values"
        plan["categorical_imputation"] = "Most-frequent imputation for missing categorical values"
        plan["encoding"] = "One-Hot Encoding for categorical features"
        plan["scaling"] = "StandardScaler (z-score normalization) on numeric features"
        plan["validation"] = "5-fold cross-validation with 80/20 holdout split"
        
        # Outlier handling
        outliers = self.target_summary.get("outlier_count", 0)
        if outliers > 0:
            plan["outlier_handling"] = f"{outliers} extreme values flagged via IQR; no automatic removal applied"
        
        # Duplicates
        dup_count = int(self.df.duplicated().sum())
        if dup_count > 0:
            plan["duplicate_handling"] = f"{dup_count} duplicate rows detected and retained for now; flagged for review"
        
        return plan

    def get_candidate_model_list(self) -> List[Dict[str, str]]:
        """List models the engine plans to benchmark without training them."""
        if self.task_type == "classification":
            models = [
                {"name": "Baseline (Prior)", "tag": "baseline"},
                {"name": "Logistic Regression", "tag": "interpretable"},
                {"name": "Decision Tree", "tag": "interpretable"},
                {"name": "Random Forest", "tag": "ensemble"},
                {"name": "Gradient Boosting", "tag": "ensemble"},
                {"name": "KNN", "tag": "nonlinear"},
                {"name": "Naive Bayes", "tag": "interpretable"},
            ]
            if HAS_XGBOOST:
                models.append({"name": "XGBoost", "tag": "recommended"})
            return models
        models = [
            {"name": "Baseline (Mean)", "tag": "baseline"},
            {"name": "Linear Regression", "tag": "interpretable"},
            {"name": "Ridge Regression", "tag": "interpretable"},
            {"name": "Decision Tree", "tag": "nonlinear"},
            {"name": "Random Forest", "tag": "ensemble"},
            {"name": "Gradient Boosting", "tag": "ensemble"},
            {"name": "KNN", "tag": "nonlinear"},
        ]
        if HAS_XGBOOST:
            models.append({"name": "XGBoost", "tag": "recommended"})
        return models

    def _looks_like_id(self, series: pd.Series, col_name: str) -> bool:
        name = col_name.lower()
        if any(tok in name for tok in ["id", "uuid", "index", "listing_id", "record_id"]):
            return True
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        if pd.api.types.is_numeric_dtype(non_null):
            values = pd.to_numeric(non_null, errors="coerce").dropna()
            if values.empty:
                return False
            if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
                return False
            return (
                values.nunique() / len(values) > 0.98
                and (values.is_monotonic_increasing or values.is_monotonic_decreasing)
            )
        unique_ratio = non_null.nunique() / len(non_null)
        return unique_ratio > 0.98

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            return False
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8

    def _looks_like_long_text(self, series: pd.Series) -> bool:
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return False
        avg_len = sample.str.len().mean()
        nunique = sample.nunique()
        return avg_len > 30 or nunique > min(100, len(sample) * 0.8)

    def _looks_like_zip(self, series: pd.Series, col_name: str) -> bool:
        name = col_name.lower()
        if any(tok in name for tok in ["zip", "zipcode", "postal", "postcode"]):
            return True
        return False

    def _inspect_columns(self, X: pd.DataFrame) -> ColumnRoles:
        numeric: List[str] = []
        categorical: List[str] = []
        datetime_cols: List[str] = []
        text_cols: List[str] = []
        id_like: List[str] = []
        dropped: List[str] = []

        for col in X.columns:
            s = X[col]
            if self._looks_like_id(s, col):
                id_like.append(col)
                dropped.append(col)
                continue

            if self._looks_like_datetime(s):
                datetime_cols.append(col)
                dropped.append(col)
                continue

            if self._looks_like_long_text(s):
                text_cols.append(col)
                dropped.append(col)
                continue

            if self._looks_like_zip(s, col):
                categorical.append(col)
                continue

            if pd.api.types.is_numeric_dtype(s):
                nunique = s.nunique(dropna=True)
                if nunique <= 20 and nunique / max(len(s.dropna()), 1) < 0.05:
                    categorical.append(col)
                else:
                    numeric.append(col)
            else:
                categorical.append(col)

        return ColumnRoles(
            numeric=numeric,
            categorical=categorical,
            datetime=datetime_cols,
            text=text_cols,
            id_like=id_like,
            dropped=dropped,
        )

    def preprocess_data(self):
        clean_df = self.df.dropna(subset=[self.target]).copy()

        sample_limit = 10000
        if len(clean_df) > sample_limit:
            clean_df = clean_df.sample(n=sample_limit, random_state=42)
            self.dataset_warnings.append(f"Dataset was downsampled to {sample_limit:,} rows for responsiveness.")

        y = clean_df[self.target]
        X = clean_df.drop(columns=[self.target])

        self.column_roles = self._inspect_columns(X)
        X = X.drop(columns=self.column_roles.dropped, errors="ignore")

        if len(self.column_roles.text) > 0:
            self.dataset_warnings.append(
                f"Excluded long text columns by default: {', '.join(self.column_roles.text[:5])}."
            )
        if len(self.column_roles.id_like) > 0:
            self.dataset_warnings.append(
                f"Excluded likely ID columns: {', '.join(self.column_roles.id_like[:5])}."
            )
        if len(self.column_roles.datetime) > 0:
            self.dataset_warnings.append(
                f"Excluded datetime columns for now: {', '.join(self.column_roles.datetime[:5])}."
            )

        numeric_features = [c for c in self.column_roles.numeric if c in X.columns]
        categorical_features = [c for c in self.column_roles.categorical if c in X.columns]

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        stratify = y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        self.X_train_transformed = self.pipeline.fit_transform(self.X_train)
        self.X_test_transformed = self.pipeline.transform(self.X_test)

        onehot = self.pipeline.named_transformers_["cat"].named_steps["onehot"] if categorical_features else None
        cat_names = onehot.get_feature_names_out(categorical_features).tolist() if onehot is not None else []
        self.feature_names = numeric_features + cat_names
        self.training_profile = {
            "row_count": int(len(clean_df)),
            "feature_count": int(len(self.feature_names)),
        }

    def _build_cv(self):
        row_count = self.training_profile.get("row_count", len(self.df))
        feature_count = self.training_profile.get("feature_count", 0)
        n_splits = 3 if row_count > 8000 or feature_count > 80 else 5
        if self.task_type == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _get_candidate_models(self) -> Dict[str, Any]:
        if self.task_type == "classification":
            models = {
                "Baseline": DummyClassifier(strategy="prior"),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
            }
            if HAS_XGBOOST:
                models["XGBoost"] = XGBClassifier(
                    n_estimators=120,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                )
            return models

        models = {
            "Baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            )
        return models

    def _get_scoring_dict(self) -> Dict[str, str]:
        if self.task_type == "classification":
            return {
                "accuracy": "accuracy",
                "precision_weighted": "precision_weighted",
                "recall_weighted": "recall_weighted",
                "f1_weighted": "f1_weighted",
                "roc_auc": "roc_auc" if self._is_binary_target(self.y_train) else "roc_auc_ovr_weighted",
            }
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    def _summarize_cv_metrics(self, cv_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, values in cv_result.items():
            arr = np.asarray(values, dtype=float)
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                if metric in {"rmse", "mae"}:
                    arr = -arr
                summary[f"cv_{metric}_mean"] = float(np.nanmean(arr))
                summary[f"cv_{metric}_std"] = float(np.nanstd(arr))
            elif key in {"fit_time", "score_time"}:
                summary[f"cv_{key}_mean"] = float(np.nanmean(arr))
                summary[f"cv_{key}_std"] = float(np.nanstd(arr))
        return summary

    def _is_binary_target(self, y: pd.Series) -> bool:
        return y.dropna().nunique() == 2

    def _encode_target_if_needed(self, y: pd.Series, fit_encoder: bool) -> np.ndarray:
        if self.task_type != "classification":
            return np.asarray(y)

        clean = pd.Series(y).copy()
        if pd.api.types.is_numeric_dtype(clean) and clean.dropna().nunique() <= 20:
            return np.asarray(clean)

        if fit_encoder or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(clean.astype(str))
        return self.label_encoder.transform(clean.astype(str))

    def _evaluate_on_holdout(self, model, X_test, y_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        extras: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        if self.task_type == "classification":
            y_true = self._encode_target_if_needed(y_test, fit_encoder=False)
            y_pred = model.predict(X_test)

            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(X_test)
                    if prob.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="weighted"))
                except Exception:
                    metrics["roc_auc"] = None

            extras["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
            return metrics, extras

        y_pred = model.predict(X_test)
        y_true = np.asarray(y_test)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        return metrics, extras

    def _pick_best_model(self) -> Optional[str]:
        valid = {k: v for k, v in self.results.items() if v.get("cv_metrics") and not v.get("error")}
        if not valid:
            return None

        if self.task_type == "classification":
            ranked = sorted(
                valid.items(),
                key=lambda kv: (
                    kv[1]["cv_metrics"].get("cv_accuracy_mean", -np.inf),
                    kv[1]["cv_metrics"].get("cv_roc_auc_mean", -np.inf),
                    kv[1]["test_metrics"].get("accuracy", -np.inf),
                ),
                reverse=True,
            )
            return ranked[0][0]

        ranked = sorted(
            valid.items(),
            key=lambda kv: (
                kv[1]["cv_metrics"].get("cv_r2_mean", -np.inf),
                -kv[1]["cv_metrics"].get("cv_rmse_mean", np.inf),
                kv[1]["test_metrics"].get("r2", -np.inf),
            ),
            reverse=True,
        )
        return ranked[0][0]

    def _build_best_model_reason(self) -> str:
        if not self.best_model_name:
            return ""
        result = self.results[self.best_model_name]
        if self.task_type == "classification":
            acc = result["cv_metrics"].get("cv_accuracy_mean")
            auc = result["cv_metrics"].get("cv_roc_auc_mean")
            return (
                f"{self.best_model_name} was selected because it ranked highest on cross-validated accuracy. "
                f"CV accuracy was {acc:.3f}" if acc is not None else f"{self.best_model_name} was selected because it ranked highest on cross-validated accuracy."
            ) + (
                f" and CV ROC AUC was {auc:.3f}" if auc is not None else ""
            ) + "."
        r2 = result["cv_metrics"].get("cv_r2_mean")
        rmse = result["cv_metrics"].get("cv_rmse_mean")
        holdout_r2 = result["test_metrics"].get("r2")
        details = []
        if r2 is not None:
            details.append(f"CV R² was {r2:.3f}")
        if rmse is not None:
            details.append(f"CV RMSE was {rmse:.3f}")
        if holdout_r2 is not None:
            details.append(f"holdout R² was {holdout_r2:.3f}")
        detail_text = ", ".join(details)
        if detail_text:
            reason = f"{self.best_model_name} was selected because it ranked highest on cross-validated R². {detail_text}."
        else:
            reason = f"{self.best_model_name} was selected because it ranked highest on cross-validated R²."
        if self.use_log_target:
            reason += " Training used log1p(target), while reported metrics remain on the original target scale."
        return reason

    def _build_recommendations(self) -> List[str]:
        recs: List[str] = []
        if self.column_roles is None:
            return recs

        if self.column_roles.text:
            recs.append(
                "If text columns are important, build a separate text pipeline with TF-IDF or embeddings instead of one-hot encoding raw text."
            )
        if self.column_roles.datetime:
            recs.append(
                "Convert datetime columns into engineered features such as year, month, weekday, or recency for better model performance."
            )
        if len(self.column_roles.categorical) > 15:
            recs.append(
                "High numbers of categorical variables can increase sparsity. Consider grouping rare categories or using target/frequency encoding in a future version."
            )
        if self.task_type == "regression":
            if self.use_log_target:
                recs.append(
                    "The target was trained with log1p scaling, which is often useful for positive, right-skewed regression targets."
                )
            recs.append(
                "For regression, compare residual plots and consider log-transforming a heavily skewed target if appropriate."
            )
        else:
            recs.append(
                "For classification, review class balance, threshold choice, and calibration if false positives and false negatives have different costs."
            )
        return recs

    def _align_prediction_frame(self, input_df: pd.DataFrame) -> pd.DataFrame:
        expected_cols = self.X_train.columns.tolist()
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan
        return input_df[expected_cols]

    def _to_json_scalar(self, value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def _wrap_model_for_training(self, model):
        if self.task_type == "regression" and self.use_log_target:
            return TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        return model

    def _unwrap_model(self, model):
        return getattr(model, "regressor_", getattr(model, "regressor", model))
