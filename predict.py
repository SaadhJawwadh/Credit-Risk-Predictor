import os
from typing import Dict, List, Tuple
import numpy as np  # noqa: F401  # reserved for future numerical ops
import pandas as pd
import joblib
import warnings

try:
    import shap
except Exception:  # pragma: no cover
    shap = None  # will handle gracefully in explain_shap

MODEL_FILENAME = "xgboost_model.joblib"

class CreditRiskModel:
    def __init__(self, model_path: str = MODEL_FILENAME):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        self.feature_names = self._infer_feature_names()

    def _infer_feature_names(self) -> List[str]:
        # Preferred: sklearn-style feature names
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            return list(names)
        # Fallback: xgboost booster feature names
        try:
            booster = self.model.get_booster()
            if booster is not None and booster.feature_names is not None:
                return list(booster.feature_names)
        except Exception:
            pass
        # Final fallback: known schema from common credit risk dataset
        # Warn: order must match training order; this is best-effort.
        default_features = [
            # Numeric features
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            # One-hot encoded categorical features (drop_first=True was used)
            # person_home_ownership: typically {MORTGAGE, OWN, RENT, OTHER}
            "person_home_ownership_MORTGAGE",
            "person_home_ownership_OWN",
            "person_home_ownership_RENT",
            "person_home_ownership_OTHER",
            # loan_intent categories commonly seen in dataset
            "loan_intent_DEBTCONSOLIDATION",
            "loan_intent_EDUCATION",
            "loan_intent_HOMEIMPROVEMENT",
            "loan_intent_MEDICAL",
            "loan_intent_PERSONAL",
            "loan_intent_VENTURE",
            # loan_grade: {A,B,C,D,E,F,G}
            "loan_grade_B",
            "loan_grade_C",
            "loan_grade_D",
            "loan_grade_E",
            "loan_grade_F",
            "loan_grade_G",
            # cb_person_default_on_file: {Y} (drop_first removes N)
            "cb_person_default_on_file_Y",
        ]
        return default_features

    def ideal_defaults(self) -> Dict[str, object]:
        """Return a set of ideal low-risk default values for the UI."""
        return {
            "person_age": 35,
            "person_income": 85000,
            "person_emp_length": 8,
            "loan_amnt": 10000,
            "loan_int_rate": 11.0,
            "loan_percent_income": 0.15,
            "cb_person_cred_hist_length": 10,
            # OWN/MORTGAGE typically lower risk
            "person_home_ownership": "OWN",
            "loan_intent": "DEBTCONSOLIDATION",  # or HOMEIMPROVEMENT
            "loan_grade": "A",
            "cb_person_default_on_file": "N",
        }

    def _one_hot_columns(self) -> Dict[str, List[str]]:
        """Group categorical dummies by prefix based on model feature names."""
        groups: Dict[str, List[str]] = {
            "person_home_ownership": [],
            "loan_intent": [],
            "loan_grade": [],
            "cb_person_default_on_file": [],
        }
        for col in self.feature_names:
            if col.startswith("person_home_ownership_"):
                groups["person_home_ownership"].append(col)
            elif col.startswith("loan_intent_"):
                groups["loan_intent"].append(col)
            elif col.startswith("loan_grade_"):
                groups["loan_grade"].append(col)
            elif col.startswith("cb_person_default_on_file_"):
                groups["cb_person_default_on_file"].append(col)
        return groups

    def _make_background(self, n: int = 200) -> pd.DataFrame:
        """
        Create a synthetic background dataset for SHAP interventional
        explainer. The ranges are heuristic but aligned with common
        credit datasets.
        """
        defaults = self.ideal_defaults()
        rng = np.random.default_rng(42)

        home_opts = ["OWN", "MORTGAGE", "RENT", "OTHER"]
        intent_opts = [
            "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT",
            "EDUCATION",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
        ]
        grade_opts = ["A", "B", "C", "D", "E", "F", "G"]
        cb_opts = ["N", "Y"]

        rows: List[pd.DataFrame] = []
        for _ in range(n):
            sample = {
                "person_age": float(
                    np.clip(
                        rng.normal(defaults["person_age"], 8.0), 18, 90
                    )
                ),
                "person_income": float(
                    np.clip(
                        rng.normal(defaults["person_income"], 30000.0),
                        10000,
                        300000,
                    )
                ),
                "person_emp_length": float(
                    np.clip(
                        rng.normal(defaults["person_emp_length"], 3.0), 0, 40
                    )
                ),
                "loan_amnt": float(
                    np.clip(
                        rng.normal(defaults["loan_amnt"], 8000.0), 500, 100000
                    )
                ),
                "loan_int_rate": float(
                    np.clip(
                        rng.normal(defaults["loan_int_rate"], 5.0), 1.0, 40.0
                    )
                ),
                "loan_percent_income": float(
                    np.clip(
                        rng.normal(defaults["loan_percent_income"], 0.08),
                        0.0,
                        1.0,
                    )
                ),
                "cb_person_cred_hist_length": float(
                    np.clip(
                        rng.normal(
                            defaults["cb_person_cred_hist_length"], 5.0
                        ),
                        0,
                        40,
                    )
                ),
                "person_home_ownership": rng.choice(home_opts),
                "loan_intent": rng.choice(intent_opts),
                "loan_grade": rng.choice(grade_opts),
                "cb_person_default_on_file": rng.choice(cb_opts),
            }
            rows.append(self.build_feature_vector(sample))
        bg = pd.concat(rows, ignore_index=True)
        return bg

    def build_feature_vector(self, inputs: Dict[str, object]) -> pd.DataFrame:
        """
        Build a single-row DataFrame matching the model's feature_names order.
        """
        # Start with zeros for all features
        data = {name: 0.0 for name in self.feature_names}

        # Numeric assignments (if present in features)
        numeric_map = {
            "person_age": float(inputs.get("person_age", 0)),
            "person_income": float(inputs.get("person_income", 0)),
            "person_emp_length": float(inputs.get("person_emp_length", 0)),
            "loan_amnt": float(inputs.get("loan_amnt", 0)),
            "loan_int_rate": float(inputs.get("loan_int_rate", 0)),
            "loan_percent_income": float(inputs.get("loan_percent_income", 0)),
            "cb_person_cred_hist_length": float(
                inputs.get("cb_person_cred_hist_length", 0)
            ),
        }
        for k, v in numeric_map.items():
            if k in data:
                data[k] = v

        # One-hot assignments
        groups = self._one_hot_columns()
        # person_home_ownership
        pho_val = str(inputs.get("person_home_ownership", "")).upper()
        for col in groups.get("person_home_ownership", []):
            # col like person_home_ownership_RENT
            if col.endswith(f"_{pho_val}"):
                data[col] = 1.0
        # loan_intent
        li_val = str(inputs.get("loan_intent", "")).upper()
        for col in groups.get("loan_intent", []):
            if col.endswith(f"_{li_val}"):
                data[col] = 1.0
        # loan_grade
        lg_val = str(inputs.get("loan_grade", "")).upper()
        for col in groups.get("loan_grade", []):
            if col.endswith(f"_{lg_val}"):
                data[col] = 1.0
        # cb_person_default_on_file -> typically only _Y dummy
        cb_val = str(inputs.get("cb_person_default_on_file", "")).upper()
        for col in groups.get("cb_person_default_on_file", []):
            # set 1.0 if user selected Y
            if col.endswith("_Y") and cb_val == "Y":
                data[col] = 1.0

        # Return as DataFrame with ordered columns
        df_row = pd.DataFrame(
            [[data[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        return df_row

    def predict_proba(self, inputs: Dict[str, object]) -> Tuple[float, int]:
        X_row = self.build_feature_vector(inputs)
        # XGBoost returns proba for both classes; assume 1 is risk/positive
        proba = float(self.model.predict_proba(X_row)[:, 1][0])
        pred = int(self.model.predict(X_row)[0])
        return proba, pred

    @staticmethod
    def risk_category(proba: float) -> Tuple[str, str]:
        """Return (label, color) based on probability threshold."""
        if proba < 0.2:
            return "Low Risk", "#2e7d32"  # green
        elif proba < 0.5:
            return "Medium Risk", "#f9a825"  # amber
        else:
            return "High Risk", "#c62828"  # red

    def explain_shap(
        self, inputs: Dict[str, object]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute SHAP contributions for a single prediction.

        Returns:
            (df, meta):
              - df: DataFrame with columns [feature, shap_value, abs_shap]
              - meta: dict with keys like base_value and prediction_probability
        """
        if shap is None:
            raise RuntimeError(
                "SHAP is not installed. Please install 'shap' "
                "to use explanations."
            )

        X_row = self.build_feature_vector(inputs)

        # Try TreeExplainer with probability output
        explainer = None
        shap_values = None
        base_value = None
        try:
            background = self._make_background(n=200)
            explainer = shap.TreeExplainer(
                self.model,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            shap_values = explainer.shap_values(X_row)
            base_value = float(np.ravel(explainer.expected_value)[0])
        except Exception as e:
            warnings.warn(
                f"TreeExplainer failed, fallback to generic Explainer: {e}"
            )
            try:
                explainer = shap.Explainer(self.model, X_row)
                explanation = explainer(X_row)
                shap_values = explanation.values
                base_value = float(np.ravel(explanation.base_values)[0])
            except Exception as e2:
                raise RuntimeError(f"Failed to compute SHAP values: {e2}")

        # shap_values may be list for multiclass; handle binary/classic
        if isinstance(shap_values, list):
            # choose index 1 for positive class contributions if available
            shap_arr = np.array(shap_values[1]).reshape(-1)
        else:
            shap_arr = np.array(shap_values).reshape(-1)

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": shap_arr,
            }
        )
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

        proba, _ = self.predict_proba(inputs)
        meta = {"base_value": base_value, "prediction_probability": proba}
        return df, meta


if __name__ == "__main__":
    # Simple CLI sanity test
    model = CreditRiskModel()
    defaults = model.ideal_defaults()
    proba, pred = model.predict_proba(defaults)
    label, color = CreditRiskModel.risk_category(proba)
    print(
        f"Sanity test -> proba: {proba:.4f}, pred: {pred}, category: {label}"
    )
