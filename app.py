import streamlit as st
import pandas as pd
from predict import CreditRiskModel

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🧮",
    layout="centered",
)


@st.cache_resource
def load_model():
    return CreditRiskModel()

 
model = load_model()

model = load_model()
feature_names = model.feature_names
defaults = model.ideal_defaults()

st.title("Credit Risk Predictor")
st.caption(
    "Predict the risk of a credit application with ideal defaults and clear "
    "guidance."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses an XGBoost classifier trained on a cleaned and one-hot "
        "encoded credit risk dataset. Provide applicant and loan details "
        "below to estimate risk."
    )
    st.write(
        "Model file:",
        (
            model.MODEL_FILENAME
            if hasattr(model, "MODEL_FILENAME")
            else "xgboost_model.joblib"
        ),
    )
    st.markdown("---")
    st.write("Feature count:", len(feature_names))

st.subheader("Applicant & Loan Details")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=int(defaults["person_age"]),
    )
    person_income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        max_value=1000000,
        value=int(defaults["person_income"]),
    )
    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=60.0,
        step=0.5,
        value=float(defaults["person_emp_length"]),
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0,
        max_value=60,
        value=int(defaults["cb_person_cred_hist_length"]),
    )

with col2:
    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500,
        max_value=1000000,
        value=int(defaults["loan_amnt"]),
    )
    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=60.0,
        step=0.1,
        value=float(defaults["loan_int_rate"]),
    )
    loan_percent_income = st.number_input(
        "Loan Payment / Income",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=float(defaults["loan_percent_income"]),
    )

st.subheader("Categorical Features")

home_options = ["OWN", "MORTGAGE", "RENT", "OTHER"]
intent_options = [
    "DEBTCONSOLIDATION",
    "HOMEIMPROVEMENT",
    "EDUCATION",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
]
grade_options = ["A", "B", "C", "D", "E", "F", "G"]
cb_default_options = ["N", "Y"]

person_home_ownership = st.selectbox(
    "Home Ownership",
    options=home_options,
    index=home_options.index(defaults["person_home_ownership"]),
)
loan_intent = st.selectbox(
    "Loan Intent",
    options=intent_options,
    index=intent_options.index(defaults["loan_intent"]),
)
loan_grade = st.selectbox(
    "Loan Grade",
    options=grade_options,
    index=grade_options.index(defaults["loan_grade"]),
)
cb_person_default_on_file = st.radio(
    "Default on File",
    options=cb_default_options,
    index=cb_default_options.index(defaults["cb_person_default_on_file"]),
)

st.markdown("---")

if st.button("Predict Risk", type="primary"):
    # Gather inputs
    inputs = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
    }

    proba, pred = model.predict_proba(inputs)
    label, color = CreditRiskModel.risk_category(proba)

    st.markdown(
        f"### Risk: <span style='color:{color}'>{label}</span>",
        unsafe_allow_html=True,
    )
    st.metric(label="Probability of Risk", value=f"{proba:.2%}")
    st.write("Predicted class (1=risk, 0=non-risk):", int(pred))

    # Show the assembled feature row for transparency
    st.expander("View model-ready feature vector").write(
        model.build_feature_vector(inputs)
    )

    # Optional: simple feature importance display if available
    try:
        importances = getattr(model.model, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            st.subheader("Feature Importances")
            st.bar_chart(imp_df.set_index("feature").head(20))
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

    # SHAP explanation
    try:
        shap_df, meta = model.explain_shap(inputs)
        st.subheader("SHAP Explanation")
        st.write(
            "Contributions to predicted probability "
            "(positive increases risk, negative decreases)."
        )
        st.write(
            f"Base value (expected prob): {meta['base_value']:.2%} | "
            f"Predicted prob: {meta['prediction_probability']:.2%}"
        )
        st.bar_chart(shap_df.set_index("feature")["shap_value"].head(20))
        with st.expander("Top 30 SHAP contributions"):
            st.dataframe(shap_df.head(30))
    except Exception as e:
        st.info(f"SHAP explanation not available: {e}")

st.markdown("---")
st.caption(
    "Tip: The defaults are set to an ideal low-risk profile. Adjust values "
    "to explore how risk changes."
)
