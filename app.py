import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Kepler | Exoplanet Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# PATHS


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "kepler_deployment_artifacts.pkl"

DATA_CANDIDATES = [
    ROOT / "data" / "dataset.csv",
    ROOT / "data" / "kepler.csv",
]

IMAGE_CANDIDATES = {
    "shap": [
        ROOT / "shap_summary.png",
        ROOT / "notebook" / "shap_summary.png",
    ],
    "confusion": [
        ROOT / "confusion_matrix.png",
        ROOT / "notebook" / "confusion_matrix.png",
    ],
    "comparison": [
        ROOT / "model_comparison.png",
        ROOT / "notebook" / "model_comparison.png",
    ],
}




st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(77, 129, 255, 0.10), transparent 28%),
                radial-gradient(circle at 85% 20%, rgba(143, 92, 255, 0.09), transparent 25%),
                #070b14;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(20, 29, 52, .96), rgba(12, 18, 34, .96));
            border: 1px solid rgba(255,255,255,.09);
            margin-bottom: 1.4rem;
        }

        .eyebrow {
            color: #8da8ff;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }

        .hero h1 {
            color: #f4f7ff;
            font-size: 2.55rem;
            margin: 0;
            line-height: 1.1;
        }

        .hero p {
            color: #aeb9cf;
            font-size: 1rem;
            max-width: 760px;
            margin-top: .8rem;
        }

        .metric-card {
            background: rgba(17, 25, 43, .86);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 18px;
            padding: 1.1rem 1.25rem;
            min-height: 105px;
        }

        .metric-label {
            color: #8d99b2;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .metric-value {
            color: #f4f7ff;
            font-size: 1.65rem;
            font-weight: 750;
            margin-top: .25rem;
        }

        .section-title {
            color: #f1f5ff;
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1.6rem;
            margin-bottom: .75rem;
        }

        .prediction {
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            background: linear-gradient(135deg, rgba(30, 43, 73, .95), rgba(18, 25, 43, .95));
            border: 1px solid rgba(141, 168, 255, .22);
            margin: 1rem 0;
        }

        .prediction-label {
            color: #91a0ba;
            font-size: .8rem;
            text-transform: uppercase;
            letter-spacing: .12em;
        }

        .prediction-value {
            color: #ffffff;
            font-size: 2.25rem;
            font-weight: 800;
            margin-top: .15rem;
        }

        .prediction-sub {
            color: #aeb9cf;
            margin-top: .35rem;
        }

        .hint {
            color: #8490a8;
            font-size: .82rem;
            margin-top: .35rem;
        }

        .pill {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: rgba(141, 168, 255, .10);
            color: #a9bbff;
            border: 1px solid rgba(141, 168, 255, .16);
            font-size: .78rem;
            margin-right: .35rem;
        }

        div.stButton > button {
            border-radius: 12px;
            min-height: 2.7rem;
            font-weight: 650;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(17, 25, 43, .60);
            border-radius: 16px;
            padding: .4rem;
            border: 1px solid rgba(255,255,255,.07);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# MODEL and DATA LOADING


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)


@st.cache_data
def load_dataset(path_string):
    return pd.read_csv(path_string)


artifacts = load_artifacts()
model = artifacts["model"]
outlier_caps = artifacts["outlier_caps"]

dataset_path = next(
    (p for p in DATA_CANDIDATES if p.exists()),
    None
)


# FEATURE ENGINEERING

ERROR_PAIRS = [
    ("koi_period", "koi_period_err1", "koi_period_err2"),
    ("koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2"),
    ("koi_impact", "koi_impact_err1", "koi_impact_err2"),
    ("koi_duration", "koi_duration_err1", "koi_duration_err2"),
    ("koi_depth", "koi_depth_err1", "koi_depth_err2"),
    ("koi_prad", "koi_prad_err1", "koi_prad_err2"),
    ("koi_insol", "koi_insol_err1", "koi_insol_err2"),
    ("koi_steff", "koi_steff_err1", "koi_steff_err2"),
    ("koi_slogg", "koi_slogg_err1", "koi_slogg_err2"),
    ("koi_srad", "koi_srad_err1", "koi_srad_err2"),
]

OUTLIER_COLUMNS = [
    "koi_period",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_slogg",
    "koi_srad",
]

ERROR_COLUMNS = [
    col
    for _, err1, err2 in ERROR_PAIRS
    for col in (err1, err2)
]


def prepare_deployment_data(raw_input):
    data = raw_input.copy()

    data.drop(
        columns=[
            "rowid",
            "kepid",
            "kepler_name",
            "koi_pdisposition",
            "koi_score",
            "kepoi_name",
            "koi_disposition",
        ],
        errors="ignore",
        inplace=True,
    )

    for feature in OUTLIER_COLUMNS:
        if feature in data.columns:
            data[feature] = np.where(
                data[feature] >= outlier_caps[feature],
                outlier_caps[feature],
                data[feature],
            )

    eps = 1e-8

    for feature, err1, err2 in ERROR_PAIRS:
        data[f"{feature}_uncertainity"] = (
            data[err1] - data[err2]
        )

        data[f"{feature}_uncertainity_rel"] = (
            (data[err1] - data[err2])
            / (data[feature] + eps)
        )

    data.drop(
        columns=ERROR_COLUMNS,
        errors="ignore",
        inplace=True,
    )

    data.drop(
        columns=["koi_tce_delivname"],
        errors="ignore",
        inplace=True,
    )

    return data



# RANDOM EXAMPLE


def generate_random_example():
    if dataset_path is None:
        return None

    data = load_dataset(str(dataset_path))

    # Random row every time the button is pressed.
    # random_state=None intentionally gives a fresh observation.
    row = data.sample(
        n=1,
        random_state=None
    ).copy()

    return row


if "example_input" not in st.session_state:
    st.session_state["example_input"] = None

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if "probabilities" not in st.session_state:
    st.session_state["probabilities"] = None

if "ground_truth" not in st.session_state:
    st.session_state["ground_truth"] = None




st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Kepler Mission • Machine Learning</div>
        <h1>Exoplanet Candidate Classifier</h1>
        <p>
            An end-to-end machine learning system that classifies
            Kepler Objects of Interest as
            <b>Confirmed</b>, <b>Candidate</b>, or <b>False Positive</b>.
        </p>
        <span class="pill">Gradient Boosting</span>
        <span class="pill">37 selected features</span>
        <span class="pill">SHAP interpretation</span>
    </div>
    """,
    unsafe_allow_html=True,
)


metric_cols = st.columns(4)

metrics = [
    ("Test Accuracy", f"{artifacts['test_accuracy'] * 100:.2f}%"),
    ("Macro F1", f"{artifacts['test_macro_f1']:.3f}"),
    ("Selected Features", "37"),
    ("Classes", "3"),
]

for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



st.markdown(
    '<div class="section-title">🔭 Try the classifier</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    if st.button(
        "✨ Generate Random Kepler Observation",
        use_container_width=True,
    ):
        st.session_state["example_input"] = generate_random_example()
        st.session_state["prediction"] = None
        st.session_state["probabilities"] = None
        st.session_state["ground_truth"] = None

with right:
    uploaded_file = st.file_uploader(
        "Upload a Kepler CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

if st.session_state["example_input"] is not None:
    raw_input = st.session_state["example_input"].copy()
    source_label = "Random observation from the dataset"

elif uploaded_file is not None:
    raw_input = pd.read_csv(uploaded_file)
    source_label = "Uploaded CSV"

else:
    raw_input = None
    source_label = None

if raw_input is not None:

    st.markdown(
        f'<div class="hint">Source: {source_label} • '
        f'{len(raw_input)} observation(s)</div>',
        unsafe_allow_html=True,
    )

    # For a single random example, show the useful raw measurements.
    preview_columns = [
        c for c in [
            "koi_period",
            "koi_depth",
            "koi_prad",
            "koi_teq",
            "koi_insol",
            "koi_model_snr",
            "koi_steff",
            "koi_slogg",
            "koi_srad",
            "koi_kepmag",
        ]
        if c in raw_input.columns
    ]

    if preview_columns:
        st.dataframe(
            raw_input[preview_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "🚀 Run Prediction",
        type="primary",
        use_container_width=True,
    ):
        try:
            prepared = prepare_deployment_data(raw_input)

            predictions = model.predict(prepared)
            probabilities = model.predict_proba(prepared)

            st.session_state["prediction"] = predictions
            st.session_state["probabilities"] = probabilities

            if len(raw_input) == 1 and "koi_disposition" in raw_input.columns:
                st.session_state["ground_truth"] = (
                    raw_input["koi_disposition"].iloc[0]
                )
            else:
                st.session_state["ground_truth"] = None

        except Exception as exc:
            st.error(
                "This input does not contain the feature columns required "
                "by the trained model."
            )
            st.exception(exc)


# RESULTS


if st.session_state["prediction"] is not None:

    predictions = st.session_state["prediction"]
    probabilities = st.session_state["probabilities"]
    ground_truth = st.session_state["ground_truth"]

    if len(predictions) == 1:

        prediction = predictions[0]

        if prediction == "CONFIRMED":
            result_symbol = "🪐"
        elif prediction == "CANDIDATE":
            result_symbol = "🔎"
        else:
            result_symbol = "⚠️"

        st.markdown(
            f"""
            <div class="prediction">
                <div class="prediction-label">Model prediction</div>
                <div class="prediction-value">
                    {result_symbol} {prediction}
                </div>
                <div class="prediction-sub">
                    Gradient Boosting classification result
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_cols = st.columns(3)

        for i, class_name in enumerate(model.classes_):
            probability = probabilities[0][i] * 100
            with result_cols[i]:
                st.metric(
                    class_name,
                    f"{probability:.1f}%"
                )
                st.progress(
                    float(probabilities[0][i])
                )

        if ground_truth is not None:
            if prediction == ground_truth:
                st.success(
                    f"Prediction matches the dataset label: **{ground_truth}**"
                )
            else:
                st.warning(
                    f"Dataset label: **{ground_truth}** • "
                    f"Model predicted: **{prediction}**"
                )

    else:

        result_df = pd.DataFrame({
            "Prediction": predictions
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Model interpretation</div>',
    unsafe_allow_html=True,
)

shap_path = next(
    (p for p in IMAGE_CANDIDATES["shap"] if p.exists()),
    None,
)

if shap_path is not None:
    st.image(
        str(shap_path),
        caption="Global SHAP feature importance",
        use_container_width=True,
    )

with st.expander("What does SHAP tell us?"):
    st.write(
        "SHAP measures how much each feature contributes to the model's "
        "predictions. Larger mean absolute SHAP values indicate features "
        "that have a stronger overall influence on the classifier."
    )


# MODEL PERFORMANCE


st.markdown(
    '<div class="section-title">📊 Model performance</div>',
    unsafe_allow_html=True,
)

performance_cols = st.columns(2)

comparison_path = next(
    (p for p in IMAGE_CANDIDATES["comparison"] if p.exists()),
    None,
)

confusion_path = next(
    (p for p in IMAGE_CANDIDATES["confusion"] if p.exists()),
    None,
)

with performance_cols[0]:
    if comparison_path is not None:
        st.image(
            str(comparison_path),
            caption="Model comparison",
            use_container_width=True,
        )

with performance_cols[1]:
    if confusion_path is not None:
        st.image(
            str(confusion_path),
            caption="Confusion matrix",
            use_container_width=True,
        )

# FOOTER

st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • SHAP • Streamlit"
)
