from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from churn_preprocessing import align_prediction_columns, build_target, preprocess_raw_data


MODELS_DIR = BASE_DIR / "models"
PREFERRED_MODEL_PATH = MODELS_DIR / "best_churn_model_decision_tree_v2.joblib"
MAPPINGS_PATH = MODELS_DIR / "ordinal_mappings.json"
INCLUDED_DATA_NOTARGET_PATH = BASE_DIR / "altas_Bajas_notarget_checkbox.csv"
CSV_ENCODINGS = ("utf-8", "utf-16", "latin1")
RESULTS_STATE_KEY = "churn_app_results"
UPLOAD_STATE_KEY = "churn_app_upload_key"


st.set_page_config(page_title="Prediccion de Churn Movistar", layout="wide")


@st.cache_resource
def load_model():
    if PREFERRED_MODEL_PATH.exists():
        model_files = [PREFERRED_MODEL_PATH]
    else:
        model_files = []

    if not model_files:
        return None

    path = model_files[0]
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        loaded["model_path"] = str(path)
        loaded["direct_colab_model"] = False
        return loaded

    feature_columns = list(getattr(loaded, "feature_names_in_", []))
    if not feature_columns:
        raise ValueError(
            "El .joblib no trae feature_names_in_. Necesito saber las columnas usadas al entrenar."
        )

    return {
        "pipeline": loaded,
        "feature_columns": feature_columns,
        "threshold": 0.5,
        "metrics": {},
        "model_path": str(path),
        "direct_colab_model": True,
    }


@st.cache_data
def load_ordinal_mappings():
    if not MAPPINGS_PATH.exists():
        return None
    return json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))


def read_csv_auto_encoding(source, sep: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=sep, encoding=encoding)
        except UnicodeError as exc:
            last_error = exc
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "No pude leer el CSV con los formatos soportados: "
        + ", ".join(CSV_ENCODINGS)
    ) from last_error


@st.cache_data
def load_included_data(path: Path, sep: str, file_mtime: float, file_size: int) -> pd.DataFrame:
    return read_csv_auto_encoding(path, sep)


def read_uploaded_csv(uploaded_file, sep: str) -> pd.DataFrame:
    return read_csv_auto_encoding(uploaded_file, sep)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predicciones")
    return buffer.getvalue()


def extract_model_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model")
    return model


def get_feature_importances(model, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = extract_model_estimator(model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None

    return (
        pd.DataFrame({"variable": feature_columns, "importancia": importances})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(feature_importances: pd.DataFrame) -> plt.Figure:
    top_10_features = feature_importances.head(10)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_10_features,
        x="importancia",
        y="variable",
        hue="variable",
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_title("Top 10 variables mas importantes - Decision Tree", fontsize=16)
    ax.set_xlabel("Importancia", fontsize=14)
    ax.set_ylabel("Variable", fontsize=14)
    sns.despine(left=True, bottom=True)
    return fig


def target_from_raw_data(raw_df: pd.DataFrame) -> pd.Series | None:
    if "PERIODO_BAJA" not in raw_df.columns:
        return None
    return build_target(raw_df)


def prepare_features(
    raw_df: pd.DataFrame,
    artifact: dict,
    ordinal_mappings: dict | None,
) -> pd.DataFrame:
    processed_df = preprocess_raw_data(
        raw_df,
        training=False,
        ordinal_mappings=ordinal_mappings,
    )
    X = align_prediction_columns(processed_df, artifact["feature_columns"])

    if artifact.get("direct_colab_model"):
        X = X.apply(pd.to_numeric, errors="coerce")
        columns_with_missing = X.columns[X.isna().any()].tolist()
        if columns_with_missing:
            raise ValueError(
                "El modelo espera columnas codificadas como numeros. Revisa los mapeos "
                "o categorias desconocidas en: " + ", ".join(columns_with_missing)
            )

    return X


def run_churn_analysis(
    raw_df: pd.DataFrame,
    artifact: dict,
    ordinal_mappings: dict | None,
) -> dict:
    X = prepare_features(raw_df, artifact, ordinal_mappings)
    model = artifact["pipeline"]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X).astype(int)
    labels = ["Churn" if prediction == 1 else "No churn" for prediction in predictions]

    results_df = raw_df.copy()
    results_df["estado_churn"] = labels
    results_df["prediccion_churn"] = predictions
    results_df["probabilidad_churn"] = probabilities
    results_df["probabilidad_churn_pct"] = [f"{probability:.2%}" for probability in probabilities]

    return {
        "results_df": results_df,
        "target": target_from_raw_data(raw_df),
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_importances": get_feature_importances(model, artifact["feature_columns"]),
        "input_columns": list(raw_df.columns),
        "rows_scored": len(X),
    }


artifact = load_model()
ordinal_mappings = load_ordinal_mappings()

st.title("Prediccion de Churn")
st.caption("Carga clientes, ejecuta el analisis y prioriza los casos con mayor riesgo de abandono.")

if artifact is None:
    st.error(
        "No encontre el modelo models/best_churn_model_decision_tree_v2.joblib."
    )
    st.stop()

with st.sidebar:
    st.header("Configuracion")
    sep = st.text_input("Separador", value="|", max_chars=3)

uploaded_file = st.file_uploader("CSV de clientes", type=["csv"])
included_files_available = (
    INCLUDED_DATA_NOTARGET_PATH.exists()
)
use_included_data = st.checkbox(
    "Usar datos incluidos",
    value=uploaded_file is None,
    disabled=not included_files_available,
)

if uploaded_file is None and not use_included_data:
    st.info("Carga un CSV o activa los datos incluidos para calcular el riesgo de churn.")
    st.stop()

try:
    if use_included_data:
        included_path = INCLUDED_DATA_NOTARGET_PATH
        included_stat = included_path.stat()
        raw_df = load_included_data(
            included_path,
            sep,
            included_stat.st_mtime,
            included_stat.st_size,
        )
        data_key = f"{included_path}_{included_stat.st_mtime}_{included_stat.st_size}_{sep}"
    elif uploaded_file is not None:
        raw_df = read_uploaded_csv(uploaded_file, sep=sep)
        data_key = f"{uploaded_file.name}_{uploaded_file.size}_{sep}"
    else:
        st.info("Carga un CSV o activa los datos incluidos para calcular el riesgo de churn.")
        st.stop()
except Exception as exc:
    st.error("No pude leer el archivo. Revisa que sea un CSV separado por el caracter configurado.")
    st.exception(exc)
    st.stop()

if st.session_state.get(UPLOAD_STATE_KEY) != data_key:
    st.session_state[UPLOAD_STATE_KEY] = data_key
    st.session_state[RESULTS_STATE_KEY] = None

st.subheader("Vista previa")
st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

target = target_from_raw_data(raw_df)
if target is not None:
    st.caption("El archivo trae PERIODO_BAJA; se usara solo para evaluar resultados, no como variable predictora.")
    target_preview = pd.DataFrame({"churn_real": target})
    if "ID" in raw_df.columns:
        target_preview.insert(0, "ID", raw_df["ID"].values)
    st.dataframe(
        target_preview.sort_values("churn_real", ascending=False).head(20),
        width="stretch",
        hide_index=True,
    )

if st.button("Run Churn Analysis", type="primary"):
    try:
        st.session_state[RESULTS_STATE_KEY] = run_churn_analysis(
            raw_df,
            artifact,
            ordinal_mappings,
        )
    except Exception as exc:
        st.session_state[RESULTS_STATE_KEY] = None
        st.error("No pude procesar el archivo. Revisa columnas, separador y mapeos.")
        st.exception(exc)

cached = st.session_state.get(RESULTS_STATE_KEY)
if cached is None:
    st.info("Cuando el archivo este listo, ejecuta el analisis para generar predicciones.")
    st.stop()

results_df = cached["results_df"]
feature_importances = cached["feature_importances"]
input_columns = cached["input_columns"]

st.success("Analisis completo.")
st.info(f"Modelo ejecutado sobre {cached['rows_scored']:,} registros del archivo cargado.")

total_clientes = len(results_df)
churn_predicho = int((results_df["estado_churn"] == "Churn").sum())
no_churn_predicho = int((results_df["estado_churn"] == "No churn").sum())
probabilidad_promedio = float(results_df["probabilidad_churn"].mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes evaluados", f"{total_clientes:,}")
col2.metric("Churn predicho", f"{churn_predicho:,}")
col3.metric("No churn predicho", f"{no_churn_predicho:,}")
col4.metric("Probabilidad promedio", f"{probabilidad_promedio:.1%}")

st.subheader("Distribucion de resultados")
counts = results_df["estado_churn"].value_counts().reindex(
    ["No churn", "Churn"],
    fill_value=0,
)
st.bar_chart(counts)

if feature_importances is not None:
    st.write("### Feature Importance")
    fig = plot_feature_importance(feature_importances)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Ver importancia completa del modelo"):
        st.dataframe(feature_importances, width="stretch", hide_index=True)

st.subheader("Predicciones")
ordered_columns = [
    "estado_churn",
    "probabilidad_churn_pct",
    "prediccion_churn",
] + [column for column in input_columns if column not in {"PERIODO_BAJA"}]

st.dataframe(
    results_df.sort_values("probabilidad_churn", ascending=False)[ordered_columns],
    width="stretch",
    hide_index=True,
)

download_option = st.selectbox(
    "Descargar",
    ["Todas las predicciones", "Solo churn", "Solo no churn"],
)
format_option = st.selectbox("Formato", ["CSV", "Excel"])

if download_option == "Solo churn":
    output_df = results_df[results_df["estado_churn"] == "Churn"]
    base_name = "clientes_churn"
elif download_option == "Solo no churn":
    output_df = results_df[results_df["estado_churn"] == "No churn"]
    base_name = "clientes_no_churn"
else:
    output_df = results_df
    base_name = "predicciones_churn"

if format_option == "CSV":
    file_data = dataframe_to_csv_bytes(output_df)
    file_name = f"{base_name}.csv"
    mime = "text/csv"
else:
    file_data = dataframe_to_excel_bytes(output_df)
    file_name = f"{base_name}.xlsx"
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.download_button(
    label=f"Descargar {download_option.lower()}",
    data=file_data,
    file_name=file_name,
    mime=mime,
)
