from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_preprocessing import align_prediction_columns, preprocess_raw_data


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
MAPPINGS_PATH = MODELS_DIR / "ordinal_mappings.json"


st.set_page_config(page_title="Prediccion de Churn Movistar", layout="wide")


@st.cache_resource
def load_model():
    model_files = [MODEL_PATH] if MODEL_PATH.exists() else sorted(MODELS_DIR.glob("*.joblib"))
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


def read_uploaded_csv(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=sep, encoding=encoding)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


artifact = load_model()
ordinal_mappings = load_ordinal_mappings()

st.title("Prediccion de Churn")
st.caption("Sube un CSV de clientes y recibe la probabilidad estimada de abandono.")

if artifact is None:
    st.error(
        "No encontre el modelo entrenado. Primero ejecuta el script de entrenamiento "
        "para crear models/churn_model.joblib."
    )
    st.code(
        "python train_churn_model.py --csv \"C:\\ruta\\a\\Altas_Bajas.csv\"",
        language="powershell",
    )
    st.stop()

st.success(f"Modelo cargado: {artifact['model_path']}")
if ordinal_mappings:
    st.caption(f"Mapeos cargados: {MAPPINGS_PATH}")

with st.sidebar:
    st.header("Archivo")
    sep = st.text_input("Separador", value="|", max_chars=3)
    encoding = st.selectbox("Encoding", ["utf-16", "utf-8", "latin1"], index=0)
    threshold = st.slider(
        "Umbral de churn",
        min_value=0.05,
        max_value=0.95,
        value=float(artifact.get("threshold", 0.5)),
        step=0.05,
    )

uploaded_file = st.file_uploader("CSV de clientes", type=["csv"])

if uploaded_file is None:
    st.info("Carga un archivo CSV para calcular el riesgo de churn.")
    st.stop()

try:
    raw_df = read_uploaded_csv(uploaded_file, sep=sep, encoding=encoding)
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
                "El modelo descargado de Colab espera estas columnas ya codificadas como numeros. "
                "El archivo subido todavia tiene valores de texto o categorias no codificadas en: "
                + ", ".join(columns_with_missing)
            )
    probabilities = artifact["pipeline"].predict_proba(X)[:, 1]
except Exception as exc:
    st.error("No pude procesar el archivo. Revisa separador, encoding y columnas.")
    st.exception(exc)
    st.stop()

result_df = raw_df.copy()
result_df["probabilidad_churn"] = probabilities
result_df["prediccion_churn"] = (probabilities >= threshold).astype(int)
result_df["riesgo_churn"] = pd.cut(
    result_df["probabilidad_churn"],
    bins=[-0.01, 0.35, 0.65, 1.0],
    labels=["Bajo", "Medio", "Alto"],
)

total_clientes = len(result_df)
clientes_en_riesgo = int(result_df["prediccion_churn"].sum())
riesgo_promedio = float(result_df["probabilidad_churn"].mean())

col1, col2, col3 = st.columns(3)
col1.metric("Clientes evaluados", f"{total_clientes:,}")
col2.metric("Clientes con churn probable", f"{clientes_en_riesgo:,}")
col3.metric("Probabilidad promedio", f"{riesgo_promedio:.1%}")

st.subheader("Resultados")
st.dataframe(
    result_df.sort_values("probabilidad_churn", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Descargar resultados",
    data=dataframe_to_csv_bytes(result_df),
    file_name="predicciones_churn.csv",
    mime="text/csv",
)

metrics = artifact.get("metrics")
if metrics:
    with st.expander("Metricas del modelo"):
        st.write(f"ROC AUC: {metrics.get('roc_auc', 0):.4f}")
