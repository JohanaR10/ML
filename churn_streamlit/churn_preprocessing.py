from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


TARGET_COLUMN = "churn_target"


DROP_COLUMNS = [
    "SEC_CUOTA",
    "VAC",
    "Segmento_Alta",
    "ID",
    "id_articulo",
    "nom_punto",
    "Nom_Oficina",
    "nom_agente",
    "Id_Oficina",
    "id_agente",
    "TIP_CHURN",
    "TIP_ACCION",
    "CANAL_VENTA",
    "SEGMENTO",
    "PERIODO_BAJA_numeric",
    "PERIODO_BAJA",
    "Fecha_Venta",
    "Periodo",
    "descripcion_articulo",
    "Regional",
    "departamento",
    "cod_plantarif",
    "des_fabricante",
    "id_vendedor",
    "id_punto",
]


BINARY_FLAG_COLUMNS = [
    "Disney_Incluido",
    "Disney_Activo",
    "Netflix_Incluido",
    "Netflix_Activo",
    "Clientes_domiciliados",
    "Domiciliacion_Pago",
]


RENTA_CATEGORY_COLUMNS = [
    "categoria_renta_RENTA_MAYOR_70K",
    "categoria_renta_RENTA_MEDIA_50K_70k",
    "categoria_renta_RENTA_MEDIA_40K_50K",
    "categoria_renta_RENTA_MEDIA_30K_40K",
    "categoria_renta_RENTA_BAJA_MENOR_30K",
]


def build_target(df: pd.DataFrame) -> pd.Series:
    periodo_baja = pd.to_numeric(df.get("PERIODO_BAJA"), errors="coerce")
    return periodo_baja.notna().astype(int)


def _normalize_binary_flag(value: object) -> int:
    value_text = str(value).upper().strip()
    if value_text in {"0", "0.0", "NO", "N", "FALSE", "NAN", "NONE", ""}:
        return 0
    if value_text in {"1", "1.0", "SI", "S", "TRUE"}:
        return 1
    return 1


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.upper().str.strip()
        df[col] = df[col].replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
    return df


def _add_renta_categories(df: pd.DataFrame) -> pd.DataFrame:
    if "cb_coniva" not in df.columns:
        return df

    renta = pd.to_numeric(df["cb_coniva"], errors="coerce").fillna(33000)
    df["categoria_renta_RENTA_MAYOR_70K"] = (renta > 70000).astype(int)
    df["categoria_renta_RENTA_MEDIA_50K_70k"] = ((renta >= 50000) & (renta <= 70000)).astype(int)
    df["categoria_renta_RENTA_MEDIA_40K_50K"] = ((renta >= 40000) & (renta < 50000)).astype(int)
    df["categoria_renta_RENTA_MEDIA_30K_40K"] = ((renta >= 30000) & (renta < 40000)).astype(int)
    df["categoria_renta_RENTA_BAJA_MENOR_30K"] = (renta < 30000).astype(int)
    return df


def _apply_ordinal_mappings(
    df: pd.DataFrame, ordinal_mappings: Mapping[str, Mapping[str, int]] | None
) -> pd.DataFrame:
    if not ordinal_mappings:
        return df

    for col_name, mapping_dict in ordinal_mappings.items():
        if col_name not in df.columns:
            continue

        normalized_mapping = {
            str(key).upper().strip(): value for key, value in mapping_dict.items()
        }
        unknown_value = normalized_mapping.get("UNKNOWN", -1)
        df[col_name] = df[col_name].astype(str).str.upper().str.strip()
        df[col_name] = df[col_name].map(normalized_mapping).fillna(unknown_value)
    return df


def preprocess_raw_data(
    df: pd.DataFrame,
    training: bool,
    ordinal_mappings: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.replace("?", np.nan)

    if training:
        if "PERIODO_BAJA" not in df.columns:
            raise ValueError("El archivo de entrenamiento debe incluir la columna PERIODO_BAJA.")
        df[TARGET_COLUMN] = build_target(df)

    if "Periodo" in df.columns:
        df["Periodo"] = pd.to_numeric(df["Periodo"], errors="coerce")

    if "Fecha_Venta" in df.columns:
        df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], errors="coerce", dayfirst=True)

    if "Ind_Portado" in df.columns:
        df["Ind_Portado"] = (
            df["Ind_Portado"].astype(str).str.upper().str.strip().eq("SI").astype(int)
        )

    if "Flag_Trafico_Saliente" in df.columns:
        trafico = df["Flag_Trafico_Saliente"].astype(str).str.upper().str.strip()
        df["Flag_Trafico_Saliente"] = trafico.replace(
            {"SI": 1, "NO": 0, "CLIENTE INTERNO": 1, "SIN INFORMACION": 0}
        )
        df["Flag_Trafico_Saliente"] = pd.to_numeric(
            df["Flag_Trafico_Saliente"], errors="coerce"
        ).fillna(0)

    for col in BINARY_FLAG_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_binary_flag).astype(int)

    if "FLAGENTREGASIM" in df.columns:
        entrega = df["FLAGENTREGASIM"].astype(str).str.upper().str.strip()
        df["FLAGENTREGASIM"] = entrega.replace({"SI": 1, "NO": 0})
        df["FLAGENTREGASIM"] = pd.to_numeric(df["FLAGENTREGASIM"], errors="coerce")
        if "Flag_Trafico_Saliente" in df.columns:
            df["FLAGENTREGASIM"] = np.where(
                df["FLAGENTREGASIM"].isna(),
                np.where(df["Flag_Trafico_Saliente"] == 1, 1, 0),
                df["FLAGENTREGASIM"],
            )
        else:
            df["FLAGENTREGASIM"] = df["FLAGENTREGASIM"].fillna(0)

    if "cb_coniva" in df.columns:
        df["cb_coniva"] = pd.to_numeric(df["cb_coniva"], errors="coerce").fillna(33000)
        df = _add_renta_categories(df)

    if "id_vendedor" in df.columns:
        df["id_vendedor"] = pd.to_numeric(df["id_vendedor"], errors="coerce")

    if "id_punto" in df.columns:
        df["id_punto"] = pd.to_numeric(df["id_punto"], errors="coerce")

    target = df[TARGET_COLUMN] if training else None
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    if training:
        df[TARGET_COLUMN] = target

    df = _normalize_text_columns(df)
    df = _apply_ordinal_mappings(df, ordinal_mappings)
    if training:
        df = df.drop_duplicates()
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo {TARGET_COLUMN}.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def align_prediction_columns(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[list(feature_columns)]
