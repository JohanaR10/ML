# App de prediccion de churn

Aplicacion Streamlit para predecir churn de clientes telco usando el modelo:

```text
models/best_churn_model_decision_tree_v2.joblib
```

## Ejecutar en Streamlit Community Cloud

Configura la app con:

- Repository: `JohanaR10/ML`
- Branch: `main`
- Main file path: `churn_streamlit/app_streamlit.py`
- Python: `3.12`

## Ejecutar localmente

```powershell
cd "C:\Users\johan\OneDrive\Documentos\Modelado-proyecto final\churn_streamlit"
python -m pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Archivos necesarios

```text
app_streamlit.py
churn_preprocessing.py
requirements.txt
models/best_churn_model_decision_tree_v2.joblib
models/ordinal_mappings.json
altas_Bajas_target_checkbox.csv
altas_Bajas_notarget_checkbox.csv
```

Los dos CSV incluidos permiten probar la app desde el checkbox `Usar datos incluidos`:

- `Con variable objetivo (PERIODO_BAJA)`: permite mostrar `churn_real`, matriz de confusion y reporte de clasificacion.
- `Sin variable objetivo`: solo genera predicciones.

La app detecta internamente el encoding entre `utf-8`, `utf-16` y `latin1`. El separador esperado por defecto es `|`.
