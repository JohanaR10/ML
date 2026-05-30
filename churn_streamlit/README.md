# App de prediccion de churn

Esta carpeta separa el proyecto en dos pasos:

1. Entrenar y guardar el modelo con el CSV historico.
2. Abrir una app web en Streamlit para subir clientes nuevos y calcular su probabilidad de churn.

La app actual esta adaptada al modelo:

```text
models/best_churn_model_decision_tree_v2.joblib
```

Si ese archivo no existe, la app usa como respaldo `models/best_churn_model_decision_tree.joblib`.

Tambien carga `models/ordinal_mappings.json` para convertir las categorias del CSV original al formato numerico que espera el modelo.

## 1. Instalar librerias

```powershell
cd "C:\Users\johan\OneDrive\Documentos\Modelado-proyecto final\churn_streamlit"
python -m pip install -r requirements.txt
```

En este computador `python` no esta en el PATH de PowerShell. Por eso tambien deje este comando listo:

```powershell
.\install_deps.ps1
```

## 2. Entrenar el modelo

Usa el archivo historico que usaban en Colab, por ejemplo `Altas_Bajas.csv`.

```powershell
python train_churn_model.py --csv "C:\ruta\a\Altas_Bajas.csv"
```

O con el script listo para este equipo:

```powershell
.\train_model.ps1 -CsvPath "C:\ruta\a\Altas_Bajas.csv"
```

El notebook leia el archivo con `encoding='utf-16'` y `sep='|'`, por eso esos son los valores por defecto.
Si tu CSV usa coma y UTF-8:

```powershell
python train_churn_model.py --csv "C:\ruta\a\Altas_Bajas.csv" --sep "," --encoding utf-8
.\train_model.ps1 -CsvPath "C:\ruta\a\Altas_Bajas.csv" -Sep "," -Encoding utf-8
```

Al finalizar se crea:

```text
models/churn_model.joblib
```

## 3. Abrir la web

```powershell
streamlit run app_streamlit.py
```

O con el script listo para este equipo:

```powershell
.\run_app.ps1
```

Streamlit abrira una pagina local. Ahi subes el CSV de clientes a evaluar y descargas `predicciones_churn.csv`.

La app incluye un checkbox para probar directamente con datos incluidos:

```text
altas_Bajas_target_checkbox.csv
altas_Bajas_notarget_checkbox.csv
```

El primero trae `PERIODO_BAJA` para evaluar contra `churn_real`; el segundo no trae variable objetivo y solo genera predicciones.

Si subes un archivo manualmente, manten por defecto:

- Separador: `|`

La app detecta internamente el encoding entre `utf-8`, `utf-16` y `latin1`, por eso no muestra un selector de encoding en la interfaz.

## Salidas de la app

Despues de ejecutar el analisis, la app muestra:

- Vista previa del archivo cargado.
- Conteo de clientes evaluados, churn predicho, no churn predicho y probabilidad promedio.
- Distribucion de resultados.
- Matriz de confusion y reporte de clasificacion si el archivo trae `PERIODO_BAJA`.
- Top 10 variables mas influyentes del arbol de decision.
- Tabla de predicciones ordenada por probabilidad de churn.
- Descarga en CSV o Excel de todas las predicciones, solo churn o solo no churn.

## Columnas importantes

El archivo historico de entrenamiento debe traer `PERIODO_BAJA`, porque de ahi se construye la variable objetivo:

- `PERIODO_BAJA` con valor numerico significa churn = 1.
- `PERIODO_BAJA` vacio, `?` o no numerico significa churn = 0.

El archivo de prediccion no necesita `PERIODO_BAJA`. Si viene incluida, la app la ignora para evitar fuga de informacion.

## Usar el modelo exportado desde Colab

La app tambien puede usar directamente un `.joblib` descargado de Colab dentro de `models/`.
Para que acepte un CSV crudo como `Altas_Bajas.csv`, tambien debe existir:

```text
models/ordinal_mappings.json
```

Ese archivo contiene los diccionarios `ordinal_mappings` del notebook y permite convertir categorias de texto a los codigos numericos usados durante entrenamiento.

En este proyecto ya quedo creado desde `Proyecto_Churn_Clientes_Movistar_v23.ipynb`.

## Archivos de prueba

Para probar la app con los datos incluidos usa el checkbox `Usar datos incluidos` y selecciona:

- `Con variable objetivo (PERIODO_BAJA)`
- `Sin variable objetivo`
