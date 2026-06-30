# Regresión Avanzada con PyCaret: Predicción de Precios de Bienes Raíces

Este repositorio contiene el desarrollo del proyecto final para la materia **AI & MLOps** del programa **MFSD** en la **Universidad Católica Boliviana "San Pablo"**. El objetivo es construir, evaluar e interpretar modelos de regresión avanzada para predecir precios de viviendas residenciales utilizando **PyCaret** como herramienta principal.

> **Nota metodológica:** el caso de negocio plantea la expansión de Surprise Housing al mercado australiano. Sin embargo, el dataset académico utilizado corresponde al conjunto Ames Housing. Para fines del proyecto, se mantiene el contexto comercial propuesto y se utiliza el dataset como base tabular estructurada para demostrar el flujo completo de regresión avanzada: limpieza, EDA, ingeniería de características, modelado, evaluación e interpretabilidad.

---

## Estructura del Repositorio

```text
├── _data/
│   ├── dataset.csv                  # Dataset histórico original (1460 filas, 81 columnas)
│   ├── data_description.txt         # Diccionario de variables detallado
│   └── cleaned_dataset.csv          # Dataset procesado e imputado listo para modelar
├── MFSDv1p1_Regresion_Avanzada_con_PyCaret_CORREGIDO.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

> El entorno virtual `.venv/` no debe subirse al repositorio. Se debe crear localmente siguiendo las instrucciones de instalación.

---

## 1. Descripción del Proyecto y Contexto de Negocio

La firma estadounidense de corretaje inmobiliario **Surprise Housing** busca adquirir viviendas subvaloradas, remodelarlas y venderlas posteriormente a valor de mercado. Para reducir el riesgo financiero, necesita un modelo cuantitativo que permita estimar el precio real de una vivienda a partir de sus características físicas, ubicación, calidad y condición de venta.

### Objetivos Principales

- Identificar las variables físicas, cualitativas y geográficas más influyentes en el precio de venta.
- Medir qué tan bien las características del inmueble explican el valor final de venta.
- Ajustar modelos regularizados Ridge, Lasso y ElasticNet para controlar sobreajuste y multicolinealidad.
- Comparar distintos modelos de regresión mediante PyCaret y seleccionar el mejor según métricas de desempeño.
- Interpretar el modelo final en términos simples para apoyar decisiones comerciales.

---

## 2. Tecnologías Empleadas

- Python 3.10
- PyCaret 3.3.0
- Pandas 2.1.4
- NumPy 1.26.4
- Scikit-learn 1.4.2
- Matplotlib 3.7.5
- Seaborn 0.13.2
- Jupyter Notebook
- SHAP

---

## 3. Preparación de Datos e Ingeniería de Características

El notebook realiza los siguientes pasos principales:

1. **Eliminación del identificador:** se elimina `Id`, ya que no aporta información predictiva.
2. **Control de calidad inicial:** se genera una tabla de valores faltantes antes de la limpieza y se revisan duplicados.
3. **Tratamiento de valores faltantes:**
   - En variables categóricas como `PoolQC`, `Fence`, `Alley`, `BsmtQual`, `GarageQual`, etc., el valor faltante representa ausencia real del atributo.
   - Se reemplaza por `NoFeature` para preservar significado comercial y evitar que Pandas lo interprete nuevamente como nulo al recargar el CSV.
   - `LotFrontage` se imputa con la mediana por `Neighborhood`.
   - `GarageYrBlt` se imputa con `YearBuilt`.
   - `MasVnrArea` se imputa con `0`.
   - `MasVnrType` se imputa con `NoFeature`.
   - `Electrical` se imputa con la moda.
4. **Ingeniería de características:**
   - `HouseAge`: antigüedad de la vivienda al momento de venta.
   - `RemodAge`: años desde la última remodelación.
   - `TotalSF`: superficie total útil construida.
   - `TotalBath`: total ponderado de baños completos y medios baños.
   - `HasPool`: indicador binario de piscina.
5. **Transformación logarítmica:** se aplica `np.log1p()` sobre `SalePrice` para reducir el sesgo de la distribución.
6. **Verificación posterior:** se valida que el dataset limpio no conserve valores faltantes, incluso después de recargar `cleaned_dataset.csv`.

---

## 4. Análisis Exploratorio de Datos

El EDA fue reorganizado en celdas separadas para mejorar la lectura del notebook. Se incluyen:

- Distribución de `SalePrice` en escala original y logarítmica.
- Matriz de correlación de las variables numéricas más relacionadas con el precio.
- Relación entre `OverallQual` y `SalePrice`.
- Análisis visual de outliers usando `GrLivArea` vs. `SalePrice`.
- Estadísticas descriptivas de variables clave y nuevas métricas creadas.

Los outliers se analizan visualmente, pero no se eliminan automáticamente, porque pueden representar propiedades reales de alto valor. Para reducir su influencia se usa transformación logarítmica, escalado robusto y modelos regularizados/no lineales.

---

## 5. Modelado y Evaluación

El experimento se configura con PyCaret usando:

- División 70% entrenamiento y 30% holdout.
- Normalización robusta.
- Remoción de multicolinealidad.
- Codificación automática de variables categóricas.
- Validación cruzada para comparar modelos.

### Modelos Regularizados Ajustados

| Modelo | Hiperparámetro óptimo | R² Log-Scale (CV Mean) | MAE Log-Scale (CV Mean) |
| :--- | :---: | :---: | :---: |
| Ridge Regression | `alpha = 20.0` | `0.8420` | `0.0931` |
| Lasso Regression | `alpha = 0.0005` | `0.8514` | `0.0869` |
| ElasticNet Regression | `alpha = 0.001`, `l1_ratio = 0.8` | `0.8440` | `0.0905` |

### Modelo Final Seleccionado

La comparación de modelos de PyCaret seleccionó **Gradient Boosting Regressor** como el mejor modelo general en términos de precisión. Evaluado sobre holdout y convertido de vuelta a la escala monetaria original, se obtuvieron los siguientes resultados:

- **R² Score:** `0.9097`
- **MAE:** `$15,225.79 AUD`
- **RMSE:** `$25,068.79 AUD`
- **MAPE:** `8.90%`

---

## 6. Interpretabilidad del Modelo

El notebook incluye un gráfico de importancia de variables usando `plot_model(..., plot='feature')`. Las variables más relevantes para la interpretación comercial son:

1. `TotalSF`: superficie útil construida.
2. `OverallQual`: calidad general de materiales y acabados.
3. `HouseAge` y `RemodAge`: antigüedad y efecto de remodelación.
4. `Neighborhood`: ubicación o zona de la vivienda.

Estas variables permiten traducir el resultado técnico del modelo en recomendaciones para adquisición y remodelación de propiedades.

---

## 7. Instrucciones de Ejecución Local

### Requisitos Previos

- Python 3.10 instalado.
- En macOS, instalar `libomp` si PyCaret o LightGBM lo requieren:

```bash
brew install libomp
```

### Crear entorno virtual

Con `uv`:

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Con `venv` tradicional:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias

Con `uv`:

```bash
uv pip install -r requirements.txt
```

Con `pip`:

```bash
pip install -r requirements.txt
```

### Ejecutar el notebook

```bash
jupyter notebook "MFSDv1p1_Regresion_Avanzada_con_PyCaret_CORREGIDO.ipynb"
```

---

## 8. Entregables del Proyecto

- Notebook Jupyter documentado y listo para ejecución.
- Dataset original.
- Dataset limpio generado por el notebook.
- Diccionario de datos.
- README del proyecto.
- Archivo `requirements.txt` para reproducibilidad.
- Archivo `.gitignore` para evitar subir `.venv`, cachés y archivos temporales.

---

## Integrantes del Grupo

- Rios Nuñez David Samuel
- Rivera Quisberth Juan Enrique
- Terceros Beltran Oscar Alvaro
- Torrez Azuga Marcelo
