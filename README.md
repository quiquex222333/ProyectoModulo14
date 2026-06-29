# Predicción de Precios de Propiedades Inmobiliarias — Regresión Avanzada

## Índice

- [Introducción](#introducción)
  - [Métodos Utilizados](#métodos-utilizados)
  - [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
  - [Requisitos del Sistema](#requisitos-del-sistema)
  - [Requisitos Previos](#requisitos-previos)
  - [Cómo Ejecutar](#cómo-ejecutar)
  - [Estructura de Directorios](#estructura-de-directorios)
  - [Solución de Problemas](#solución-de-problemas)
- [Declaración del Problema](#declaración-del-problema)
  - [Objetivo Comercial](#objetivo-comercial)
  - [Preparación de Datos](#preparación-de-datos)
  - [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
  - [Conclusiones](#conclusiones)

---

## Introducción

Este proyecto aplica regresión avanzada con regularización (Ridge, Lasso y ElasticNet) para predecir el precio de venta de propiedades inmobiliarias en el mercado australiano. El módulo de valuación forma parte de una plataforma de gestión de alquileres tipo Airbnb Clone cuyo panel administrativo requiere estimaciones de precio confiables para alimentar decisiones de inversión y módulos de segmentación de mercado basados en K-Means.

El dataset utilizado es el **Ames Housing Dataset** (mercado australiano): 1,460 registros y 81 variables que describen características físicas, de calidad, de ubicación y de condición de cada propiedad. La variable objetivo es `SalePrice`.

**Integrantes del Grupo:**
- Rios Nuñez David Samuel
- Rivera Quisberth Juan Enrique
- Terceros Beltrán Oscar Alvaro
- Torrez Azuga Marcelo

---

### Métodos Utilizados

- Análisis Exploratorio de Datos (EDA): univariable, bivariable y multivariable
- Imputación de valores nulos con lógica semántica del dominio
- Detección y remoción de outliers extremos
- Feature Engineering: variables derivadas (TotalSF, TotalBath, HouseAge, RemodAge, HasGarage, HasPool, HasFireplace)
- Encoding ordinal para variables de calidad; One-Hot encoding para variables nominales
- Selección de variables con RFE (Recursive Feature Elimination)
- Control de multicolinealidad con VIF (Variance Inflation Factor)
- Modelos de regresión regularizada: Ridge (L2), Lasso (L1), ElasticNet (L1+L2)
- Análisis de residuos y estadístico Durbin-Watson

---

### Tecnologías

| Librería | Versión mínima | Uso |
|---|---|---|
| Python | 3.9+ | Lenguaje base |
| pandas | 1.3+ | Manipulación de datos |
| numpy | 1.21+ | Operaciones numéricas |
| matplotlib | 3.4+ | Visualizaciones |
| seaborn | 0.11+ | Visualizaciones estadísticas |
| scipy | 1.7+ | Q-Q plot (análisis de residuos) |
| scikit-learn | 1.0+ | RFE, StandardScaler, métricas |
| statsmodels | 0.13+ | VIF, Durbin-Watson |
| pycaret | 3.0+ | Setup, Ridge, Lasso, ElasticNet |
| jupyter | cualquiera | Ejecución del notebook |

---

## Descarga y Configuración

### Requisitos del Sistema

- **Sistema Operativo:** Windows 10+, macOS 10.15+, o Linux (Ubuntu 20.04+)
- **Python:** versión **3.9, 3.10 u 3.11** (PyCaret 3.x no soporta Python 3.12+)
- **RAM:** mínimo 4 GB (recomendado 8 GB)
- **Espacio en disco:** ~2 GB (para dependencias de PyCaret)

> ⚠️ **Importante:** PyCaret 3.x requiere Python 3.9, 3.10 u 3.11. No instalar con Python 3.12 o superior.

---

### Requisitos Previos

#### Opción A — Anaconda (recomendado)

1. Instalar **Anaconda**: https://docs.anaconda.com/anaconda/install/index.html

2. Crear un entorno virtual con Python 3.10:
```bash
conda create -n regresion_env python=3.10
conda activate regresion_env
```

3. Instalar todas las dependencias:
```bash
pip install pycaret==3.3.2 scikit-learn statsmodels seaborn scipy jupyter
```

4. Verificar la instalación:
```bash
python -c "import pycaret; print('PyCaret', pycaret.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "import statsmodels; print('statsmodels', statsmodels.__version__)"
```

#### Opción B — pip (entorno virtual estándar)

1. Verificar que Python 3.10 o 3.11 esté instalado:
```bash
python3 --version
```

2. Crear un entorno virtual:
```bash
python3 -m venv regresion_env
# macOS/Linux:
source regresion_env/bin/activate
# Windows:
regresion_env\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install pycaret==3.3.2 scikit-learn statsmodels seaborn scipy jupyter
```

> 💡 **Nota sobre PyCaret:** La instalación puede tardar 5–10 minutos porque instala muchas dependencias (lightgbm, xgboost, etc.). Esperar a que finalice completamente.

---

### Cómo Ejecutar

1. Abrir una terminal (macOS/Linux) o Anaconda Prompt (Windows)

2. Activar el entorno virtual:
```bash
# Anaconda:
conda activate regresion_env
# pip venv (macOS/Linux):
source regresion_env/bin/activate
# pip venv (Windows):
regresion_env\Scripts\activate
```

3. Clonar el repositorio:
```bash
git clone <GITHUB_REPO_URL>
cd SubProyecto_Regresion
```

4. Verificar que el dataset esté en su lugar:
```bash
# Debe existir este archivo:
ls _data/dataset.csv
```

5. Abrir el notebook:
```bash
jupyter notebook "MFSDv1p1_Regresion Avanzada con PyCaret.ipynb"
```

6. En el navegador, ejecutar todas las celdas en orden:
   - **Kernel → Restart & Run All**
   - O presionar `Shift + Enter` en cada celda de arriba hacia abajo

> ⚠️ **No saltar celdas.** Cada celda depende de la ejecución de las anteriores. Ejecutar siempre de forma secuencial desde la primera celda.

---

### Estructura de Directorios

El repositorio debe tener exactamente esta estructura para que el notebook funcione:

```
SubProyecto_Regresion/
│
├── MFSDv1p1_Regresion Avanzada con PyCaret.ipynb   ← notebook principal
├── README.md                                        ← este archivo
│
└── _data/
    ├── dataset.csv          ← dataset Ames Housing (1,460 filas × 81 cols)
    └── data_description.txt ← descripción de cada columna del dataset
```

> ⚠️ **Ruta del dataset:** El notebook carga los datos con la ruta relativa `_data/dataset.csv`. Para que funcione correctamente, Jupyter debe ser abierto **desde la carpeta `SubProyecto_Regresion/`**, no desde una carpeta superior.

**Correcto:**
```bash
cd SubProyecto_Regresion/
jupyter notebook "MFSDv1p1_Regresion Avanzada con PyCaret.ipynb"
```

**Incorrecto** (causará `FileNotFoundError`):
```bash
cd ..
jupyter notebook "SubProyecto_Regresion/MFSDv1p1_Regresion Avanzada con PyCaret.ipynb"
```

---

### Solución de Problemas

| Error | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'pycaret'` | PyCaret no instalado en el entorno activo | Activar el entorno virtual correcto y ejecutar `pip install pycaret==3.3.2` |
| `FileNotFoundError: _data/dataset.csv` | Jupyter no fue abierto desde la carpeta del proyecto | Abrir la terminal **dentro** de `SubProyecto_Regresion/` antes de ejecutar `jupyter notebook` |
| `ModuleNotFoundError: No module named 'sklearn'` | scikit-learn no instalado | `pip install scikit-learn` |
| `ModuleNotFoundError: No module named 'statsmodels'` | statsmodels no instalado | `pip install statsmodels` |
| Kernel muerto / crash al importar PyCaret | Python 3.12+ incompatible | Usar Python 3.10 o 3.11 |
| `ValueError: could not convert string to float` | Celda de encoding no ejecutada | Reiniciar kernel y ejecutar **todas** las celdas desde el inicio |
| Las celdas de PyCaret no muestran output | `verbose=False` configurado | Comportamiento esperado; las métricas se imprimen manualmente |

---

## Declaración del Problema

Una empresa de análisis inmobiliario busca ingresar al mercado australiano de bienes raíces adquiriendo propiedades cuyo precio de mercado está por debajo de su valor real. Se cuenta con un dataset de ventas de casas en Australia y se debe construir un modelo de regresión regularizada que permita responder:

1. ¿Qué variables son más significativas para predecir el precio de venta (`SalePrice`)?
2. ¿Qué tan bien describen esas variables el precio de la propiedad?
3. ¿Cuál es el valor óptimo de lambda (alpha) para Ridge y Lasso?

### Objetivo Comercial

Modelar el precio de venta usando las variables independientes disponibles para que la gerencia pueda: identificar propiedades subvaloradas en el mercado, entender qué factores (tamaño, calidad de construcción, vecindario, antigüedad) determinan el precio, y alimentar el módulo de segmentación de mercado K-Means de la plataforma Airbnb Clone con estimaciones de precio confiables.

---

### Preparación de Datos

1. Limpieza de Datos y Análisis de Datos Faltantes: eliminación de columnas con >40% nulos; imputación semántica del resto (NA categórico → "None", NA numérico → 0, LotFrontage → mediana por vecindario).
2. Análisis y Tratamiento de Valores Atípicos: remoción de propiedades con GrLivArea > 4,000 sqft y SalePrice < $300,000 (outliers confirmados del dataset Ames).
3. Derivación de Variables: TotalSF (área total), TotalBath (baños totales ponderados), HouseAge, RemodAge, HasGarage, HasPool, HasFireplace.
4. Análisis Univariable: distribuciones de variables numéricas y frecuencias de categóricas.
5. Análisis Bivariable: correlaciones con SalePrice, scatterplots, boxplots por calidad y tipo de inmueble.
6. Análisis Multivariable: heatmap de correlación entre top-15 variables predictoras.

### Construcción y Evaluación del Modelo

1. División de datos de entrenamiento y prueba (80/20, random_state=42).
2. Escalado de Características: StandardScaler aplicado sobre entrenamiento, transformado en test.
3. Ingeniería y Selección de Características usando RFE (top-20 features) y VIF (eliminar VIF > 5) para reducir multicolinealidad.
4. Regresión Lineal usando PyCaret: configuración del entorno con pycaret.regression.setup().
5. Modelos de Regularización: Ridge, Lasso y ElasticNet entrenados y optimizados con tune_model() de PyCaret.
6. Análisis de Residuos: plots de residuos vs predichos, histograma de residuos, Q-Q plot.
7. Evaluación y Valoración del Modelo: R2 Train/Test, RMSE Test, estadístico Durbin-Watson.
8. Predicción: comparación real vs predicho en el test set.
9. Conclusión y Análisis Final: comparación de los tres modelos y selección del mejor.

---

### Conclusiones

**R2 Score para Ridge regression: 0.##**
**R2 Score para Lasso regression: 0.##**
**R2 Score para ElasticNet regression: 0.##**

#### Ridge Regression (Segun PyCaret)
* **Optimal Lambda Value:** ##
* **R2 Score Train:** 0.##
* **R2 Test Score:** 0.##
* **RMSE Test:** ##.##

#### Lasso Regression (Segun PyCaret)
* **Optimal Lambda Value:** 0.####
* **R2 Score Train:** 0.##
* **R2 Test Score:** 0.##
* **RMSE Test:** ##.##

#### ElasticNet Regression (Segun PyCaret)
* **Optimal Lambda Value:** 0.####
* **L1 Ratio:** 0.##
* **R2 Score Train:** 0.##
* **R2 Test Score:** 0.##
* **RMSE Test:** ##.##

#### Las Variables Más Significativas Son:
* ...
* ...
* ...
* ...

> Completar esta sección ejecutando el notebook completo y copiando los valores impresos por la celda de **Comparación de Modelos** e **Importancia de Variables**.
