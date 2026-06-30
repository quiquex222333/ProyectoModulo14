# Regresión Avanzada con PyCaret: Predicción de Precios de Bienes Raíces en Australia

Este repositorio contiene el desarrollo del proyecto final para la materia **AI
& MLOps** del programa **MFSD** en la **Universidad Católica Boliviana "San
Pablo"**. El objetivo del proyecto es construir y evaluar modelos de regresión
avanzados para predecir precios de viviendas residenciales en el mercado de
Australia utilizando la librería **PyCaret** como herramienta principal.

---

## Estructura del Repositorio

```text
├── _data/
│   ├── dataset.csv                  # Dataset histórico original (1460 filas, 81 columnas)
│   ├── data_description.txt         # Diccionario de variables detallado
│   └── cleaned_dataset.csv          # Dataset procesado e imputado listo para modelar
├── MFSDv1p1_Regresion Avanzada con PyCaret.ipynb   # Jupyter Notebook principal (ejecutado y documentado)
├── README.md                        # Documentación general del repositorio (este archivo)
└── .venv/                           # Entorno virtual de Python 3.10 (gestionado localmente)
```

---

## 1. Descripción del Proyecto y Contexto de Negocio

La firma estadounidense de corretaje inmobiliario **Surprise Housing** ha
decidido expandirse ingresando al mercado de bienes raíces de Australia. El
modelo comercial de la empresa se basa en:

1. **Adquirir** viviendas que se encuentran listadas a precios por debajo de su
   valor intrínseco de mercado.
2. **Remodelar** aspectos clave del diseño o de su estructura física.
3. **Vender** las viviendas remodeladas a valor de mercado para obtener un
   margen de ganancia comercial.

Para minimizar la incertidumbre financiera y maximizar el retorno de inversión
(ROI), Surprise Housing necesita un **modelo cuantitativo de regresión** que sea
capaz de estimar de forma precisa el valor real de mercado de una propiedad
residencial a partir de sus atributos físicos y de diseño.

### Objetivos Principales

- Identificar qué variables físicas, cualitativas y geográficas son las más
  influyentes para determinar el precio de venta de las viviendas en Australia.
- Medir qué tan bien estas características explican el valor final de la
  vivienda ($R^2$ y error promedio en dólares).
- Estimar los parámetros óptimos de regularización ($\alpha$ / $\lambda$) para
  Ridge, Lasso y ElasticNet a fin de controlar el sobreajuste (overfitting) y
  mitigar la multicolinealidad.

---

## 2. Tecnologías Empleadas

El desarrollo se fundamenta en el ecosistema científico de Python:

- **Python 3.10** (como lenguaje base por compatibilidad con PyCaret).
- **PyCaret 3.3.0** (librería de MLOps y AutoML para la automatización,
  comparación y sintonización de experimentos).
- **Pandas 2.1.4 y NumPy 1.26.4** (para la manipulación, limpieza de datos e
  ingeniería de características).
- **Scikit-learn 1.4.2** (para cómputo de métricas y regularizaciones lineales
  Ridge y Lasso).
- **Seaborn 0.13.2 y Matplotlib 3.7.5** (para visualizaciones y gráficos
  estadísticos de análisis exploratorio).
- **Jupyter y NbConvert** (para el desarrollo interactivo y ejecución
  automatizada del cuaderno).
- **uv 0.7.21** (instalador de paquetes de Python ultra-rápido para la creación
  del entorno virtual).

---

## 3. Preparación de Datos e Ingeniería de Características

El pipeline implementado en el cuaderno Jupyter realiza los siguientes pasos
obligatorios:

1. **Eliminación del Identificador:** Se remueve la columna `Id` por no aportar
   valor comercial ni predictivo.
2. **Tratamiento de Valores Faltantes (Imputación Fina):**
   - En 14 columnas categóricas (como `PoolQC`, `Fence`, `Alley`, `BsmtQual`,
     `GarageQual`, etc.), el nulo `NA` representa la **ausencia del atributo**
     (ej. "No tiene piscina" o "No tiene sótano"). Se imputaron como la cadena
     `"None"` para preservar su significado comercial.
   - `LotFrontage` (frente de calle de la propiedad) se imputa dinámicamente con
     la **mediana del vecindario (`Neighborhood`)** en el que se encuentra la
     casa.
   - `GarageYrBlt` (año de garaje) faltante se imputa con el año de construcción
     de la casa (`YearBuilt`).
   - `MasVnrArea` se imputa con `0` y `MasVnrType` con `"None"`.
   - `Electrical` se imputa con la moda.
3. **Ingeniería de Características:** Se derivaron nuevas métricas predictivas
   con interpretación comercial directa:
   - `HouseAge`: Antigüedad de la propiedad al momento de la venta (`YrSold` -
     `YearBuilt`).
   - `RemodAge`: Años desde la última remodelación al momento de la venta
     (`YrSold` - `YearRemodAdd`).
   - `TotalSF`: Área construida habitable total útil (`1stFlrSF` + `2ndFlrSF` +
     `TotalBsmtSF`).
   - `TotalBath`: Cantidad total de baños de la propiedad sumando completos y
     medios baños (`FullBath` + `0.5*HalfBath` + `BsmtFullBath` +
     `0.5*BsmtHalfBath`).
   - `HasPool`: Indicador binario indicando si cuenta con piscina (1) o no (0).
4. **Transformación Logarítmica:** Se aplicó la transformación $\log(1 + x)$
   sobre el precio de venta (`SalePrice`) para normalizar su distribución
   asimétrica y estabilizar la varianza del modelo.

---

## 4. Resultados de Modelado y Evaluación

El experimento se configuró dividiendo el dataset en **70% de datos de
entrenamiento** (1,022 propiedades) y **30% de datos de prueba / holdout** (438
propiedades) con normalización robusta y remoción de multicolinealidad.

### Modelos de Regularización Lineal (Sintonizados con Grilla Fina)

Al evaluar los modelos en el espacio logarítmico mediante validación cruzada (10
folds):

| Modelo                    | Hiperparámetro Óptimo ($\alpha$ / $\lambda$) | $R^2$ Log-Scale (CV Mean) | MAE Log-Scale (CV Mean) |
| :------------------------ | :------------------------------------------: | :-----------------------: | :---------------------: |
| **Ridge Regression**      |               $\alpha = 20.0$                |         $0.8420$          |        $0.0931$         |
| **Lasso Regression**      |              $\alpha = 0.0005$               |         $0.8514$          |        $0.0869$         |
| **ElasticNet Regression** |     $\alpha = 0.001$, $l1\_ratio = 0.9$      |         $0.8440$          |        $0.0905$         |

> [!NOTE]
> Dado que el precio fue log-transformado, un Lasso con la penalización por
> defecto ($\alpha=1.0$) devalúa todos los coeficientes a cero. Mediante una
> grilla personalizada, encontramos que el $\alpha$ óptimo para Lasso es
> $0.0005$, logrando un excelente balance que elimina variables redundantes pero
> mantiene el poder predictivo con un $R^2$ promedio de **0.8514**.

### Modelo Final Seleccionado

Aunque los modelos lineales regularizados (especialmente Lasso) demostraron gran
robustez, la comparación de PyCaret seleccionó a **Gradient Boosting Regressor**
como el mejor modelo en términos de precisión. Tras sintonizar sus
hiperparámetros y evaluarlo en el conjunto de test independiente,
**exponencializando los precios de vuelta a su escala original en Dólares
(AUD)**, obtuvimos:

- **$R^2$ Score (Varianza explicada):** **$0.9097$** (el modelo explica el
  90.97% de los precios de venta)
- **MAE (Error Absoluto Medio):** **$15,225.79 AUD** (desviación promedio por
  propiedad de solo quince mil dólares)
- **RMSE (Raíz de Error Cuadrático Medio):** **$25,068.79 AUD** (control
  estricto de predicciones erráticas en mansiones)
- **MAPE (Error Porcentual Medio):** **$8.90\%$** (error promedio menor al 9%
  por propiedad)

---

## 5. Interpretabilidad del Modelo y Recomendaciones para Surprise Housing

### Variables Más Significativas

1. **`TotalSF` (Superficie Útil Construida):** Atributo más importante. Cada
   metro cuadrado adicional incrementa de forma directa la valoración de
   mercado.
2. **`OverallQual` (Calidad Material General):** Factor multiplicativo clave del
   valor residencial.
3. **`HouseAge` y `RemodAge` (Antigüedad / Remodelación):** La antigüedad
   devalúa la propiedad, pero realizar remodelaciones completas resetea este
   contador e incrementa sustancialmente el precio, lo cual valida
   científicamente el modelo de negocio de remodelaciones de Surprise Housing.
4. **`Neighborhood` (Vecindario):** Ciertas localizaciones geográficas agregan
   primas significativas al precio intrínseco.

### Recomendaciones Estratégicas para Inversión

- **Adquisición Basada en Desviación:** Listar de manera automatizada
  propiedades cuyo precio publicado en el mercado sea un **15% o más inferior**
  al valor real estimado por nuestro modelo. Esto identifica propiedades
  severamente subvaloradas para compra inmediata.
- **Foco en Acabados y Calidad:** Dado el inmenso peso de `OverallQual` y
  `KitchenQual`, el presupuesto de remodelación debe enfocarse en acabados de
  alta calidad e interiores (cocina/baños) antes que en extensiones de metros
  cuadrados físicos de terreno, ya que estas remodelaciones estéticas de alta
  gama ofrecen el mayor margen de ganancia por dólar invertido.
- **Revitalización de Viviendas Antiguas:** Adquirir propiedades con cimientos
  sólidos pero con años de construcción elevados, aplicando remodelaciones
  completas para reducir el contador de `RemodAge` a cero, maximizando la
  valorización del activo en el mercado.

---

## 6. Instrucciones de Descarga y Configuración Local

Para ejecutar el pipeline y el Jupyter Notebook localmente, se recomienda
utilizar el gestor de paquetes rápido `uv` o en su defecto `pip` tradicional:

### Requisitos Previos

- **Python 3.10** instalado en el sistema.
- **libomp** instalado en macOS (si usa Mac, instálelo ejecutando
  `brew install libomp`).

### Instalación y Ejecución paso a paso:

1. **Clonar este repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd 02-proyecto
   ```

2. **Crear e iniciar el entorno virtual:**
   - Utilizando `uv` (Recomendado):
     ```bash
     uv venv --python 3.10
     source .venv/bin/activate
     ```
   - Utilizando `venv` clásico:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   uv pip install pycaret pandas numpy matplotlib seaborn scikit-learn ipykernel notebook shap scipy category-encoders==2.6.4 joblib==1.3.2
   ```
   _(Nota: Se fijan las versiones de `category-encoders` y `joblib` por
   compatibilidad estricta con PyCaret)._

4. **Ejecutar el Jupyter Notebook:**
   ```bash
   jupyter notebook "MFSDv1p1_Regresion Avanzada con PyCaret.ipynb"
   ```

---

## Integrantes del Grupo

- Rios Nuñez David Samuel
- Rivera Quisberth Juan Enrique
- Terceros Beltran Oscar Alvaro
- Torrez Azuga Marcelo
