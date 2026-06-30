# Correcciones Aplicadas

## Notebook

- Se reorganizó el notebook de 16 celdas grandes a 46 celdas más legibles.
- Se agregaron celdas Markdown antes de cada gráfico, tabla o bloque de modelado.
- Se agregó control de calidad antes de limpiar los datos.
- Se agregó verificación posterior para demostrar que el dataset limpio no conserva nulos.
- Se reemplazó la imputación con `None` por `NoFeature` para evitar que Pandas vuelva a leer esos valores como nulos.
- Se corrigió el gráfico de distribución original vs. logarítmica de `SalePrice`.
- Se aclaró que los outliers se analizan visualmente, pero no se eliminan automáticamente.
- Se separaron los gráficos de EDA en celdas individuales.
- Se separó la construcción del modelo en setup, comparación, modelos regularizados, tuning y modelo final.
- Se separó la evaluación en métricas, gráfico real vs. predicho y análisis de residuos.
- Se corrigió ElasticNet a `alpha = 0.001` y `l1_ratio = 0.8`.
- Se agregó una nota metodológica sobre el uso del dataset Ames Housing bajo el caso de negocio australiano.

## README

- Se eliminó `.venv/` de la estructura del repositorio.
- Se agregó `requirements.txt` como forma recomendada de instalación.
- Se agregó `.gitignore` para evitar subir entorno virtual, cachés y archivos temporales.
- Se corrigió ElasticNet de `l1_ratio = 0.9` a `l1_ratio = 0.8`.
- Se actualizó la explicación de imputación usando `NoFeature`.
- Se agregó nota metodológica sobre Ames Housing vs. Australia.
- Se reorganizaron las instrucciones de instalación y ejecución local.

## Dataset limpio

- Se regeneró `cleaned_dataset.csv` usando `NoFeature`.
- Se verificó que el archivo recargado con `pd.read_csv()` no vuelva a mostrar valores faltantes.
