# Tarea 1 - Análisis exploratorio y Machine Learning básico con Python

## Descripción del repositorio

Este repositorio contiene los entregables de la Tarea 1 del módulo **Big Data, Analytics & Data Scientist**, correspondiente al desarrollo de un análisis exploratorio de datos y un modelo básico de Machine Learning supervisado utilizando Python en Google Colab.

El caso de estudio se centra en el análisis de la experiencia de usuarios en una plataforma digital educativa, a partir de un dataset sintético diseñado con fines académicos.

## Objetivo del proyecto

Aplicar un flujo completo de análisis de datos en Python para identificar factores asociados con la satisfacción de los usuarios de una plataforma digital educativa y construir un modelo básico de clasificación que permita predecir si un usuario se encuentra satisfecho o no.

## Archivos del repositorio

| Archivo                                  | Descripción                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `Digital_Service_Experience.ipynb`       | Notebook desarrollado en Google Colab con carga de datos, limpieza, EDA, visualizaciones, modelo de Machine Learning y conclusiones. |
| `digital_service_experience_dataset.csv` | Dataset sintético utilizado para el análisis.                                                                                        |
| `README.md`                              | Documento explicativo del repositorio, metodología, dataset y forma de ejecución.                                                    |

## Dataset utilizado

El dataset `digital_service_experience_dataset.csv` fue construido de forma sintética con fines académicos. No contiene información real, sensible ni confidencial.

Cada fila representa una interacción o evaluación de un usuario sobre una plataforma digital educativa.

El dataset incluye variables demográficas, tecnológicas, operativas y de percepción, entre ellas:

* Edad del usuario.
* Género.
* Ciudad.
* Tipo de usuario.
* Tipo de dispositivo.
* Calidad de internet.
* Sesiones por semana.
* Duración promedio de sesión.
* Actividades completadas y pendientes.
* Tickets de soporte.
* Tiempo de respuesta del soporte.
* Errores de plataforma.
* Capacitación recibida.
* Canal de soporte.
* Módulo utilizado.
* Usabilidad percibida.
* Puntaje de recomendación.
* Nivel de satisfacción.
* Variable objetivo `satisfied`.

## Variable objetivo

La variable objetivo del modelo es:

```text
satisfied
```

Esta variable representa si el usuario tiene una experiencia satisfactoria o no.

| Valor | Interpretación        |
| ----- | --------------------- |
| `1`   | Usuario satisfecho    |
| `0`   | Usuario no satisfecho |

El problema fue abordado como una tarea de **clasificación binaria**.

## Metodología aplicada

El notebook sigue el siguiente pipeline:

```text
Carga de datos → Revisión inicial → Limpieza y preparación → EDA → Visualización → Modelo ML → Evaluación → Resultados
```

Las principales etapas fueron:

1. **Carga de librerías**
   Se utilizaron Pandas, NumPy, Matplotlib, Seaborn y scikit-learn.

2. **Lectura del dataset**
   Se cargó el archivo CSV en un DataFrame de Pandas.

3. **Revisión inicial**
   Se analizaron dimensiones, columnas, tipos de datos, valores nulos, duplicados y estadísticas descriptivas.

4. **Limpieza de datos**
   Se eliminaron duplicados, se imputaron valores nulos y se convirtió la columna de fecha a formato `datetime`.

5. **Transformaciones**
   Se crearon variables temporales: `year`, `month` y `year_month`.

6. **Análisis exploratorio de datos**
   Se aplicaron operaciones con `value_counts()`, `nunique()`, `groupby()`, `agg()` y `pd.crosstab()`.

7. **Visualización de datos**
   Se generaron gráficos de barras, pastel, histograma, boxplot, scatter plot, violin plot, línea temporal y mapa de calor.

8. **Machine Learning básico**
   Se entrenó un modelo supervisado de clasificación usando `DecisionTreeClassifier`.

## Principales resultados del análisis

El dataset inicial contenía:

| Indicador                           | Valor |
| ----------------------------------- | ----: |
| Registros iniciales                 | 1.515 |
| Columnas iniciales                  |    22 |
| Registros finales luego de limpieza | 1.500 |
| Columnas finales                    |    25 |
| Valores nulos finales               |     0 |
| Duplicados finales                  |     0 |

Hallazgos principales:

* La satisfacción alta no es predominante en el dataset.
* El 17,27% de los usuarios fueron clasificados como satisfechos.
* La calidad de internet alta se asocia con mayor proporción de satisfacción.
* Los usuarios satisfechos presentan mayor usabilidad percibida y mayor puntaje de recomendación.
* Los usuarios no satisfechos reportan más errores de plataforma y más tickets de soporte.
* La variable más relevante para el modelo fue `perceived_usability`.

## Modelo de Machine Learning

Se utilizó un modelo `DecisionTreeClassifier` de scikit-learn.

El modelo fue entrenado con una división de datos de:

```text
80% entrenamiento / 20% prueba
```

Se utilizó un pipeline con:

* `ColumnTransformer`
* `OneHotEncoder`
* `DecisionTreeClassifier(max_depth=5, random_state=42)`

## Métricas del modelo

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |    85,33% |
| Precision |    61,11% |
| Recall    |    42,31% |
| F1-score  |    50,00% |

La matriz de confusión obtuvo:

| Resultado            | Valor |
| -------------------- | ----: |
| Verdaderos negativos |   234 |
| Falsos positivos     |    14 |
| Falsos negativos     |    30 |
| Verdaderos positivos |    22 |

El modelo presenta buen desempeño general en accuracy, pero tiene una capacidad moderada para identificar usuarios satisfechos. Esto se explica en parte por el desbalance de la variable objetivo, donde la clase satisfecha representa una proporción menor del dataset.

## Cómo ejecutar el notebook

### Opción 1: Ejecutar en Google Colab

1. Abrir el archivo `Digital_Service_Experience.ipynb`.
2. Seleccionar la opción **Open in Colab** o cargar el notebook manualmente en Google Colab.
3. Subir el archivo `digital_service_experience_dataset.csv` al entorno de Colab o montarlo desde Google Drive.
4. Verificar que la ruta del archivo CSV en el notebook sea correcta.
5. Ejecutar las celdas en orden desde el menú:

```text
Entorno de ejecución → Ejecutar todas
```

## Requisitos principales

El notebook utiliza las siguientes librerías:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Limitaciones

* El dataset es sintético y fue construido con fines académicos.
* La variable objetivo está desbalanceada.
* El modelo debe interpretarse como una aproximación inicial, no como una solución productiva.
* Se recomienda comparar modelos adicionales en futuros análisis.

## Recomendaciones futuras

* Probar otros algoritmos como `LogisticRegression` o `RandomForestClassifier`.
* Aplicar técnicas de balanceo de clases.
* Realizar ajuste de hiperparámetros.
* Incorporar nuevas variables relacionadas con experiencia de usuario.
* Comparar resultados con un dataset real anonimizado, si estuviera disponible.

## Autor

**Sonia Guama**
Maestría en Tecnologías de la Información
Mención Transformación Digital e Innovación
Universidad Hemisferios
