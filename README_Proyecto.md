# Proyecto de Machine Learning - Riesgo Crediticio

Este repositorio contiene un experimento completo de machine learning que incluye descarga de datos, análisis exploratorio, preparación de datasets y entrenamiento de modelo CatBoost optimizado.


1. Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```

##  Ejecución del Experimento

El experimento debe ejecutarse en el orden indicado, ya que cada paso depende del anterior.

### Paso 1: Descarga de Datasets

Este script descarga los datasets originales desde Hugging Face.

```bash
python src/1_download_dataset.py
```

**¿Qué hace este paso?**
- Se conecta a Hugging Face Hub
- Descarga los datasets originales
- Guarda los datos en la carpeta `data/raw/`

**Salida esperada:** Archivos de datos originales en `data/raw/`

---

### Paso 2: Análisis Exploratorio y Limpieza (EDA)

Realiza el análisis exploratorio de datos, limpieza y depuración.

```bash
python src/2_EDA.py
```

**¿Qué hace este paso?**
- Analiza la estructura y calidad de los datos
- Identifica y trata valores nulos, duplicados y outliers
- Realiza transformaciones y limpieza de datos
- Genera visualizaciones y estadísticas descriptivas
- Guarda los datos limpios en `data/processed/`

**Salida esperada:** 
- Datasets limpios en `data/processed/`
- Reportes de análisis y gráficas (opcional)

---

### Paso 3: Creación del Dataset Completo

Combina el dataset limpio con otros datasets del repositorio.

```bash
python src/3_create_full_dataset.py
```

**¿Qué hace este paso?**
- Lee los datos limpios del paso anterior
- Integra múltiples datasets del repositorio
- Realiza merge o concatenación según la lógica del negocio
- Crea el dataset final unificado
- Guarda el resultado en `data/processed/`

**Salida esperada:** Dataset completo y unificado en `data/processed/`

---

### Paso 4: Entrenamiento del Modelo CatBoost Optimizado

Crea y entrena el modelo CatBoost con los mejores hiperparámetros.

```bash
python src/4_catboost_best_scores.py
```

**¿Qué hace este paso?**
- Carga el dataset final
- Realiza división train/test
- Entrena modelo CatBoost con hiperparámetros optimizados
- Evalúa el rendimiento del modelo
- Guarda el modelo entrenado en `models/`

**Salida esperada:** 
- Modelo entrenado en `models/`
- Métricas de evaluación
- Reportes de performance

---

## Ejecución Completa

Si deseas ejecutar todo el pipeline de una vez, puedes usar:

```bash
python src/1_download_dataset.py && \
python src/2_EDA.py && \
python src/3_create_full_dataset.py && \
python src/4_catboost_best_scores.py
```


## ⚠️ Notas Importantes

- Cada paso debe ejecutarse en orden secuencial
- Asegúrate de que cada paso se complete exitosamente antes de continuar con el siguiente
- Los scripts pueden tardar varios minutos dependiendo del tamaño de los datos
- Verifica que tienes suficiente espacio en disco para los datasets

## 🐛 Solución de Problemas

**Error de conexión a Hugging Face:**
- Verifica tu conexión a internet
- Asegúrate de tener instalado `huggingface-hub`

**Error de memoria:**
- Considera procesar los datos en lotes más pequeños
- Aumenta la memoria disponible o usa una máquina más potente

**Dependencias faltantes:**
- Ejecuta `pip install -r requirements.txt` nuevamente
- Verifica la versión de Python



## 👥 Contribuciones

Iago Rivadulla, Agustin Marquez y Gabriel De Almeida

## 📧 Contacto

- Iago Rivadulla: (link github)
- Agustin Marquez: (link github)
- Gabriel De Almeida: (link github)