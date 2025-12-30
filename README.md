# Clasificación y recomendación de revistas científicas

Trabajo de curso para la asignatura **Ciencia de Datos en Ingeniería**, cuyo objetivo es comparar una **aproximación clásica** y una **aproximación conexionista** para la clasificación automática de artículos científicos en revistas.

---

##  Descripción del proyecto

El sistema clasifica artículos científicos a partir de:

* **Título**
* **Resumen (abstract)**
* **Palabras clave**

Se emplean artículos de varias revistas de Elsevier (2020–2024), y se comparan:

* **Aproximación clásica**: TF-IDF + modelos de Sklearn
* **Aproximación conexionista**: Red neuronal BiGRU implementada en PyTorch

---

## Estructura del proyecto

```
.
├── Dataset/                # Revistas organizadas por carpetas 
│   ├── Applied_Ergonomics/
│   ├── Neural_Networks/
│   ├── Robotics/
│   └── Visual_Communication/
├── src/
│   ├── 01_build_dataset.py
│   ├── 02_classical_sklearn.py
│   └── 03_neural_pytorch.py
├── outputs/                # Resultados generados
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Requisitos

* Python **3.9+**
* pip
* Sistema operativo: Windows / Linux / macOS

---

## Creación y activación del entorno virtual

### 1️⃣ Crear entorno virtual

```bash
python -m venv .venv
```

### 2️⃣ Activar entorno virtual

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

---

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución del proyecto

### 1️⃣ Construcción del dataset

```bash
python src/build_dataset.py
```

Genera:

* `outputs/dataset.csv`

---

### 2️⃣ Aproximación clásica (Sklearn)

```bash
python src/classical_sklearn.py
```

Modelos entrenados:

* LinearSVC
* Regresión logística
* Naive Bayes

Resultados:

* Accuracy
* Precision / Recall / F1
* `outputs/metrics_classical.json`

---

### 3️⃣ Aproximación conexionista (PyTorch)

```bash
python src/neural_pytorch.py
```

Modelo:

* BiGRU con early stopping

Resultados:

* Accuracy
* F1-macro y F1-ponderado
* Matriz de confusión
* `outputs/metrics_neural.json`

---

## Validación

* División **80/20 estratificada**
* Métricas sobre conjunto de validación
* Comparación directa entre enfoques

---

## Tecnologías utilizadas

* Python
* Scikit-learn
* PyTorch
* Pandas
* NumPy

