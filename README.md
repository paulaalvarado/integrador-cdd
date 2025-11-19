# CuartaPresentacionStreamlitApp
Aplicación en Streamlit que permita explorar los datos y resultados visualizados, así como también ofrecer una interfaz sencilla para que un usuario final pueda ingresar datos nuevos y probar el modelo entrenado basada en los mismos datos y modelos presentados en los notebooks
# 👶 Predicción de Tasas de Natalidad Global

Sistema de predicción de tasas de natalidad utilizando Machine Learning y variables socioeconómicas.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Descripción

Esta aplicación utiliza modelos de Machine Learning para predecir y analizar las tasas de natalidad a nivel global, considerando múltiples factores socioeconómicos como PIB per cápita, educación femenina, mortalidad infantil y urbanización.

## 🚀 Características

- 📊 **Visualizaciones interactivas** con Altair
- 🤖 **Predictor en tiempo real** con inputs personalizables
- 📁 **Exploración de datos** con filtros dinámicos
- 📈 **Análisis temporal** de tendencias de natalidad
- 🗺️ **Comparaciones regionales** y por país

## 🛠️ Tecnologías

- **Frontend:** Streamlit
- **Visualización:** Altair, Plotly
- **ML:** Scikit-learn, XGBoost
- **Data:** Pandas, NumPy

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/natalidad-predictor.git
cd natalidad-predictor
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🎮 Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Estructura de navegación

- **🏠 Inicio:** Información general y métricas del modelo
- **📊 Visualizaciones:** Gráficos interactivos de tendencias
- **🤖 Predictor:** Herramienta de predicción personalizada
- **📁 Datos:** Exploración y descarga del dataset

## 📂 Estructura del Proyecto

```
natalidad-predictor/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── data/                     # Datasets
├── models/                   # Modelos entrenados
├── src/                      # Código fuente
│   ├── functions.py          # Funciones de procesamiento
│   ├── visualizations.py     # Gráficos
│   └── model.py              # Lógica del modelo
└── notebooks/                # Notebooks de desarrollo
```

## 📊 Datos

El dataset incluye:
- **Periodo:** 2000-2023
- **Países:** 195
- **Variables:** PIB, educación, salud, urbanización, etc.

## 🤖 Modelo

- **Algoritmos:** Random Forest, Gradient Boosting, XGBoost
- **R² Score:** 0.89
- **RMSE:** 2.34

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en GitHub.

---

⭐ Si te gusta este proyecto, dame una estrella en GitHub!