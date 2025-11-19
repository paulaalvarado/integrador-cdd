"""
Aplicación Streamlit: Predicción de Tasas de Natalidad Global
Basada en CuartaPresentacion.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Importaciones corregidas
from src.pipeline import ejecutar_pipeline_completo, get_resumen_pipeline, cargar_datos
from src.visualizations import (
    viz_evolucion_temporal_regiones,
    viz_dinamica_natalidad_vs_variable_region,
    viz_mapa_mundial_natalidad,
    viz_evolucion_paises_highlight,
    viz_dinamica_natalidad_vs_variable,
    viz_correlaciones_interactivas,   
    get_available_visualizations
)
from src.model import (
    load_model, 
    load_scaler,
    load_imputer,
    predict_birth_rate,
    predict_batch,
    interpret_prediction,
    get_prediction_category,
    evaluate_model
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predicción de Natalidad Global",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS CSS
# ============================================
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #364152;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE DATOS (CON CACHE)
# ============================================

@st.cache_data
def cargar_datos_app():
    """Carga y procesa los datos con cache de Streamlit"""
    # IMPORTANTE: Ruta al CSV ORIGINAL (merged_dataset.csv)
    ruta = 'data/raw/merged_dataset.csv'
    
    # Verificar si existe
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo: {ruta}")
        st.info("💡 Asegúrate de tener el archivo merged_dataset.csv en la carpeta data/raw/")
        return None
    
    # Ejecutar pipeline completo (limpieza + features + regiones)
    df_procesado = ejecutar_pipeline_completo(ruta, umbral_faltantes=60)
    
    return df_procesado


# ============================================
# SIDEBAR: NAVEGACIÓN
# ============================================

st.sidebar.title("Navegación")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Visualizaciones", "🧠 Predictor", "📁 Datos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
"""
**📌 Predicción de Natalidad Global**  

**📌 Ingeniería en Sistemas**

**📌 Ciencia de Datos - Grupo 7**

**📌 Dataset Banco Mundial (2000-2023)** 

**📌 Última actualización:** Nov 2024
"""
)

# ============================================
# CARGAR DATOS
# ============================================

with st.spinner("🔄 Cargando y procesando datos..."):
    df = cargar_datos_app()
    
    if df is None:
        st.stop()  # Detener ejecución si no hay datos
    
    # Cargar dataset original solo para el resumen
    df_original = cargar_datos('data/raw/merged_dataset.csv')
    resumen = get_resumen_pipeline(df_original, df)

# ============================================
# PÁGINA: INICIO
# ============================================

if pagina == "🏠 Inicio":
    st.title("Predicción de Tasas de Natalidad Global")
    st.markdown("---")
    
    # Introducción
    st.markdown("""
    ### Bienvenido al Sistema de Análisis y Predicción de Natalidad
    
    Esta aplicación utiliza **Machine Learning** para analizar y predecir las tasas de natalidad 
    a nivel global, considerando múltiples factores socioeconómicos y temporales.
    
    #### Objetivo del Proyecto
    Comprender los factores que influyen en las tasas de natalidad y crear modelos predictivos 
    que ayuden a entender tendencias demográficas globales.
    """)
    
    # Métricas principales
    st.markdown("### 📊 Estadísticas del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Países Analizados",
            f"{resumen['paises_unicos']}",
            help="Número de países con datos completos"
        )
    
    with col2:
        st.metric(
            "Regiones Geográficas",
            f"{resumen['regiones_unicas']}",
            help="Divisiones geográficas para análisis regional"
        )
    
    with col3:
        st.metric(
            "Años de Datos",
            f"{resumen['años_max'] - resumen['años_min'] + 1}",
            f"{resumen['años_min']}-{resumen['años_max']}"
        )
    
    with col4:
        st.metric(
            "Variables Analizadas",
            f"{resumen['columnas_procesado']}",
            help="Features después del procesamiento"
        )
    
    st.markdown("---")
    
    # Información del pipeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Pipeline de Procesamiento")
        st.markdown("""
        1. **Carga de Datos** - Dataset original del Banco Mundial
        2. **Limpieza** - Eliminación de nulos y duplicados
        3. **Eliminación de Leakage** - Variables que generan sesgo
        4. **Feature Engineering** - Creación de features temporales
        5. **Asignación Geográfica** - Continentes y regiones
        6. **Preparación para ML** - Escalado e imputación
        """)
    
    with col2:
        st.markdown("#### 📈 Variables Principales")
        st.markdown("""
        - **Socioeconómicas:** PIB per cápita, Ingreso medio, Desempleo
        - **Educación:** Acceso a educación, Matrícula escolar
        - **Salud:** Esperanza de vida, Acceso a salud, Vacunación
        - **Demografía:** Urbanización, Densidad poblacional
        - **Género:** Participación laboral femenina, Educación femenina
        - **Temporales:** Crisis 2008, Pandemia COVID-19
        """)
    
    st.markdown("---")
    
    # Continentes disponibles
    st.markdown("### 🌍 Continentes en el Dataset")
    continentes_cols = st.columns(len(resumen['continentes']))
    
    for idx, continente in enumerate(resumen['continentes']):
        with continentes_cols[idx]:
            n_paises = df[df['Continente'] == continente]['Pais'].nunique()
            st.info(f"**{continente}**\n\n{n_paises} países")
    
    st.markdown("---")
    
    # Resumen de transformaciones
    with st.expander("ℹ️ Ver detalles del procesamiento de datos"):
        st.markdown(f"""
        **Dataset Original:**
        - Filas: {resumen['filas_original']:,}
        - Columnas: {resumen['columnas_original']}
        
        **Dataset Procesado:**
        - Filas: {resumen['filas_procesado']:,} ({resumen['filas_procesado']/resumen['filas_original']*100:.1f}% conservado)
        - Columnas: {resumen['columnas_procesado']} (eliminadas {resumen['columnas_original'] - resumen['columnas_procesado']} por leakage/nulos)
        
        **Calidad de Datos:**
        - ✅ Sin duplicados
        - ✅ Variables con leakage eliminadas
        - ✅ Features temporales creadas
        - ✅ Regiones geográficas asignadas
        """)

# ============================================
# PÁGINA: VISUALIZACIONES
# ============================================

elif pagina == "📊 Visualizaciones":
    st.title("📊 Visualizaciones Interactivas")
    st.markdown("---")
    
    # Selector de visualización
    vizs = get_available_visualizations()
    
    viz_seleccionada = st.selectbox(
        "Selecciona una visualización:",
        options=[viz['nombre'] for viz in vizs],
        format_func=lambda x: f"📈 {x}"
    )
    
    # Encontrar la viz seleccionada
    viz_actual = next(viz for viz in vizs if viz['nombre'] == viz_seleccionada)
    
    # Mostrar descripción
    st.info(f"**{viz_actual['descripcion']}**")
    
    st.markdown("---")
    
    # Generar y mostrar visualización
    # Generar y mostrar visualización
    with st.spinner("🎨 Generando visualización..."):
        try:
            chart = None
            mostrar_error = True

            if viz_actual['id'] == 'evolucion_temporal':
                # Gráfico 1: evolución por región
                chart = viz_evolucion_temporal_regiones(df)
            elif viz_actual['id'] == 'dinamica_continente':
                st.markdown("#### 🌍 Dinámica de natalidad vs variable (por continente)")

                variable_x = st.selectbox(
                    "Variable a comparar con Natalidad:",
                    [c for c in df.columns if c not in ["Año", "Pais", "Continente", "Natalidad"]]
                )

                anio = st.slider(
                    "Seleccioná el año:",
                    int(df["Año"].min()),
                    int(df["Año"].max()),
                    2010,
                    step=1
                )

                usar_densidad = st.checkbox("Usar tamaño según Densidad Poblacional", value=True)

                continentes_disponibles = sorted(df["Continente"].dropna().unique().tolist())
                continentes_resaltados = st.multiselect(
                    "Seleccioná uno o varios continentes para resaltar:",
                    options=continentes_disponibles,
                )

                chart = viz_dinamica_natalidad_vs_variable_region(
                    df,
                    variable_x=variable_x,
                    anio=anio,
                    usar_densidad=usar_densidad,
                    continentes_resaltados=continentes_resaltados
                )

            elif viz_actual['id'] == 'mapa_mundial':
                # Mapa mundial
                chart = viz_mapa_mundial_natalidad(df)

            elif viz_actual['id'] == 'evolucion_paises':
                # Gráfico 4: natalidad por país (serie) con highlight

                # Lista de países disponibles en el dataset
                paises_disponibles = sorted(df['Pais'].dropna().unique().tolist())

                # Multiselect para elegir qué países mostrar
                paises_seleccionados = st.multiselect(
                    "Elegí país(es) a mostrar:",
                    options=paises_disponibles,
                    default=["Argentina"] if "Argentina" in paises_disponibles else paises_disponibles[:3]
                )

                if not paises_seleccionados:
                    st.info("Seleccioná al menos un país para ver la serie de natalidad.")
                    chart = None
                    mostrar_error = False
                else:
                    # Entre los seleccionados, elegimos cuál resaltar
                    pais_resaltado = st.selectbox(
                        "¿Qué país querés resaltar?",
                        options=paises_seleccionados,
                        index=0
                    )

                    chart = viz_evolucion_paises_highlight(
                        df,
                        paises_seleccionados=paises_seleccionados,
                        pais_resaltado=pais_resaltado
                    )
            elif viz_actual['id'] == 'dinamica_natalidad':
                chart = viz_dinamica_natalidad_vs_variable(df)

            elif viz_actual['id'] == 'correlaciones_interactivas':
                chart = viz_correlaciones_interactivas(df)

            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            elif mostrar_error:
                st.error("❌ No se pudo generar el gráfico")

        except Exception as e:
            st.error(f"❌ Error al generar la visualización: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)

    
    # Tips de interacción
    with st.expander("💡 Tips de interacción"):
        st.markdown("""
        - **Zoom:** Rueda del mouse sobre el gráfico
        - **Pan:** Click y arrastra
        - **Tooltip:** Pasa el mouse sobre los elementos
        - **Filtros:** Usa los selectores interactivos
        - **Reset:** Doble click en el gráfico
        """)

# ============================================
# PÁGINA: PREDICTOR CON FEATURE IMPACT
# ============================================

elif pagina == "🧠 Predictor":
    st.title("🧠 Predictor de Natalidad")
    st.markdown("---")
    
    # Verificar si existen los modelos
    modelo_existe = os.path.exists('models/best_model.pkl')
    scaler_existe = os.path.exists('models/scaler.pkl')
    imputer_existe = os.path.exists('models/imputer.pkl')
    
    if not modelo_existe or not scaler_existe or not imputer_existe:
        st.warning("⚠️ **Modelos no encontrados**")
        st.markdown("""
        ### 🔧 Configuración Necesaria
        
        Para usar el predictor, necesitas:
        
        1. **Entrenar el modelo** ejecutando el notebook `CuartaPresentacion.ipynb`
        2. **Exportar el modelo** con el código proporcionado en las instrucciones
        3. **Copiar los archivos** a la carpeta `models/`:
           - `best_model.pkl`
           - `scaler.pkl`
           - `imputer.pkl`
        """)
        st.stop()
    
    # ============================================
    # CARGAR MODELO Y PREPARAR DATOS
    # ============================================
    
    with st.spinner("🔄 Cargando modelo y preparando datos..."):
        # Cargar modelo, scaler e imputer
        model = load_model()
        scaler = load_scaler()
        
        try:
            import joblib
            imputer = joblib.load('models/imputer.pkl')
        except:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            st.warning("⚠️ Imputer no encontrado, usando uno por defecto")
        
        # Preparar datos para el modelo
        from src.pipeline import preparar_para_modelo
        
        X_train, X_test, y_train, y_test, scaler_prep, feature_names, imputer_prep = preparar_para_modelo(
            df, 
            año_corte=2021,
            random_state=42
        )
        
        # Obtener los índices originales para mapear países
        train_mask = df['Año'] <= 2021
        test_mask = df['Año'] > 2021
        
        df_train_original = df[train_mask].copy()
        df_test_original = df[test_mask].copy()
    
    st.success("✅ Modelo cargado correctamente")
    
    # ============================================
    # TABS DE NAVEGACIÓN
    # ============================================
    
    tab1, tab2 = st.tabs(["📊 Evaluación por País", "📈 Métricas Generales"])
    
    # ============================================
    # TAB 1: EVALUACIÓN POR PAÍS
    # ============================================
    
    with tab1:
        st.markdown("### 🌍 Selecciona un País para Evaluar")
        st.markdown("Visualiza cómo el modelo predice la natalidad y qué variables influyeron en la decisión.")
        
        # Selector de país
        paises_disponibles = sorted(df['Pais'].unique())
        pais_seleccionado = st.selectbox(
            "Selecciona un país:",
            options=paises_disponibles,
            index=paises_disponibles.index('Argentina') if 'Argentina' in paises_disponibles else 0
        )
        
        # Filtrar datos del país seleccionado
        df_pais = df[df['Pais'] == pais_seleccionado].sort_values('Año').copy()
        
        if len(df_pais) == 0:
            st.error(f"No hay datos disponibles para {pais_seleccionado}")
            st.stop()
        
        st.markdown("---")
        
        # ============================================
        # REALIZAR PREDICCIONES PARA EL PAÍS
        # ============================================
        
        with st.spinner(f"🔮 Realizando predicciones para {pais_seleccionado}..."):
            # Separar datos del país en train y test
            df_pais_train = df_pais[df_pais['Año'] <= 2021].copy()
            df_pais_test = df_pais[df_pais['Año'] > 2021].copy()
            
            predicciones_test = []
            años_test = []
            X_test_pais_list = []  # Guardar para análisis
            
            # Columnas a excluir para features
            columnas_excluir = ['Natalidad', 'Año', 'Pais', 'CodigoPais', 'Continente', 'Region']
            columnas_excluir_existentes = [col for col in columnas_excluir if col in df_pais_test.columns]
            
            # Hacer predicciones para cada año de test
            for idx, row in df_pais_test.iterrows():
                X_row = row.drop(labels=columnas_excluir_existentes)
                X_row_df = pd.DataFrame([X_row])
                
                # Imputar y escalar
                try:
                    X_row_imputed = imputer.transform(X_row_df)
                    X_row_scaled = scaler.transform(X_row_imputed)
                    
                    # Predecir
                    pred = model.predict(X_row_scaled)[0]
                    predicciones_test.append(pred)
                    años_test.append(row['Año'])
                    X_test_pais_list.append(X_row_scaled[0])  # Guardar features escaladas
                except Exception as e:
                    st.error(f"Error al predecir para año {row['Año']}: {e}")
                    predicciones_test.append(None)
                    años_test.append(row['Año'])
                    X_test_pais_list.append(None)
        
        # ============================================
        # GRÁFICO: REAL VS PREDICHO
        # ============================================
        
        st.markdown("### 📈 Evolución Temporal: Real vs Predicho")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Línea de datos reales (toda la serie)
        fig.add_trace(go.Scatter(
            x=df_pais['Año'],
            y=df_pais['Natalidad'],
            mode='lines+markers',
            name='Datos Reales',
            line=dict(color='steelblue', width=3),
            marker=dict(size=8)
        ))
        
        # Línea de predicciones (solo años test)
        if len(predicciones_test) > 0:
            fig.add_trace(go.Scatter(
                x=años_test,
                y=predicciones_test,
                mode='lines+markers',
                name='Predicciones del Modelo',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond')
            ))
            
            # Línea vertical separando train/test
            año_corte = 2021
            fig.add_vline(
                x=año_corte,
                line_dash="dot",
                line_color="red",
                annotation_text="Inicio Test",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=f"Tasa de Natalidad: {pais_seleccionado}",
            xaxis_title="Año",
            yaxis_title="Natalidad (nacimientos por 1000 hab)",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # MÉTRICAS DEL PAÍS
        # ============================================
        
        if len(predicciones_test) > 0 and len(df_pais_test) > 0:
            st.markdown("### 📊 Métricas de Predicción")
            
            # Calcular métricas
            y_real_pais = df_pais_test['Natalidad'].values
            y_pred_pais = np.array(predicciones_test)
            
            # Filtrar NaN si hay
            mask = ~np.isnan(y_pred_pais) & ~np.isnan(y_real_pais)
            y_real_pais_clean = y_real_pais[mask]
            y_pred_pais_clean = y_pred_pais[mask]
            
            if len(y_real_pais_clean) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                rmse_pais = np.sqrt(mean_squared_error(y_real_pais_clean, y_pred_pais_clean))
                mae_pais = mean_absolute_error(y_real_pais_clean, y_pred_pais_clean)
                r2_pais = r2_score(y_real_pais_clean, y_pred_pais_clean)
                mape_pais = np.mean(np.abs((y_real_pais_clean - y_pred_pais_clean) / y_real_pais_clean)) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{rmse_pais:.2f}", help="Root Mean Squared Error")
                
                with col2:
                    st.metric("MAE", f"{mae_pais:.2f}", help="Mean Absolute Error")
                
                with col3:
                    st.metric("R² Score", f"{r2_pais:.3f}", help="Coeficiente de Determinación")
                
                with col4:
                    st.metric("MAPE", f"{mape_pais:.1f}%", help="Mean Absolute Percentage Error")
                
                # Interpretación
                st.markdown("---")
                st.markdown("#### 💬 Interpretación")
                
                if r2_pais > 0.9:
                    st.success(f"🎯 **Excelente ajuste** - El modelo predice muy bien para {pais_seleccionado} (R² > 0.9)")
                elif r2_pais > 0.7:
                    st.info(f"✅ **Buen ajuste** - El modelo predice razonablemente bien para {pais_seleccionado} (R² > 0.7)")
                elif r2_pais > 0.5:
                    st.warning(f"⚠️ **Ajuste moderado** - Las predicciones tienen margen de mejora (R² > 0.5)")
                else:
                    st.error(f"❌ **Ajuste débil** - El modelo tiene dificultades con {pais_seleccionado} (R² < 0.5)")
                
                st.markdown(f"""
                - **Error promedio:** {mae_pais:.2f} nacimientos por 1000 habitantes
                - **Error porcentual:** {mape_pais:.1f}% de desviación en promedio
                - **Varianza explicada:** {r2_pais*100:.1f}% de la variabilidad es capturada por el modelo
                """)
                
                # ============================================
                # NUEVO: ANÁLISIS DE IMPORTANCIA DE FEATURES
                # ============================================
                
                st.markdown("---")
                st.markdown("### 🔍 ¿En qué se fijó el modelo para predecir?")
                st.markdown(f"Análisis de las variables que más influyeron en la predicción para **{pais_seleccionado}**")
                
                # Selector de año para analizar (si hay múltiples años en test)
                if len(años_test) > 1:
                    año_analizar = st.selectbox(
                        "Selecciona el año a analizar:",
                        options=años_test,
                        index=len(años_test)-1  # Último año por defecto
                    )
                    idx_analizar = años_test.index(año_analizar)
                else:
                    año_analizar = años_test[0]
                    idx_analizar = 0
                
                # Obtener las features escaladas para ese año
                X_analizar = X_test_pais_list[idx_analizar]
                pred_analizar = predicciones_test[idx_analizar]
                
                if X_analizar is not None and hasattr(model, 'feature_importances_'):
                    # Obtener importancias globales del modelo
                    importancias_globales = model.feature_importances_
                    
                    # Obtener valores de las features para este caso
                    valores_features = X_analizar
                    
                    # Calcular "contribución" de cada feature
                    # Usamos importancia * valor (simplificado, no es SHAP exacto pero es interpretable)
                    contribuciones = importancias_globales * valores_features
                    
                    # Calcular baseline (predicción promedio del modelo en train)
                    baseline = y_train.mean()
                    
                    # Crear DataFrame con la información
                    df_features = pd.DataFrame({
                        'Feature': feature_names,
                        'Valor_Escalado': valores_features,
                        'Importancia_Global': importancias_globales,
                        'Contribucion': contribuciones
                    })
                    
                    # Ordenar por contribución absoluta
                    df_features['Contribucion_Abs'] = df_features['Contribucion'].abs()
                    df_features = df_features.sort_values('Contribucion_Abs', ascending=False)
                    
                    # Top 15 features más influyentes
                    top_features = df_features.head(15).copy()
                    
                    # Determinar si empuja hacia arriba o abajo
                    top_features['Efecto'] = top_features['Contribucion'].apply(
                        lambda x: 'Aumenta Natalidad' if x > 0 else 'Disminuye Natalidad'
                    )
                    top_features['Color'] = top_features['Contribucion'].apply(
                        lambda x: '#2ecc71' if x > 0 else '#e74c3c'
                    )
                    
                    # GRÁFICO: Feature Impact (estilo LIME)
                    fig_impact = go.Figure()
                    
                    # Ordenar para que quede visual (positivos arriba, negativos abajo)
                    top_features_sorted = top_features.sort_values('Contribucion', ascending=True)
                    
                    fig_impact.add_trace(go.Bar(
                        y=top_features_sorted['Feature'],
                        x=top_features_sorted['Contribucion'],
                        orientation='h',
                        marker=dict(
                            color=top_features_sorted['Contribucion'],
                            colorscale=[[0, '#e74c3c'], [0.5, '#f0f0f0'], [1, '#2ecc71']],
                            line=dict(color='black', width=1)
                        ),
                        text=top_features_sorted['Contribucion'].apply(lambda x: f"{x:+.3f}"),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Contribución: %{x:.4f}<br><extra></extra>'
                    ))
                    
                    # Línea vertical en 0
                    fig_impact.add_vline(
                        x=0,
                        line_dash="dash",
                        line_color="black",
                        line_width=2
                    )
                    
                    fig_impact.update_layout(
                        title=f"Impacto de Variables en la Predicción - {pais_seleccionado} ({año_analizar})",
                        xaxis_title="Contribución a la Predicción",
                        yaxis_title="Variable",
                        height=600,
                        showlegend=False,
                        annotations=[
                            dict(
                                x=0.02,
                                y=1.05,
                                xref='paper',
                                yref='paper',
                                text='← Disminuye Natalidad',
                                showarrow=False,
                                font=dict(color='#e74c3c', size=12, family='Arial Black')
                            ),
                            dict(
                                x=0.98,
                                y=1.05,
                                xref='paper',
                                yref='paper',
                                text='Aumenta Natalidad →',
                                showarrow=False,
                                font=dict(color='#2ecc71', size=12, family='Arial Black')
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig_impact, use_container_width=True)
                    
                    # Explicación de las predicciones
                    st.markdown("#### 💡 Interpretación del Análisis")
                    
                    # Identificar top 3 positivas y negativas
                    top_positivas = top_features[top_features['Contribucion'] > 0].head(3)
                    top_negativas = top_features[top_features['Contribucion'] < 0].head(3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 📈 Variables que AUMENTAN la natalidad:")
                        if len(top_positivas) > 0:
                            for idx, row in top_positivas.iterrows():
                                st.markdown(f"- **{row['Feature']}** (+{row['Contribucion']:.3f})")
                        else:
                            st.markdown("- *Ninguna variable aumenta significativamente*")
                    
                    with col2:
                        st.markdown("##### 📉 Variables que DISMINUYEN la natalidad:")
                        if len(top_negativas) > 0:
                            for idx, row in top_negativas.iterrows():
                                st.markdown(f"- **{row['Feature']}** ({row['Contribucion']:.3f})")
                        else:
                            st.markdown("- *Ninguna variable disminuye significativamente*")
                    
                    # Resumen explicativo
                    st.info(f"""
                    **📊 Resumen de la Predicción:**
                    
                    - **Predicción del modelo:** {pred_analizar:.2f} nacimientos/1000 hab
                    - **Línea base (promedio):** {baseline:.2f} nacimientos/1000 hab
                    - **Desviación:** {pred_analizar - baseline:+.2f}
                    
                    El modelo consideró **{len(top_features)} variables principales** para hacer esta predicción.
                    Las barras verdes indican variables que empujan la predicción hacia arriba (más natalidad),
                    mientras que las rojas la empujan hacia abajo (menos natalidad).
                    """)
                    
                else:
                    st.warning("No se puede calcular el análisis de importancia para este modelo o predicción.")
                
        else:
            st.info("ℹ️ No hay datos de test disponibles para este país (todos los datos son de entrenamiento)")
    
    # ============================================
    # TAB 2: MÉTRICAS GENERALES (SIN CAMBIOS)
    # ============================================
    
    with tab2:
        st.markdown("### 📊 Rendimiento General del Modelo")
        st.markdown("Evaluación del modelo en todo el conjunto de prueba")
        
        with st.spinner("Calculando métricas generales..."):
            # Predecir todo el conjunto de test
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas generales
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse_general = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_general = mean_absolute_error(y_test, y_pred_test)
            r2_general = r2_score(y_test, y_pred_test)
            mape_general = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Mostrar métricas
        st.markdown("#### 🎯 Métricas del Conjunto de Test")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "RMSE",
                f"{rmse_general:.2f}",
                help="Error cuadrático medio"
            )
        
        with col2:
            st.metric(
                "MAE",
                f"{mae_general:.2f}",
                help="Error absoluto medio"
            )
        
        with col3:
            st.metric(
                "R² Score",
                f"{r2_general:.4f}",
                help="Proporción de varianza explicada"
            )
        
        with col4:
            st.metric(
                "MAPE",
                f"{mape_general:.1f}%",
                help="Error porcentual absoluto medio"
            )
        
        st.markdown("---")
        
        # ============================================
        # TAB 2: MÉTRICAS GENERALES
        # ============================================
        
        with tab2:
            st.markdown("### 📊 Rendimiento General del Modelo")
            st.markdown("Evaluación del modelo en todo el conjunto de prueba")
            
            with st.spinner("Calculando métricas generales..."):
                # Predecir todo el conjunto de test
                y_pred_test = model.predict(X_test)
                
                # Calcular métricas generales
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                rmse_general = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae_general = mean_absolute_error(y_test, y_pred_test)
                r2_general = r2_score(y_test, y_pred_test)
                mape_general = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            # Mostrar métricas
            st.markdown("#### 🎯 Métricas del Conjunto de Test")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RMSE",
                    f"{rmse_general:.2f}",
                    help="Error cuadrático medio - Penaliza más los errores grandes"
                )
            
            with col2:
                st.metric(
                    "MAE",
                    f"{mae_general:.2f}",
                    help="Error absoluto medio - Promedio de desviación"
                )
            
            with col3:
                st.metric(
                    "R² Score",
                    f"{r2_general:.4f}",
                    help="Proporción de varianza explicada (0-1, mayor es mejor)"
                )
            
            with col4:
                st.metric(
                    "MAPE",
                    f"{mape_general:.1f}%",
                    help="Error porcentual absoluto medio"
                )
            
            st.markdown("---")
            
            # ============================================
            # GRÁFICO: DISTRIBUCIÓN DE ERRORES
            # ============================================
            
            st.markdown("#### 📉 Distribución de Errores")
            
            residuos = y_test - y_pred_test
            
            fig_residuos = go.Figure()
            
            fig_residuos.add_trace(go.Histogram(
                x=residuos,
                nbinsx=50,
                marker_color='steelblue',
                opacity=0.7,
                name='Residuos'
            ))
            
            fig_residuos.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Error = 0"
            )
            
            fig_residuos.update_layout(
                title="Distribución de Residuos (Real - Predicho)",
                xaxis_title="Residuo",
                yaxis_title="Frecuencia",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_residuos, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Estadísticas de Residuos:**
                - Media: {residuos.mean():.3f}
                - Mediana: {np.median(residuos):.3f}
                - Desviación Estándar: {residuos.std():.3f}
                """)
            
            with col2:
                # Calcular porcentaje de predicciones dentro de ciertos rangos
                dentro_1 = (np.abs(residuos) <= 1).sum() / len(residuos) * 100
                dentro_2 = (np.abs(residuos) <= 2).sum() / len(residuos) * 100
                dentro_3 = (np.abs(residuos) <= 3).sum() / len(residuos) * 100
                
                st.success(f"""
                **Precisión por Rango:**
                - {dentro_1:.1f}% predicciones con error < 1
                - {dentro_2:.1f}% predicciones con error < 2
                - {dentro_3:.1f}% predicciones con error < 3
                """)
            
            st.markdown("---")
            
            # ============================================
            # GRÁFICO: REAL VS PREDICHO (SCATTER GENERAL)
            # ============================================
            
            st.markdown("#### 🎯 Real vs Predicho (Todo el Test Set)")
            
            # Muestrear si hay muchos puntos
            n_points = len(y_test)
            if n_points > 1000:
                indices = np.random.choice(n_points, 1000, replace=False)
                y_test_sample = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
                y_pred_sample = y_pred_test[indices]
            else:
                y_test_sample = y_test
                y_pred_sample = y_pred_test
            
            fig_scatter_general = go.Figure()
            
            fig_scatter_general.add_trace(go.Scatter(
                x=y_test_sample,
                y=y_pred_sample,
                mode='markers',
                marker=dict(
                    size=6,
                    color=np.abs(y_test_sample - y_pred_sample),
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Error Abs"),
                    opacity=0.6
                ),
                hovertemplate='<b>Real:</b> %{x:.2f}<br><b>Predicho:</b> %{y:.2f}<extra></extra>'
            ))
            
            # Línea de predicción perfecta
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig_scatter_general.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig_scatter_general.update_layout(
                title=f"Real vs Predicho - R² = {r2_general:.4f}",
                xaxis_title="Natalidad Real",
                yaxis_title="Natalidad Predicha",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter_general, use_container_width=True)
            
            # ============================================
            # TOP/BOTTOM PAÍSES POR ERROR
            # ============================================
            
            st.markdown("---")
            st.markdown("#### 🏆 Países con Mejor y Peor Predicción")
            
            # Calcular errores por país
            df_test_with_pred = df_test_original.copy()
            df_test_with_pred['Prediccion'] = y_pred_test
            df_test_with_pred['Error_Abs'] = np.abs(df_test_with_pred['Natalidad'] - df_test_with_pred['Prediccion'])
            
            # Agrupar por país
            errores_por_pais = df_test_with_pred.groupby('Pais').agg({
                'Error_Abs': 'mean',
                'Natalidad': 'mean'
            }).reset_index()
            errores_por_pais.columns = ['Pais', 'Error_Promedio', 'Natalidad_Promedio']
            errores_por_pais = errores_por_pais.sort_values('Error_Promedio')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### ✅ Mejores Predicciones (Menor Error)")
                top_5_mejores = errores_por_pais.head(10)
                st.dataframe(
                    top_5_mejores[['Pais', 'Error_Promedio', 'Natalidad_Promedio']].style.format({
                        'Error_Promedio': '{:.2f}',
                        'Natalidad_Promedio': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.markdown("##### ❌ Predicciones Más Difíciles (Mayor Error)")
                top_5_peores = errores_por_pais.tail(10)
                st.dataframe(
                    top_5_peores[['Pais', 'Error_Promedio', 'Natalidad_Promedio']].style.format({
                        'Error_Promedio': '{:.2f}',
                        'Natalidad_Promedio': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# ============================================
# PÁGINA: DATOS
# ============================================

elif pagina == "📁 Datos":
    st.title("📁 Exploración de Datos")
    st.markdown("---")
    
    # Tabs para organizar
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Vista Previa", "📊 Estadísticas", "🔍 Filtros", "⬇️ Descargar"])
    
    with tab1:
        st.subheader("Primeras filas del dataset procesado")
        
        # Selector de número de filas
        n_rows = st.slider("Número de filas a mostrar:", 5, 100, 10)
        
        st.dataframe(
            df.head(n_rows),
            use_container_width=True,
            height=400
        )
        
        st.markdown(f"**Total de filas:** {len(df):,} | **Columnas:** {len(df.columns)}")
    
    with tab2:
        st.subheader("Estadísticas Descriptivas")
        
        # Selector de columnas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        columnas_seleccionadas = st.multiselect(
            "Selecciona columnas:",
            options=columnas_numericas,
            default=columnas_numericas[:5] if len(columnas_numericas) >= 5 else columnas_numericas
        )
        
        if columnas_seleccionadas:
            st.dataframe(
                df[columnas_seleccionadas].describe(),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Selecciona al menos una columna")
    
    with tab3:
        st.subheader("Filtrar Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por continente
            continentes = ['Todos'] + sorted(df['Continente'].unique().tolist())
            continente_filtro = st.selectbox("Continente:", continentes)
            
            # Filtro por año
            años = sorted(df['Año'].unique().tolist())
            año_filtro = st.select_slider("Año:", options=años, value=(años[0], años[-1]))
        
        with col2:
            # Filtro por región
            if continente_filtro != 'Todos':
                regiones = ['Todas'] + sorted(df[df['Continente'] == continente_filtro]['Region'].unique().tolist())
            else:
                regiones = ['Todas'] + sorted(df['Region'].unique().tolist())
            
            region_filtro = st.selectbox("Región:", regiones)
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        if continente_filtro != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['Continente'] == continente_filtro]
        
        if region_filtro != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Region'] == region_filtro]
        
        df_filtrado = df_filtrado[
            (df_filtrado['Año'] >= año_filtro[0]) & 
            (df_filtrado['Año'] <= año_filtro[1])
        ]
        
        st.markdown(f"**Resultados:** {len(df_filtrado):,} filas")
        
        st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    with tab4:
        st.subheader("Descargar Datos")
        
        st.markdown("""
        Descarga el dataset procesado en formato CSV para análisis externos.
        """)
        
        # Botón de descarga
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar CSV Completo",
            data=csv,
            file_name=f"natalidad_procesado_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Descarga el dataset completo procesado"
        )
        
        st.info(f"📊 El archivo contendrá {len(df):,} filas y {len(df.columns)} columnas")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Predicción de Tasas de Natalidad Global</strong></p>
    <p>Proyecto de Ingeniería en Sistemas | Datos: Banco Mundial | Tecnología: Python + Streamlit</p>
</div>
""", unsafe_allow_html=True)