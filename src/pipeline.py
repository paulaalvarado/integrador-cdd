"""
Pipeline completo de procesamiento de datos
Replica EXACTAMENTE el flujo del notebook CuartaPresentacion.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import streamlit as st
from src.geography_utils import REGIONES_DICT


# ============================================
# PASO 1: CARGAR DATASET ORIGINAL
# ============================================

@st.cache_data
def cargar_datos(ruta_archivo='data/raw/merged_dataset.csv'):
    """
    Carga el dataset original desde CSV con validaciones básicas
    
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        # Intentar cargar con UTF-8
        df = pd.read_csv(ruta_archivo)
    except UnicodeDecodeError:
        # Si falla, probar con latin-1
        df = pd.read_csv(ruta_archivo, encoding='latin-1')
    
    # Validar columnas esenciales
    columnas_requeridas = ['Año', 'Pais', 'Natalidad']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        raise ValueError(f"❌ Faltan columnas esenciales: {columnas_faltantes}")
    
    return df


# ============================================
# PASO 2: LIMPIEZA Y NULOS
# ============================================

def limpieza_y_nulos(df, umbral_faltantes=60):
    """
    Realiza limpieza básica del dataset
    
    Args:
        df: Dataset original
        umbral_faltantes: Porcentaje máximo de valores faltantes permitido
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    df_limpio = df.copy()
    
    # 1. ELIMINAR COLUMNAS CON MUCHOS NULOS
    porcentaje_faltantes = (df_limpio.isnull().sum() / len(df_limpio)) * 100
    columnas_eliminar = porcentaje_faltantes[porcentaje_faltantes > umbral_faltantes].index.tolist()
    
    if columnas_eliminar:
        df_limpio = df_limpio.drop(columns=columnas_eliminar)
    
    # 2. ELIMINAR DUPLICADOS
    df_limpio = df_limpio.drop_duplicates(subset=['Pais', 'Año'], keep='first')
    
    # 3. ELIMINAR REGIONES GEOGRÁFICAS (no son países reales)
    regiones = [
        'Africa Eastern and Southern', 'Africa Western and Central',
        'Arab World', 'Caribbean small states', 'Central Europe and the Baltics',
        'Early-demographic dividend', 'East Asia & Pacific',
        'East Asia & Pacific (excluding high income)',
        'East Asia & Pacific (IDA & IBRD countries)', 'Euro area',
        'Europe & Central Asia', 'Europe & Central Asia (excluding high income)',
        'Europe & Central Asia (IDA & IBRD countries)', 'European Union',
        'Fragile and conflict affected situations',
        'Heavily indebted poor countries (HIPC)', 'High income', 'IBRD only',
        'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total',
        'Late-demographic dividend', 'Latin America & Caribbean',
        'Latin America & Caribbean (excluding high income)',
        'Latin America & the Caribbean (IDA & IBRD countries)',
        'Least developed countries: UN classification',
        'Low & middle income', 'Low income', 'Lower middle income',
        'Middle East, North Africa, Afghanistan & Pakistan',
        'Middle East, North Africa, Afghanistan & Pakistan (excluding high income)',
        'Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)',
        'Middle income', 'North America', 'Not classified', 'OECD members',
        'Other small states', 'Pacific island small states',
        'Post-demographic dividend', 'Pre-demographic dividend', 'Small states',
        'South Asia', 'South Asia (IDA & IBRD)', 'Sub-Saharan Africa',
        'Sub-Saharan Africa (excluding high income)',
        'Sub-Saharan Africa (IDA & IBRD countries)', 'Upper middle income',
        'World'
    ]
    
    df_limpio = df_limpio[~df_limpio['Pais'].isin(regiones)].copy()
    
    return df_limpio


# ============================================
# PASO 3: ELIMINAR DATA LEAKAGE
# ============================================

def limpiar_leakage(df_limpio):
    """
    Elimina variables con data leakage y filas con Natalidad faltante
    
    Args:
        df_limpio: Dataset después de limpieza básica
        
    Returns:
        pd.DataFrame: Dataset sin leakage
    """
    df_sin_leakage = df_limpio.copy()
    
    # 1. ELIMINAR FILAS CON NATALIDAD FALTANTE
    df_sin_leakage = df_sin_leakage.dropna(subset=['Natalidad'])
    
    # 2. ELIMINAR VARIABLES CON LEAKAGE DIRECTO
    variables_leakage_directo = [
        'TasaFertilidad',  # Es otra forma de medir natalidad
        'MortalidadInfantil',  # Necesita nacimientos para calcularse
        'MortalidadMenores5',
        'MortalidadNeonatal',
        'MortalidadMaterna',
        'PrevalenciaAnemiaEmbarazadas',
        'RatioDependenciaJovenes'
    ]
    
    columnas_eliminar = [var for var in variables_leakage_directo if var in df_sin_leakage.columns]
    if columnas_eliminar:
        df_sin_leakage = df_sin_leakage.drop(columns=columnas_eliminar)
    
    # 3. ELIMINAR VARIABLES CON >50% MISSING Y BAJA RELEVANCIA
    porcentaje_missing = (df_sin_leakage.isnull().sum() / len(df_sin_leakage)) * 100
    
    variables_high_missing = ['EsperanzaEscolaridad', 'GastoIDPorcPBI']
    columnas_eliminar_missing = [
        var for var in variables_high_missing 
        if var in df_sin_leakage.columns and porcentaje_missing.get(var, 0) > 50
    ]
    
    if columnas_eliminar_missing:
        df_sin_leakage = df_sin_leakage.drop(columns=columnas_eliminar_missing)
    
    return df_sin_leakage


# ============================================
# PASO 4: FEATURES TEMPORALES
# ============================================

def crear_features_temporales(df_sin_leakage):
    """
    Crea features temporales para capturar tendencias
    
    Args:
        df_sin_leakage: Dataset sin leakage
        
    Returns:
        pd.DataFrame: Dataset con features temporales
    """
    df_features = df_sin_leakage.copy()
    
    # 1. AÑOS DESDE 2000
    df_features['AñosDesde2000'] = df_features['Año'] - 2000
    
    # 2. DÉCADA
    df_features['Decada'] = (df_features['Año'] // 10) * 10
    
    # 3. CRISIS ECONÓMICA 2008
    df_features['CrisisEconomica2008'] = (
        (df_features['Año'] >= 2008) & (df_features['Año'] <= 2009)
    ).astype(int)
    
    # 4. PANDEMIA COVID-19
    df_features['PandemiaCOVID'] = (
        (df_features['Año'] >= 2020) & (df_features['Año'] <= 2022)
    ).astype(int)
    
    return df_features


# ============================================
# PASO 5: ASIGNAR REGIONES
# ============================================

def asignar_regiones(df_features):
    """
    Asigna continente y región a cada país
    
    Args:
        df_features: Dataset con features temporales
        
    Returns:
        pd.DataFrame: Dataset con columnas Continente y Region
    """
    df_con_regiones = df_features.copy()
    
    # Asignar usando el diccionario
    df_con_regiones['Continente'] = df_con_regiones['Pais'].map(
        lambda x: REGIONES_DICT.get(x, {}).get('Continente', 'Sin clasificar')
    )
    df_con_regiones['Region'] = df_con_regiones['Pais'].map(
        lambda x: REGIONES_DICT.get(x, {}).get('Region', 'Sin clasificar')
    )
    
    return df_con_regiones


# ============================================
# PIPELINE COMPLETO
# ============================================

@st.cache_data
def ejecutar_pipeline_completo(ruta_archivo='data/raw/merged_dataset.csv', 
                               umbral_faltantes=60):
    """
    Ejecuta TODO el pipeline de procesamiento en memoria
    
    Args:
        ruta_archivo: Ruta al CSV original
        umbral_faltantes: Porcentaje máximo de nulos permitido
        
    Returns:
        pd.DataFrame: Dataset completamente procesado y listo para usar
    """
    # PASO 1: Cargar datos originales
    df = cargar_datos(ruta_archivo)
    
    # PASO 2: Limpieza
    df = limpieza_y_nulos(df, umbral_faltantes)
    
    # PASO 3: Eliminar leakage
    df = limpiar_leakage(df)
    
    # PASO 4: Features temporales
    df = crear_features_temporales(df)
    
    # PASO 5: Asignar regiones
    df = asignar_regiones(df)
    
    return df


# ============================================
# PREPARACIÓN PARA MODELO
# ============================================

def preparar_para_modelo(df_procesado, año_corte=2021, test_size=0.2, random_state=42):
    """
    Prepara los datos para entrenamiento/predicción del modelo
    
    Args:
        df_procesado: Dataset procesado completo
        año_corte: Año que separa train/test
        test_size: Proporción de test (si no se usa año_corte)
        random_state: Semilla aleatoria
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Columnas a excluir de X (metadatos y target)
    columnas_excluir = ['Natalidad', 'Año', 'Pais', 'CodigoPais', 'Continente', 'Region']
    
    # Filtrar solo las que existen
    columnas_excluir_existentes = [col for col in columnas_excluir if col in df_procesado.columns]
    
    # Split temporal
    train_mask = df_procesado['Año'] <= año_corte
    test_mask = df_procesado['Año'] > año_corte
    
    df_train = df_procesado[train_mask].copy()
    df_test = df_procesado[test_mask].copy()
    
    # Separar X e y
    X_train = df_train.drop(columns=columnas_excluir_existentes)
    y_train = df_train['Natalidad']
    
    X_test = df_test.drop(columns=columnas_excluir_existentes)
    y_test = df_test['Natalidad']
    
    # Guardar nombres de features
    feature_names = X_train.columns.tolist()
    
    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, imputer


# ============================================
# UTILIDADES
# ============================================

def get_resumen_pipeline(df_original, df_procesado):
    """
    Genera un resumen del pipeline ejecutado
    
    Returns:
        dict: Resumen con estadísticas
    """
    return {
        'filas_original': len(df_original),
        'filas_procesado': len(df_procesado),
        'columnas_original': len(df_original.columns),
        'columnas_procesado': len(df_procesado.columns),
        'años_min': int(df_procesado['Año'].min()),
        'años_max': int(df_procesado['Año'].max()),
        'paises_unicos': df_procesado['Pais'].nunique(),
        'regiones_unicas': df_procesado['Region'].nunique(),
        'continentes': sorted(df_procesado['Continente'].unique().tolist())
    }


if __name__ == "__main__":
    # Test del pipeline
    print("🔄 Ejecutando pipeline completo...")
    df_final = ejecutar_pipeline_completo()
    
    if df_final is not None:
        print(f"✅ Pipeline completado")
        print(f"   📊 Shape final: {df_final.shape}")
        print(f"   📅 Años: {df_final['Año'].min()} - {df_final['Año'].max()}")
        print(f"   🌍 Países: {df_final['Pais'].nunique()}")
        print(f"   🗺️ Regiones: {df_final['Region'].nunique()}")