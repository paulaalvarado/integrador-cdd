"""
Visualizaciones de Altair basadas en CuartaPresentacion.ipynb
"""

import altair as alt
import pandas as pd
import numpy as np


# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

# Habilitar manejo de datasets grandes
alt.data_transformers.disable_max_rows()


# ============================================
# VIZ 1: EVOLUCIÓN TEMPORAL POR REGIÓN
# ============================================

def viz_evolucion_temporal_regiones(df):
    """
    Visualización 1: Evolución temporal de natalidad por región
    Replica la visualización del notebook
    
    Args:
        df: DataFrame procesado con columnas Año, Natalidad, Continente, Region
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Preparar datos agregados
    df_limpio = df.dropna(subset=['Continente', 'Region'])
    
    datos_agregados = df_limpio.groupby(['Año', 'Continente', 'Region']).agg({
        'Natalidad': 'mean',
        'Pais': 'count'
    }).reset_index()
    
    datos_agregados.columns = ['Año', 'Continente', 'Region', 'Natalidad_Promedio', 'Num_Paises']
    datos_agregados['Natalidad_Promedio'] = datos_agregados['Natalidad_Promedio'].round(2)
    
    # Selector de continente
    selector_continente = alt.selection_point(
        fields=['Continente'],
        bind=alt.binding_select(
            options=[None] + sorted(list(datos_agregados['Continente'].unique())),
            labels=['Todos'] + sorted(list(datos_agregados['Continente'].unique())),
            name='Filtrar por Continente: '
        ),
        value='América'
    )
    
    # Selector para highlight de línea
    hover_region_selection = alt.selection_point(
        fields=['Region'],
        on='mouseover',
        nearest=True,
        empty=False
    )
    
    # Selector para highlight de punto
    hover_point_selection = alt.selection_point(
        on='mouseover',
        nearest=True,
        empty=False
    )
    
    # Gráfico base: líneas por región
    base = alt.Chart(datos_agregados).mark_line(
        strokeWidth=2.5
    ).encode(
        x=alt.X('Año:O',
                axis=alt.Axis(
                    title='Año',
                    labelAngle=-45,
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        y=alt.Y('Natalidad_Promedio:Q',
                axis=alt.Axis(
                    title='Natalidad Promedio (nacimientos por 1000 hab)',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                ),
                scale=alt.Scale(zero=False)),
        color=alt.Color('Continente:N',
                      legend=alt.Legend(
                          title='Continente',
                          titleFontSize=13,
                          titleFontWeight='bold',
                          labelFontSize=11
                      )),
        detail='Region:N',
        opacity=alt.condition(hover_region_selection, alt.value(1), alt.value(0.1)),
        tooltip=[
            alt.Tooltip('Region:N', title='Región'),
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('Año:O', title='Año'),
            alt.Tooltip('Natalidad_Promedio:Q', title='Natalidad Promedio', format='.2f'),
            alt.Tooltip('Num_Paises:Q', title='Número de Países')
        ]
    ).transform_filter(
        selector_continente
    )
    
    # Puntos
    points = base.mark_circle(size=60).add_params(
        hover_region_selection,
        hover_point_selection
    )
    
    # Línea de tendencia
    tendencia_global = alt.Chart(datos_agregados).mark_line(
        strokeDash=[5, 5],
        strokeWidth=3,
        color='red',
        opacity=0.6
    ).encode(
        x='Año:O',
        y='mean(Natalidad_Promedio):Q'
    ).transform_filter(
        selector_continente
    )
    
    # Texto con región
    text_region = base.mark_text(
        align='left',
        dx=5,
        dy=-10,
        fontSize=12,
        fontWeight='bold'
    ).encode(
        text='Region:N',
        color=alt.value('black'),
        opacity=alt.condition(hover_point_selection, alt.value(1), alt.value(0))
    )
    
    # Combinar
    chart = (base + points + tendencia_global + text_region).add_params(
        selector_continente
    ).properties(
        width=1080,
        height=720,
        title={
            'text': 'Evolución Temporal de la Natalidad por Región Geográfica',
            'subtitle': [
                'Promedio de nacimientos por 1000 habitantes | Interactivo: Selecciona continente y pasa el mouse sobre las líneas',
                'Línea roja punteada: Tendencia promedio del continente seleccionado'
            ],
            'fontSize': 18,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


# ============================================
# VIZ 2: NATALIDAD VS VARIABLE POR REGIÓN
# ============================================

def viz_dinamica_natalidad_vs_variable_region(df, variable_x, anio, usar_densidad=False, continentes_resaltados=None):
    """
    Scatter interactivo: Natalidad vs variable seleccionada (por país y año)
    Colorea y agrupa por continente.
    - Cada punto = país.
    - Eje X = variable elegida.
    - Eje Y = natalidad.
    - Colores por continente.
    - Click o multiselect: selecciona continentes (resalta todos los países de esa región).
    """

    import altair as alt
    import pandas as pd

    # --- Filtrado básico ---
    df_plot = df.copy()
    df_plot = df_plot[df_plot["Año"] >= 2000]
    df_plot = df_plot[df_plot["Año"] == anio]
    df_plot = df_plot.dropna(subset=["Pais", "Continente", "Natalidad", variable_x])

    if df_plot.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    # --- Tamaño opcional por densidad ---
    if usar_densidad and "DensidadPoblacional" in df_plot.columns:
        size_encoding = alt.Size(
            "DensidadPoblacional:Q",
            title="Densidad poblacional (hab/km²)",
            scale=alt.Scale(range=[20, 400]),
        )
        subtam = " | Tamaño indica densidad de población."
    else:
        size_encoding = alt.value(60)
        subtam = ""

    # --- Multi-selección de continentes ---
    sel_click = alt.selection_point(
        fields=["Continente"], toggle=True, on="click", empty="none"
    )

    # --- Colores fijos por continente ---
    color_scale = alt.Scale(
        domain=["África", "América", "Asia", "Europa", "Oceanía"],
        range=["#e45756", "#f58518", "#72b7b2", "#54a24b", "#eeca3b"],
    )

    titulo_x = variable_x.replace("_", " ")

    # --- Encodings comunes ---
    common_encodings = dict(
        x=alt.X(f"{variable_x}:Q", title=f"{titulo_x} (valor)"),
        y=alt.Y("Natalidad:Q", title="Natalidad (nacimientos por 1000 hab)"),
        size=size_encoding,
        tooltip=[
            alt.Tooltip("Pais:N", title="País"),
            alt.Tooltip("Continente:N", title="Continente"),
            alt.Tooltip("Año:O", title="Año"),
            alt.Tooltip("Natalidad:Q", format=".2f", title="Natalidad"),
            alt.Tooltip(f"{variable_x}:Q", format=".2f", title=titulo_x),
        ]
        + (
            [
                alt.Tooltip(
                    "DensidadPoblacional:Q",
                    format=".1f",
                    title="Densidad poblacional (hab/km²)",
                )
            ]
            if usar_densidad and "DensidadPoblacional" in df_plot.columns
            else []
        ),
    )

    # --- Capa base: todos los países en gris ---
    base = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.4, color="lightgray")
        .encode(**common_encodings)
    )

    # --- Capa resaltada por click (multi selección de continentes) ---
    highlight_click = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.9)
        .encode(
            color=alt.Color("Continente:N", scale=color_scale, legend=None),
            **common_encodings,
        )
        .add_params(sel_click)
        .transform_filter(sel_click)
    )

    layers = [base, highlight_click]

    # --- Capa adicional por multiselect desde Streamlit ---
    if continentes_resaltados:
        if not isinstance(continentes_resaltados, list):
            continentes_resaltados = [continentes_resaltados]

        highlight_dropdown = (
            alt.Chart(df_plot)
            .mark_circle(opacity=0.9)
            .encode(
                color=alt.Color("Continente:N", scale=color_scale, legend=None),
                **common_encodings,
            )
            .transform_filter(
                alt.FieldOneOfPredicate(field="Continente", oneOf=continentes_resaltados)
            )
        )
        layers.append(highlight_dropdown)

    chart = (
        alt.layer(*layers)
        .properties(
            width=900,
            height=550,
            title={
                "text": f"Natalidad vs {titulo_x} en {anio} (por país, agrupado por continente)",
                "subtitle": [
                    "Cada punto representa un país. Click o seleccioná un continente para resaltarlo.",
                    "Podés seleccionar varios continentes a la vez." + subtam,
                ],
                "fontSize": 16,
                "fontWeight": "bold",
                "anchor": "start",
                "subtitleFontSize": 12,
                "subtitleColor": "gray",
            },
        )
        .configure_axis(gridOpacity=0.3, labelFontSize=11, titleFontSize=13)
        .configure_view(strokeWidth=0)
    )

    return chart

def viz_correlaciones_interactivas(df):
    """
    Visualización 2: Scatter plot interactivo de correlaciones
    
    Args:
        df: DataFrame procesado
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Variables socioeconómicas clave
    variables_analisis = {
        'EsperanzaVida': 'Esperanza de Vida (años)',
        'PIB_per_capita': 'PIB per cápita (USD)',
        'Urbanizacion': 'Urbanización (%)',
        'GastoSalud': 'Gasto en Salud (% PIB)',
        'AccesoEducacion': 'Acceso a Educación (%)',
        'Desempleo': 'Desempleo (%)',
        'AccesoAguaPotable': 'Acceso a Agua Potable (%)',
        'MujeresParlamento': 'Mujeres en Parlamento (%)',
    }
    
    # Filtrar solo variables disponibles
    variables_disponibles = {
        var: label for var, label in variables_analisis.items()
        if var in df.columns
    }
    
    # Preparar datos (últimos 5 años)
    columnas_necesarias = ['Año', 'Pais', 'Natalidad', 'Continente'] + list(variables_disponibles.keys())
    df_viz = df[columnas_necesarias].dropna(subset=['Natalidad'])
    
    año_max = df_viz['Año'].max()
    df_viz = df_viz[df_viz['Año'] >= año_max - 4].copy()
    
    # Transformar a formato long
    df_long = df_viz.melt(
        id_vars=['Año', 'Pais', 'Natalidad', 'Continente'],
        value_vars=list(variables_disponibles.keys()),
        var_name='variable',
        value_name='valor'
    )
    
    # Limpiar datos
    df_clean = df_long.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['valor', 'Natalidad'])
    
    # Calcular correlaciones (compatible con pandas antiguo)
    def calc_corr(g):
        return g['valor'].corr(g['Natalidad'])
    
    df_corr = df_clean.groupby('variable').apply(calc_corr).reset_index(name='correlation')
    
    # Calcular líneas de regresión
    def get_reg_line(g):
        m, b = np.polyfit(g['valor'], g['Natalidad'], 1)
        x_min, x_max = g['valor'].min(), g['valor'].max()
        return pd.DataFrame({
            'valor': [x_min, x_max],
            'Natalidad_pred': [m * x_min + b, m * x_max + b]
        })
    
    df_reg_lines = df_clean.groupby('variable').apply(get_reg_line).reset_index(drop=True)
    
    # Selectores
    variable_input = alt.binding_select(
        options=list(variables_disponibles.keys()),
        name='Variable a comparar: '
    )
    
    variable_selection = alt.selection_point(
        fields=['variable'],
        bind=variable_input,
        value=list(variables_disponibles.keys())[0]
    )
    
    selector_continente = alt.selection_point(
        fields=['Continente'],
        bind='legend',
        on='click'
    )
    
    color_scale = alt.Scale(
        domain=['África', 'América', 'Asia', 'Europa', 'Oceanía'],
        range=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    )
    
    # Scatter plot
    scatter = alt.Chart(df_clean).mark_circle(
        size=100,
        opacity=0.7
    ).encode(
        x=alt.X('valor:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    titleFontSize=13,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        y=alt.Y('Natalidad:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    title='Natalidad (nacimientos por 1000 hab)',
                    titleFontSize=13,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        color=alt.condition(
            selector_continente,
            alt.Color('Continente:N',
                      scale=color_scale,
                      legend=alt.Legend(
                          title='Continente (click para filtrar)',
                          titleFontSize=12,
                          titleFontWeight='bold',
                          labelFontSize=11,
                          orient='right'
                      )),
            alt.value('lightgray')
        ),
        opacity=alt.condition(selector_continente, alt.value(0.8), alt.value(0.1)),
        tooltip=[
            alt.Tooltip('Pais:N', title='País'),
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('Año:O', title='Año'),
            alt.Tooltip('Natalidad:Q', title='Natalidad', format='.2f'),
            alt.Tooltip('valor:Q', title='Valor Variable', format='.2f')
        ]
    ).add_params(
        variable_selection,
        selector_continente
    ).transform_filter(
        variable_selection
    )
    
    # Línea de regresión
    regression = alt.Chart(df_reg_lines).mark_line(
        color='white',
        strokeWidth=3,
        strokeDash=[5, 5]
    ).encode(
        x=alt.X('valor:Q'),
        y=alt.Y('Natalidad_pred:Q', title='Natalidad')
    ).add_params(
        variable_selection
    ).transform_filter(
        variable_selection
    )
    
    # Texto de correlación
    correlation_text = alt.Chart(df_corr).mark_text(
        align='left',
        baseline='top',
        dx=10,
        dy=10,
        fontSize=14,
        fontWeight='bold',
        color='darkred'
    ).transform_filter(
        variable_selection
    ).transform_calculate(
        correlation_label='"Correlación: " + format(datum.correlation, ".3f")'
    ).encode(
        text='correlation_label:N',
        x=alt.value(10),
        y=alt.value(10)
    )
    
    # Combinar
    chart = (scatter + regression + correlation_text).properties(
        width=1080,
        height=720,
        title={
            'text': 'Explorador de Correlaciones: Variables vs Natalidad',
            'subtitle': [
                'Selecciona una variable para explorar su relación con la natalidad',
                'Click para filtrar por continente | Línea Negra: tendencia lineal'
            ],
            'fontSize': 16,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 11,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    ).interactive()
    
    return chart

# ============================================
# VIZ 3: MAPA MUNDIAL CON SLIDER (CORREGIDO)
# ============================================

def viz_mapa_mundial_natalidad(df):
    """
    Visualización 3: Mapa mundial interactivo con slider de años
    VERSIÓN FUNCIONAL DEL NOTEBOOK - Usa múltiples capas por año
    
    Args:
        df: DataFrame procesado (df_con_regiones del notebook)
        
    Returns:
        alt.Chart: Gráfico de Altair con mapa mundial
    """
    from vega_datasets import data
    
    # Cargar geodata
    countries_url = alt.topo_feature(data.world_110m.url, 'countries')
    
    # Mapeo de países a IDs (COMPLETO del notebook)
    pais_a_id = {
        # Originales
        'Afghanistan': 4, 'Albania': 8, 'Algeria': 12, 'Angola': 24,
        'Argentina': 32, 'Armenia': 51, 'Australia': 36, 'Austria': 40,
        'Azerbaijan': 31, 'Bahamas': 44, 'Bangladesh': 50, 'Belarus': 112,
        'Belgium': 56, 'Belize': 84, 'Benin': 204, 'Bhutan': 64,
        'Bolivia': 68, 'Bosnia and Herzegovina': 70, 'Botswana': 72,
        'Brazil': 76, 'Brunei': 96, 'Bulgaria': 100, 'Burkina Faso': 854,
        'Burundi': 108, 'Cambodia': 116, 'Cameroon': 120, 'Canada': 124,
        'Central African Republic': 140, 'Chad': 148, 'Chile': 152,
        'China': 156, 'Colombia': 170, 'Congo': 178, 'Costa Rica': 188,
        'Croatia': 191, 'Cuba': 192, 'Cyprus': 196, 'Czech Republic': 203,
        'Democratic Republic of Congo': 180, 'Denmark': 208, 'Djibouti': 262,
        'Dominican Republic': 214, 'Ecuador': 218, 'Egypt': 818,
        'El Salvador': 222, 'Equatorial Guinea': 226, 'Eritrea': 232,
        'Estonia': 233, 'Ethiopia': 231, 'Fiji': 242, 'Finland': 246,
        'France': 250, 'Gabon': 266, 'Gambia': 270, 'Georgia': 268,
        'Germany': 276, 'Ghana': 288, 'Greece': 300, 'Guatemala': 320,
        'Guinea': 324, 'Guinea-Bissau': 624, 'Guyana': 328, 'Haiti': 332,
        'Honduras': 340, 'Hungary': 348, 'Iceland': 352, 'India': 356,
        'Indonesia': 360, 'Iran': 364, 'Iraq': 368, 'Ireland': 372,
        'Israel': 376, 'Italy': 380, 'Ivory Coast': 384, 'Jamaica': 388,
        'Japan': 392, 'Jordan': 400, 'Kazakhstan': 398, 'Kenya': 404,
        'Korea, Rep.': 410, 'Kuwait': 414, 'Kyrgyzstan': 417, 'Laos': 418,
        'Latvia': 428, 'Lebanon': 422, 'Lesotho': 426, 'Liberia': 430,
        'Libya': 434, 'Lithuania': 440, 'Luxembourg': 442, 'Madagascar': 450,
        'Malawi': 454, 'Malaysia': 458, 'Mali': 466, 'Mauritania': 478,
        'Mauritius': 480, 'Mexico': 484, 'Moldova': 498, 'Mongolia': 496,
        'Montenegro': 499, 'Morocco': 504, 'Mozambique': 508, 'Myanmar': 104,
        'Namibia': 516, 'Nepal': 524, 'Netherlands': 528, 'New Zealand': 554,
        'Nicaragua': 558, 'Niger': 562, 'Nigeria': 566, 'Norway': 578,
        'Oman': 512, 'Pakistan': 586, 'Panama': 591, 'Papua New Guinea': 598,
        'Paraguay': 600, 'Peru': 604, 'Philippines': 608, 'Poland': 616,
        'Portugal': 620, 'Qatar': 634, 'Romania': 642, 'Russia': 643,
        'Rwanda': 646, 'Saudi Arabia': 682, 'Senegal': 686, 'Serbia': 688,
        'Sierra Leone': 694, 'Singapore': 702, 'Slovakia': 703, 'Slovenia': 705,
        'Solomon Islands': 90, 'Somalia': 706, 'South Africa': 710,
        'South Sudan': 728, 'Spain': 724, 'Sri Lanka': 144, 'Sudan': 729,
        'Suriname': 740, 'Swaziland': 748, 'Eswatini': 748, 'Sweden': 752,
        'Switzerland': 756, 'Syria': 760, 'Tajikistan': 762, 'Tanzania': 834,
        'Thailand': 764, 'Togo': 768, 'Trinidad and Tobago': 780,
        'Tunisia': 788, 'Turkey': 792, 'Turkmenistan': 795, 'Uganda': 800,
        'Ukraine': 804, 'United Arab Emirates': 784, 'United Kingdom': 826,
        'United States': 840, 'Uruguay': 858, 'Uzbekistan': 860,
        'Vanuatu': 548, 'Venezuela': 862, 'Vietnam': 704, 'Yemen': 887,
        'Zambia': 894, 'Zimbabwe': 716,
        'Bahamas, The': 44, 'Brunei Darussalam': 96, 'Congo, Dem. Rep.': 180,
        'Congo, Rep.': 178, "Cote d'Ivoire": 384, 'Czechia': 203,
        'Egypt, Arab Rep.': 818, 'Gambia, The': 270, 'Iran, Islamic Rep.': 364,
        'Kyrgyz Republic': 417, 'Lao PDR': 418, 'Russian Federation': 643,
        'Slovak Republic': 703, 'Syrian Arab Republic': 760, 'Turkiye': 792,
        'Venezuela, RB': 862, 'Viet Nam': 704, 'Yemen, Rep.': 887,
        # Extras y territorios
        'American Samoa': 16, 'Andorra': 20, 'Antigua and Barbuda': 28, 'Aruba': 533,
        'Bahrain': 48, 'Barbados': 52, 'Bermuda': 60, 'British Virgin Islands': 92,
        'Cabo Verde': 132, 'Cayman Islands': 136, 'Comoros': 174, 'Curacao': 531,
        'Dominica': 212, 'Faroe Islands': 234, 'French Polynesia': 258, 'Gibraltar': 292,
        'Greenland': 304, 'Grenada': 308, 'Guam': 316, 'Hong Kong SAR, China': 344,
        'Isle of Man': 833, 'Kiribati': 296, "Korea, Dem. People's Rep.": 408,
        'Kosovo': -99, 'Liechtenstein': 438, 'Macao SAR, China': 446, 'Maldives': 462,
        'Malta': 470, 'Marshall Islands': 584, 'Micronesia, Fed. Sts.': 583, 'Monaco': 492,
        'Nauru': 520, 'New Caledonia': 540, 'North Macedonia': 807,
        'Northern Mariana Islands': 580, 'Palau': 585, 'Puerto Rico (US)': 630,
        'Samoa': 882, 'San Marino': 674, 'Sao Tome and Principe': 678, 'Seychelles': 690,
        'Sint Maarten (Dutch part)': 534, 'St. Kitts and Nevis': 659, 'St. Lucia': 662,
        'St. Martin (French part)': 663, 'St. Vincent and the Grenadines': 670,
        'Timor-Leste': 626, 'Tonga': 776, 'Turks and Caicos Islands': 796,
        'Tuvalu': 798, 'Virgin Islands (U.S.)': 850, 'West Bank and Gaza': 275,
        'Channel Islands': 830,
    }
    
    # Agregar ID al dataset
    df_con_regiones = df.copy()
    df_con_regiones['id'] = df_con_regiones['Pais'].map(pais_a_id)
    
    # Filtrar solo países con ID
    df_mapa = df_con_regiones[df_con_regiones['id'].notna()].copy()
    
    # Calcular estadísticas por año
    stats_por_año = df_mapa.groupby('Año')['Natalidad'].agg(['mean', 'min', 'max']).reset_index()
    
    # Asegurar tipos correctos
    df_mapa['Año'] = df_mapa['Año'].astype(int)
    stats_por_año['Año'] = stats_por_año['Año'].astype(int)
    
    años_únicos = sorted(df_mapa['Año'].unique())
    
    # Slider
    slider = alt.binding_range(
        min=int(años_únicos[0]),
        max=int(años_únicos[-1]),
        step=1,
        name='Año: '
    )
    
    year_param = alt.param(
        name='year',
        value=int(años_únicos[-1]),
        bind=slider
    )
    
    # Escala de colores
    color_scale = alt.Scale(
        domain=[5, 15, 25, 35, 45],
        range=['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
        type='threshold'
    )
    
    # Mapa base gris
    background = alt.Chart(countries_url).mark_geoshape(
        fill='#e0e0e0',
        stroke='white',
        strokeWidth=0.5
    ).project(
        type='naturalEarth1'
    ).properties(
        width=1080,
        height=720
    )
    
    # CREAR CAPAS POR AÑO (método del notebook que funciona)
    data_layers = []
    for año in años_únicos:
        df_año = df_mapa[df_mapa['Año'] == año][['id', 'Pais', 'Continente', 'Region', 'Año', 'Natalidad']].copy()
        
        layer = alt.Chart(countries_url).mark_geoshape(
            stroke='white',
            strokeWidth=0.5
        ).encode(
            color=alt.Color(
                'Natalidad:Q',
                scale=color_scale,
                legend=None
            ),
            tooltip=[
                alt.Tooltip('Pais:N', title='País'),
                alt.Tooltip('Continente:N', title='Continente'),
                alt.Tooltip('Region:N', title='Región'),
                alt.Tooltip('Natalidad:Q', title='Natalidad', format='.2f')
            ]
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(
                data=df_año,
                key='id',
                fields=['Pais', 'Continente', 'Region', 'Natalidad']
            )
        ).transform_filter(
            f'year == {año}'
        ).project(
            type='naturalEarth1'
        )
        
        data_layers.append(layer)
    
    # Capa dummy para la leyenda permanente
    legend_dummy = alt.Chart(df_mapa).mark_circle(opacity=0).encode(
        color=alt.Color(
            'Natalidad:Q',
            scale=color_scale,
            legend=alt.Legend(
                title='Natalidad (nacimientos/1000 hab)',
                titleFontSize=12,
                titleFontWeight='bold',
                labelFontSize=10
            )
        )
    )
    
    # Combinar todas las capas
    all_layers = [background] + [legend_dummy] + data_layers
    mapa_completo = alt.layer(*all_layers).properties(
        width=1080,
        height=720
    ).add_params(
        year_param
    )
    
    # Texto con estadísticas
    text_stats = alt.Chart(stats_por_año).mark_text(
        align='left',
        baseline='top',
        dx=10,
        dy=10,
        fontSize=13,
        fontWeight='bold',
        color='black'
    ).encode(
        text='label:N'
    ).transform_filter(
        'datum.Año == year'
    ).transform_calculate(
        label='toString(datum.Año) + " | Media Global: " + format(datum.mean, ".1f") + " | Rango: [" + format(datum.min, ".1f") + " - " + format(datum.max, ".1f") + "]"'
    ).properties(
        width=900,
        height=50
    ).add_params(
        year_param
    )
    
    # Combinar todo
    chart = (mapa_completo & text_stats).properties(
        title={
            'text': 'Evolución de la Natalidad Mundial',
            'subtitle': 'Usa el slider para explorar por año | Pasa el mouse sobre países para más información',
            'fontSize': 18,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_view(
        strokeWidth=0
    )
    
    return chart



# ============================================
# VIZ 4: DISTRIBUCIÓN POR CONTINENTE (OPCIONAL)
# ============================================

def viz_distribucion_continentes(df, year=None):
    """
    Visualización 3: Boxplot de distribución por continente
    
    Args:
        df: DataFrame procesado
        year: Año específico (si None, usa el más reciente)
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Usar año más reciente si no se especifica
    if year is None:
        year = int(df['Año'].max())
    
    # Filtrar datos
    df_year = df[df['Año'] == year].copy()
    df_year = df_year.dropna(subset=['Natalidad', 'Continente'])
    
    # Boxplot por continente
    chart = alt.Chart(df_year).mark_boxplot(
        size=50
    ).encode(
        x=alt.X('Continente:N',
                axis=alt.Axis(
                    title='Continente',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=12,
                    labelAngle=-45
                )),
        y=alt.Y('Natalidad:Q',
                axis=alt.Axis(
                    title='Tasa de Natalidad (por 1000 hab)',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                ),
                scale=alt.Scale(zero=False)),
        color=alt.Color('Continente:N',
                       scale=alt.Scale(
                           domain=['África', 'América', 'Asia', 'Europa', 'Oceanía'],
                           range=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
                       ),
                       legend=None),
        tooltip=[
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('min(Natalidad):Q', title='Mínimo', format='.2f'),
            alt.Tooltip('q1(Natalidad):Q', title='Q1', format='.2f'),
            alt.Tooltip('median(Natalidad):Q', title='Mediana', format='.2f'),
            alt.Tooltip('q3(Natalidad):Q', title='Q3', format='.2f'),
            alt.Tooltip('max(Natalidad):Q', title='Máximo', format='.2f')
        ]
    ).properties(
        width=800,
        height=500,
        title={
            'text': f'Distribución de Natalidad por Continente ({year})',
            'subtitle': 'Boxplot mostrando mediana, cuartiles y valores atípicos',
            'fontSize': 16,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )
    
    return chart

# ============================================
# VIZ 5: EVOLUCIÓN DE NATALIDAD POR PAÍS 
# ============================================

def viz_evolucion_paises_highlight(df, paises_seleccionados=None, pais_resaltado=None):
    """
    Gráfico de líneas de natalidad por país, con posibilidad de:
    - Filtrar por un subconjunto de países (paises_seleccionados)
    - Resaltar un país específico (pais_resaltado)
    Si no se pasan parámetros extras, muestra todos los países y resalta uno por defecto.
    """
    # Limpiar datos básicos
    df_clean = df.dropna(subset=['Pais', 'Natalidad', 'Año']).copy()
    df_clean['Año'] = df_clean['Año'].astype(int)

    # (Opcional) solo años desde 2000
    df_clean = df_clean[df_clean['Año'] >= 2000]

    # Si se pasó una lista de países, filtramos
    if paises_seleccionados:
        df_plot = df_clean[df_clean['Pais'].isin(paises_seleccionados)].copy()
    else:
        df_plot = df_clean.copy()

    if df_plot.empty:
        # Devuelve un gráfico vacío pero válido
        return alt.Chart(pd.DataFrame({'Año': [], 'Natalidad': [], 'Pais': []})).mark_line()

    # Determinar país a resaltar
    paises_disponibles = df_plot['Pais'].unique().tolist()
    if not paises_disponibles:
        return alt.Chart(pd.DataFrame({'Año': [], 'Natalidad': [], 'Pais': []})).mark_line()

    if (pais_resaltado is None) or (pais_resaltado not in paises_disponibles):
        # Si no se especifica o no está en el subset, usamos uno por defecto
        pais_resaltado = 'Argentina' if 'Argentina' in paises_disponibles else paises_disponibles[0]

    # Condición de highlight
    highlight = alt.datum.Pais == pais_resaltado

    # Líneas
    lines = alt.Chart(df_plot).mark_line(strokeWidth=2).encode(
        x=alt.X(
            'Año:O',
            axis=alt.Axis(
                title='Año',
                labelAngle=-45,
                titleFontSize=14,
                titleFontWeight='bold',
                labelFontSize=11
            )
        ),
        y=alt.Y(
            'Natalidad:Q',
            axis=alt.Axis(
                title='Natalidad (nacimientos por 1000 hab)',
                titleFontSize=14,
                titleFontWeight='bold',
                labelFontSize=11
            ),
            scale=alt.Scale(zero=False)
        ),
        detail='Pais:N',
        color=alt.condition(
            highlight,
            alt.Color('Pais:N', legend=None),
            alt.value('lightgray')
        ),
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
    )

    # Puntos con tooltip
    points = alt.Chart(df_plot).mark_circle(size=50).encode(
        x='Año:O',
        y='Natalidad:Q',
        color=alt.condition(
            highlight,
            alt.Color('Pais:N', legend=None),
            alt.value('lightgray')
        ),
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.3)),
        tooltip=[
            alt.Tooltip('Pais:N', title='País'),
            alt.Tooltip('Año:O', title='Año'),
            alt.Tooltip('Natalidad:Q', title='Natalidad', format='.2f')
        ]
    )

    chart = (lines + points).properties(
        width=1080,
        height=720,
        title={
            'text': 'Evolución de la Natalidad por País',
            'subtitle': [
                'Un país se resalta y el resto queda en gris como contexto',
                'Más adelante, la app podrá pasar un conjunto de países a mostrar'
            ],
            'fontSize': 18,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )

    return chart

# ============================================
# VIZ 3: NATALIDAD VS VARIABLES  
# ============================================

def viz_dinamica_natalidad_vs_variable(df):
    import streamlit as st
    import altair as alt

    st.markdown("#### ⚙️ Configuración del gráfico")

    # 1️⃣ Definir categorías de variables (solo usamos las que existan)
    variables_por_categoria = {
        "Salud": [
            "EsperanzaVida", "MortalidadInfantil", "MortalidadMaterna",
            "AccesoAguaPotable", "AccesoSaneamientoBasico",
            "GastoSalud", "ContaminacionAirePM25"
        ],
        "Económicas": [
            "PIB_per_capita", "Desempleo", "InflacionAnual",
            "PobrezaExtremaDolarDiario", "DesigualdadIngresos"
        ],
        "Demográficas / Población": [
            "TasaFertilidad", "PoblacionTotal", "DensidadPoblacional",
            "Urbanizacion", "CrecimientoPoblacional"
        ],
        "Educación / Género": [
            "AccesoEducacion", "MatriculacionPrimaria",
            "MujeresParlamento"
        ]
    }

    # Filtrar solo las variables que existan realmente en el dataset
    variables_por_categoria = {
        cat: [v for v in vars_ if v in df.columns]
        for cat, vars_ in variables_por_categoria.items()
        if any(v in df.columns for v in vars_)
    }

    if not variables_por_categoria:
        st.warning("No hay variables adecuadas en el dataset.")
        return None

    # 2️⃣ Selección de categoría
    categoria = st.selectbox("Categoría de variables:", list(variables_por_categoria.keys()))
    variables_categoria = variables_por_categoria[categoria]

    # Checkbox: mostrar todas las variables de la categoría
    mostrar_todas = st.checkbox(f"Mostrar todas las variables de {categoria}")

    # Si no muestra todas, se selecciona solo una variable
    if not mostrar_todas:
        variable_x = st.selectbox("Variable a comparar con Natalidad:", variables_categoria)
        variables_a_graficar = [variable_x]
    else:
        variables_a_graficar = variables_categoria

    # 3️⃣ Slider de año
    anio = st.slider(
        "Seleccioná el año:",
        int(df["Año"].min()),
        int(df["Año"].max()),
        2010,
        step=1
    )

    # 4️⃣ Checkbox para usar densidad poblacional
    usar_densidad = st.checkbox("Usar tamaño según Densidad Poblacional", value=False)

    # 5️⃣ Selección de países a resaltar
    paises_disponibles = sorted(df["Pais"].dropna().unique().tolist())
    paises_resaltados = st.multiselect(
        "Seleccioná uno o varios países para resaltar:",
        options=paises_disponibles,
    )

    # 6️⃣ Filtrar datos según el año
    df_plot = df[df["Año"] == anio].copy()

    # Crear los gráficos (uno por variable)
    charts = []
    for var in variables_a_graficar:
        sub = df_plot.dropna(subset=["Natalidad", var])
        if sub.empty:
            continue

        # Tamaño según densidad o fijo
        size_enc = (
            alt.Size("DensidadPoblacional:Q",
                     scale=alt.Scale(range=[30, 400]),
                     title="Densidad poblacional (hab/km²)")
            if usar_densidad and "DensidadPoblacional" in df.columns
            else alt.value(60)
        )

        # Color: si mostramos todas, diferenciamos por variable; sino resaltamos países
        if mostrar_todas:
            color_enc = alt.Color("variable:N",
                                  scale=alt.Scale(scheme="category10"),
                                  title="Variable")
        else:
            # Si hay países resaltados
            color_enc = alt.condition(
                alt.FieldOneOfPredicate(field="Pais", oneOf=paises_resaltados)
                if paises_resaltados else alt.datum.Pais != "",
                alt.value("#ff7f0e"),
                alt.value("lightgray")
            )

        # Crear chart
        chart = (
            alt.Chart(sub)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X(f"{var}:Q", title=f"{var.replace('_', ' ')}"),
                y=alt.Y("Natalidad:Q", title="Natalidad (nacimientos por 1000 hab)"),
                color=color_enc,
                size=size_enc,
                tooltip=[
                    alt.Tooltip("Pais:N", title="País"),
                    alt.Tooltip("Año:O", title="Año"),
                    alt.Tooltip("Natalidad:Q", format=".2f", title="Natalidad"),
                    alt.Tooltip(f"{var}:Q", format=".2f", title=var.replace('_', ' '))
                ],
            )
        )

        if mostrar_todas:
            chart = chart.transform_calculate(variable=f"'{var}'")

        charts.append(chart)

    # Si no hay datos, mostramos advertencia
    if not charts:
        st.warning("No hay datos para las variables seleccionadas.")
        return None

    # Combinar todos los charts (si hay varios)
    final_chart = alt.layer(*charts).properties(
        width=850,
        height=500,
        title=f"Natalidad vs {' / '.join(variables_a_graficar[:3])}{'...' if len(variables_a_graficar) > 3 else ''} ({anio})"
    )

    st.altair_chart(final_chart, use_container_width=True)
    return final_chart


# ============================================
# UTILIDADES
# ============================================

def get_available_visualizations():
    """Devuelve las visualizaciones disponibles, en orden deseado"""
    return [
        {
            "id": "evolucion_temporal",
            "nombre": "Evolución Temporal por Región",
            "descripcion": "Evolución de la natalidad promedio por región a lo largo de los años.",
            "funcion": viz_evolucion_temporal_regiones,
        },
        {
            "id": "dinamica_natalidad",
            "nombre": "Natalidad vs Variable por País",
            "descripcion": "Relación entre natalidad y variables demográficas, económicas o sociales (por año).",
            "funcion": viz_dinamica_natalidad_vs_variable,
        },
        {
            "id": "dinamica_continente",
            "nombre": "Natalidad vs Variable por Continente",
            "descripcion": "Relación entre natalidad y otra variable, agrupada por continente, con control de año.",
            "funcion": viz_dinamica_natalidad_vs_variable_region,
        },
        {
            "id": "mapa_mundial",
            "nombre": "Mapa Mundial Interactivo",
            "descripcion": "Mapa coroplético con tasas de natalidad por país.",
            "funcion": viz_mapa_mundial_natalidad,
        },
        {
            "id": "evolucion_paises",
            "nombre": "Evolución de Natalidad por País",
            "descripcion": "Serie de tiempo de la natalidad de varios países con highlight interactivo.",
            "funcion": viz_evolucion_paises_highlight,
        },
        {            
            "id": "correlaciones_interactivas",
            "nombre": "Correlaciones Interactivas por Región",
            "descripcion": "Correlaciones interactivas por región a través de los años",
            "funcion": viz_correlaciones_interactivas,

        },
    ]



if __name__ == "__main__":
    print("✅ Módulo de visualizaciones cargado correctamente")
    print(f"📊 Visualizaciones disponibles: {len(get_available_visualizations())}")