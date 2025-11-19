"""
Funciones auxiliares para agregar información geográfica (regiones, continentes)
Basado en la función asignar_regiones() del notebook
"""

import pandas as pd

# Diccionario completo de países por continente y región
REGIONES_DICT = {
    # AMÉRICA
    'United States': {'Continente': 'América', 'Region': 'América del Norte'},
    'Canada': {'Continente': 'América', 'Region': 'América del Norte'},
    'Mexico': {'Continente': 'América', 'Region': 'América del Norte'},
    'Bermuda': {'Continente': 'América', 'Region': 'América del Norte'},
    'Greenland': {'Continente': 'América', 'Region': 'América del Norte'},

    'Belize': {'Continente': 'América', 'Region': 'América Central'},
    'Costa Rica': {'Continente': 'América', 'Region': 'América Central'},
    'El Salvador': {'Continente': 'América', 'Region': 'América Central'},
    'Guatemala': {'Continente': 'América', 'Region': 'América Central'},
    'Honduras': {'Continente': 'América', 'Region': 'América Central'},
    'Nicaragua': {'Continente': 'América', 'Region': 'América Central'},
    'Panama': {'Continente': 'América', 'Region': 'América Central'},

    'Bahamas, The': {'Continente': 'América', 'Region': 'Caribe'},
    'Barbados': {'Continente': 'América', 'Region': 'Caribe'},
    'Cuba': {'Continente': 'América', 'Region': 'Caribe'},
    'Dominican Republic': {'Continente': 'América', 'Region': 'Caribe'},
    'Haiti': {'Continente': 'América', 'Region': 'Caribe'},
    'Jamaica': {'Continente': 'América', 'Region': 'Caribe'},
    'Trinidad and Tobago': {'Continente': 'América', 'Region': 'Caribe'},
    'Antigua and Barbuda': {'Continente': 'América', 'Region': 'Caribe'},
    'Aruba': {'Continente': 'América', 'Region': 'Caribe'},
    'British Virgin Islands': {'Continente': 'América', 'Region': 'Caribe'},
    'Cayman Islands': {'Continente': 'América', 'Region': 'Caribe'},
    'Curacao': {'Continente': 'América', 'Region': 'Caribe'},
    'Dominica': {'Continente': 'América', 'Region': 'Caribe'},
    'Grenada': {'Continente': 'América', 'Region': 'Caribe'},
    'Puerto Rico (US)': {'Continente': 'América', 'Region': 'Caribe'},
    'Sint Maarten (Dutch part)': {'Continente': 'América', 'Region': 'Caribe'},
    'St. Kitts and Nevis': {'Continente': 'América', 'Region': 'Caribe'},
    'St. Lucia': {'Continente': 'América', 'Region': 'Caribe'},
    'St. Martin (French part)': {'Continente': 'América', 'Region': 'Caribe'},
    'St. Vincent and the Grenadines': {'Continente': 'América', 'Region': 'Caribe'},
    'Turks and Caicos Islands': {'Continente': 'América', 'Region': 'Caribe'},
    'Virgin Islands (U.S.)': {'Continente': 'América', 'Region': 'Caribe'},

    'Argentina': {'Continente': 'América', 'Region': 'América del Sur'},
    'Bolivia': {'Continente': 'América', 'Region': 'América del Sur'},
    'Brazil': {'Continente': 'América', 'Region': 'América del Sur'},
    'Chile': {'Continente': 'América', 'Region': 'América del Sur'},
    'Colombia': {'Continente': 'América', 'Region': 'América del Sur'},
    'Ecuador': {'Continente': 'América', 'Region': 'América del Sur'},
    'Guyana': {'Continente': 'América', 'Region': 'América del Sur'},
    'Paraguay': {'Continente': 'América', 'Region': 'América del Sur'},
    'Peru': {'Continente': 'América', 'Region': 'América del Sur'},
    'Suriname': {'Continente': 'América', 'Region': 'América del Sur'},
    'Uruguay': {'Continente': 'América', 'Region': 'América del Sur'},
    'Venezuela, RB': {'Continente': 'América', 'Region': 'América del Sur'},

    # EUROPA
    'Albania': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Andorra': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Bosnia and Herzegovina': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Croatia': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Greece': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Italy': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Malta': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Montenegro': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'North Macedonia': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Portugal': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Serbia': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Slovenia': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Spain': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Gibraltar': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'Kosovo': {'Continente': 'Europa', 'Region': 'Europa del Sur'},
    'San Marino': {'Continente': 'Europa', 'Region': 'Europa del Sur'},

    'Austria': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Belgium': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'France': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Germany': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Liechtenstein': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Luxembourg': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Monaco': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Netherlands': {'Continente': 'Europa', 'Region': 'Europa Occidental'},
    'Switzerland': {'Continente': 'Europa', 'Region': 'Europa Occidental'},

    'Denmark': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Estonia': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Finland': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Iceland': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Ireland': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Latvia': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Lithuania': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Norway': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Sweden': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'United Kingdom': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Channel Islands': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Faroe Islands': {'Continente': 'Europa', 'Region': 'Europa del Norte'},
    'Isle of Man': {'Continente': 'Europa', 'Region': 'Europa del Norte'},

    'Belarus': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Bulgaria': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Czechia': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Hungary': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Moldova': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Poland': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Romania': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Russian Federation': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Slovak Republic': {'Continente': 'Europa', 'Region': 'Europa del Este'},
    'Ukraine': {'Continente': 'Europa', 'Region': 'Europa del Este'},

    # ASIA
    'Afghanistan': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Bangladesh': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Bhutan': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'India': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Iran, Islamic Rep.': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Maldives': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Nepal': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Pakistan': {'Continente': 'Asia', 'Region': 'Asia del Sur'},
    'Sri Lanka': {'Continente': 'Asia', 'Region': 'Asia del Sur'},

    'China': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Hong Kong SAR, China': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Japan': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Korea, Rep.': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Mongolia': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    "Korea, Dem. People's Rep.": {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Macao SAR, China': {'Continente': 'Asia', 'Region': 'Asia Oriental'},
    'Taiwan': {'Continente': 'Asia', 'Region': 'Asia Oriental'},

    'Brunei Darussalam': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Cambodia': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Indonesia': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Lao PDR': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Malaysia': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Myanmar': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Philippines': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Singapore': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Thailand': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Timor-Leste': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},
    'Viet Nam': {'Continente': 'Asia', 'Region': 'Sudeste Asiático'},

    'Armenia': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Azerbaijan': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Bahrain': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Cyprus': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Georgia': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Iraq': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Israel': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Jordan': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Kuwait': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Lebanon': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Oman': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Qatar': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Saudi Arabia': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Syrian Arab Republic': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Turkiye': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'United Arab Emirates': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'Yemen, Rep.': {'Continente': 'Asia', 'Region': 'Asia Occidental'},
    'West Bank and Gaza': {'Continente': 'Asia', 'Region': 'Asia Occidental'},

    'Kazakhstan': {'Continente': 'Asia', 'Region': 'Asia Central'},
    'Kyrgyz Republic': {'Continente': 'Asia', 'Region': 'Asia Central'},
    'Tajikistan': {'Continente': 'Asia', 'Region': 'Asia Central'},
    'Turkmenistan': {'Continente': 'Asia', 'Region': 'Asia Central'},
    'Uzbekistan': {'Continente': 'Asia', 'Region': 'Asia Central'},

    # ÁFRICA
    'Algeria': {'Continente': 'África', 'Region': 'África del Norte'},
    'Egypt, Arab Rep.': {'Continente': 'África', 'Region': 'África del Norte'},
    'Libya': {'Continente': 'África', 'Region': 'África del Norte'},
    'Morocco': {'Continente': 'África', 'Region': 'África del Norte'},
    'Sudan': {'Continente': 'África', 'Region': 'África del Norte'},
    'Tunisia': {'Continente': 'África', 'Region': 'África del Norte'},

    'Benin': {'Continente': 'África', 'Region': 'África Occidental'},
    'Burkina Faso': {'Continente': 'África', 'Region': 'África Occidental'},
    'Cabo Verde': {'Continente': 'África', 'Region': 'África Occidental'},
    'Cameroon': {'Continente': 'África', 'Region': 'África Occidental'},
    'Chad': {'Continente': 'África', 'Region': 'África Occidental'},
    'Gambia, The': {'Continente': 'África', 'Region': 'África Occidental'},
    'Ghana': {'Continente': 'África', 'Region': 'África Occidental'},
    'Guinea': {'Continente': 'África', 'Region': 'África Occidental'},
    'Guinea-Bissau': {'Continente': 'África', 'Region': 'África Occidental'},
    "Cote d'Ivoire": {'Continente': 'África', 'Region': 'África Occidental'},
    'Liberia': {'Continente': 'África', 'Region': 'África Occidental'},
    'Mali': {'Continente': 'África', 'Region': 'África Occidental'},
    'Mauritania': {'Continente': 'África', 'Region': 'África Occidental'},
    'Niger': {'Continente': 'África', 'Region': 'África Occidental'},
    'Nigeria': {'Continente': 'África', 'Region': 'África Occidental'},
    'Senegal': {'Continente': 'África', 'Region': 'África Occidental'},
    'Sierra Leone': {'Continente': 'África', 'Region': 'África Occidental'},
    'Togo': {'Continente': 'África', 'Region': 'África Occidental'},

    'Angola': {'Continente': 'África', 'Region': 'África del Sur'},
    'Botswana': {'Continente': 'África', 'Region': 'África del Sur'},
    'Comoros': {'Continente': 'África', 'Region': 'África del Sur'},
    'Eswatini': {'Continente': 'África', 'Region': 'África del Sur'},
    'Lesotho': {'Continente': 'África', 'Region': 'África del Sur'},
    'Madagascar': {'Continente': 'África', 'Region': 'África del Sur'},
    'Malawi': {'Continente': 'África', 'Region': 'África del Sur'},
    'Mauritius': {'Continente': 'África', 'Region': 'África del Sur'},
    'Mozambique': {'Continente': 'África', 'Region': 'África del Sur'},
    'Namibia': {'Continente': 'África', 'Region': 'África del Sur'},
    'Seychelles': {'Continente': 'África', 'Region': 'África del Sur'},
    'South Africa': {'Continente': 'África', 'Region': 'África del Sur'},
    'Zambia': {'Continente': 'África', 'Region': 'África del Sur'},
    'Zimbabwe': {'Continente': 'África', 'Region': 'África del Sur'},

    'Burundi': {'Continente': 'África', 'Region': 'África Oriental'},
    'Djibouti': {'Continente': 'África', 'Region': 'África Oriental'},
    'Eritrea': {'Continente': 'África', 'Region': 'África Oriental'},
    'Ethiopia': {'Continente': 'África', 'Region': 'África Oriental'},
    'Kenya': {'Continente': 'África', 'Region': 'África Oriental'},
    'Rwanda': {'Continente': 'África', 'Region': 'África Oriental'},
    'Somalia': {'Continente': 'África', 'Region': 'África Oriental'},
    'South Sudan': {'Continente': 'África', 'Region': 'África Oriental'},
    'Tanzania': {'Continente': 'África', 'Region': 'África Oriental'},
    'Uganda': {'Continente': 'África', 'Region': 'África Oriental'},

    'Central African Republic': {'Continente': 'África', 'Region': 'África Central'},
    'Congo, Rep.': {'Continente': 'África', 'Region': 'África Central'},
    'Congo, Dem. Rep.': {'Continente': 'África', 'Region': 'África Central'},
    'Equatorial Guinea': {'Continente': 'África', 'Region': 'África Central'},
    'Gabon': {'Continente': 'África', 'Region': 'África Central'},
    'Sao Tome and Principe': {'Continente': 'África', 'Region': 'África Central'},

    # OCEANÍA
    'Australia': {'Continente': 'Oceanía', 'Region': 'Australia y Nueva Zelanda'},
    'New Zealand': {'Continente': 'Oceanía', 'Region': 'Australia y Nueva Zelanda'},

    'Fiji': {'Continente': 'Oceanía', 'Region': 'Melanesia'},
    'Papua New Guinea': {'Continente': 'Oceanía', 'Region': 'Melanesia'},
    'Solomon Islands': {'Continente': 'Oceanía', 'Region': 'Melanesia'},
    'Vanuatu': {'Continente': 'Oceanía', 'Region': 'Melanesia'},
    'New Caledonia': {'Continente': 'Oceanía', 'Region': 'Melanesia'},

    'Kiribati': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Marshall Islands': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Micronesia, Fed. Sts.': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Nauru': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Palau': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Guam': {'Continente': 'Oceanía', 'Region': 'Micronesia'},
    'Northern Mariana Islands': {'Continente': 'Oceanía', 'Region': 'Micronesia'},

    'Samoa': {'Continente': 'Oceanía', 'Region': 'Polinesia'},
    'Tonga': {'Continente': 'Oceanía', 'Region': 'Polinesia'},
    'Tuvalu': {'Continente': 'Oceanía', 'Region': 'Polinesia'},
    'American Samoa': {'Continente': 'Oceanía', 'Region': 'Polinesia'},
    'French Polynesia': {'Continente': 'Oceanía', 'Region': 'Polinesia'},
}


def add_geographic_info(df, country_column='Pais'):
    """
    Agrega columnas de Región y Continente al dataframe
    
    Args:
        df (pd.DataFrame): Dataset original
        country_column (str): Nombre de la columna de países
        
    Returns:
        pd.DataFrame: Dataset con columnas Continente y Region agregadas
    """
    df_with_geo = df.copy()
    
    if country_column not in df.columns:
        print(f" Columna '{country_column}' no encontrada")
        return df_with_geo
    
    # Asignar continente y región
    df_with_geo['Continente'] = df_with_geo[country_column].map(
        lambda x: REGIONES_DICT.get(x, {}).get('Continente', 'Sin clasificar')
    )
    
    df_with_geo['Region'] = df_with_geo[country_column].map(
        lambda x: REGIONES_DICT.get(x, {}).get('Region', 'Sin clasificar')
    )
    
    return df_with_geo


def get_region_mapping():
    """
    Retorna el diccionario completo de mapeo de países
    
    Returns:
        dict: Mapeo de países a región y continente
    """
    return REGIONES_DICT


def get_countries_by_region(region):
    """
    Retorna lista de países de una región específica
    
    Args:
        region (str): Nombre de la región
        
    Returns:
        list: Lista de países
    """
    return [
        country for country, info in REGIONES_DICT.items()
        if info.get('Region') == region
    ]


def get_countries_by_continent(continent):
    """
    Retorna lista de países de un continente específico
    
    Args:
        continent (str): Nombre del continente
        
    Returns:
        list: Lista de países
    """
    return [
        country for country, info in REGIONES_DICT.items()
        if info.get('Continente') == continent
    ]


def get_available_regions():
    """Retorna lista única de regiones"""
    return sorted(set(info['Region'] for info in REGIONES_DICT.values()))


def get_available_continents():
    """Retorna lista única de continentes"""
    return sorted(set(info['Continente'] for info in REGIONES_DICT.values()))


if __name__ == "__main__":
    print(" Módulo de geografía cargado")
    print(f" Países mapeados: {len(REGIONES_DICT)}")
    print(f"\n Continentes: {get_available_continents()}")
    print(f"\n Total de regiones: {len(get_available_regions())}")