import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def cardinalidad(df_in, umbral_categoria, umbral_continua):
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de valores únicos.
        Esta función analiza cada columna del DataFrame de entrada y las clasifica en uno de los siguientes tipos:
        - 'Binaria': si la variable tiene exactamente 2 valores únicos.
        - 'Categorica': si la variable tiene más de 2 pero menos de 'umbral_categoria' valores únicos.
        - 'Numérica continua': si el porcentaje de valores únicos es mayor o igual a 'umbral_continua'.
        - 'Numérica discreta': si no cumple con las condiciones anteriores.
    Args:
        - df_in (pd.DataFrame): DataFrame que contiene las variables a analizar.
        - umbral_categoria (int): Número máximo de valores únicos para que una variable sea considerada categórica.
        - umbral_continua (float): Valor de cardinalidad para que una variable sea considerada numérica continua.
    Return:
        - pd.DataFrame: DataFrame con las columnas originales como índice y las siguientes columnas:
            - 'valores_unicos': Número de valores únicos en la variable.
            - 'cardinalidad(%)': Porcentaje de valores únicos respecto al total de registros.
            - 'tipo': Clasificación de la variable según su cardinalidad.
    """
  
    # Crear un DataFrame con las estadísticas de cardinalidad
    df_variables = pd.DataFrame({
        "valores_unicos": df_in.nunique(),
        "cardinalidad(%)": round(df_in.nunique() / len(df_in) * 100, 2),
    })

    tipos = []
    # Recorrer las variables y clasificar según los valores únicos y la cardinalidad
    for var in df_variables.index:
        valores_unicos = df_variables.loc[var, "valores_unicos"]
        cardinalidad = df_variables.loc[var, "cardinalidad(%)"]
        if valores_unicos == 2:
            tipos.append("Binaria")
        elif valores_unicos < umbral_categoria:
            tipos.append("Categorica")
        elif cardinalidad >= umbral_continua:
            tipos.append("Numérica continua")
        else:
            tipos.append("Numérica discreta")

    df_variables["tipo"] = tipos
    return df_variables


def plot_frecuencias(df, variables_categoricas, palette='viridis'):
    """
    Genera gráficos de frecuencias absolutas y relativas para múltiples variables categóricas.

    Args:
        - df: DataFrame de pandas que contiene los datos.
        - var_cat: Diccionario donde las claves son los nombres de las columnas categóricas a analizar y los valores son listas de etiquetas descriptivas correspondientes a los valores únicos de cada columna.
        - palette: Paleta de colores para los gráficos (por defecto 'viridis').
    Return:
    - None. Muestra los gráficos generados.
    """
    num_vars = len(variables_categoricas)
    fig, axs = plt.subplots(nrows=num_vars, ncols=2, figsize=(15, 10 * num_vars))
    fig.suptitle("Frecuencias Absolutas y Relativas", fontsize=16)
    
    if num_vars == 1:
        axs = [axs]

    for i, columna in enumerate(variables_categoricas):
        
        # Calcular frecuencias absolutas y relativas
        frecuencias_absolutas = df[columna].value_counts()
        frecuencias_relativas = df[columna].value_counts(normalize=True) * 100

        # Determinar el desplazamiento proporcional para las etiquetas
        desp_absoluto = max(frecuencias_absolutas) * 0.01  # 1% del valor máximo
        desp_relativo = max(frecuencias_relativas) * 0.01  # 1% del valor máximo

        # Gráfico de Frecuencias Absolutas
        sns.barplot(
            x=frecuencias_absolutas.index,
            y=frecuencias_absolutas.values,
            hue=frecuencias_absolutas.index,
            # data=df,
            palette=palette,
            ax=axs[i][0],
            legend=False
        )
        axs[i][0].set_title(f'Frecuencia Absoluta de {columna}', fontsize=14)
        axs[i][0].set_xlabel('Categoría', fontsize=12)
        axs[i][0].set_ylabel('Conteo', fontsize=12)
        for j, value in enumerate(frecuencias_absolutas.values):
            axs[i][0].text(j, value + desp_absoluto, f'{value}', ha='center', fontsize=12)

        # Gráfico de Frecuencias Relativas
        sns.barplot(
            x=frecuencias_relativas.index,
            y=frecuencias_relativas.values,
            hue=frecuencias_absolutas.index,
            # data=df,
            palette=palette,
            ax=axs[i][1],
            legend=False
        )
        axs[i][1].set_title(f'Frecuencia Relativa (%) de {columna}', fontsize=14)
        axs[i][1].set_xlabel('Categoría', fontsize=12)
        axs[i][1].set_ylabel('Porcentaje (%)', fontsize=12)
        for j, value in enumerate(frecuencias_relativas.values):
            axs[i][1].text(j, value + desp_relativo, f'{value:.2f}%', ha='center', fontsize=12)

    # Ajustar el diseño para evitar solapamientos
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, variables_numericas, variable_categorica=None,whis=3, palette = "viridis"):
    """
    Genera gráficos de caja (boxplots) para variables numéricas.
    Si se proporciona una variable categórica, se agrupan los datos por esta variable.
    Los gráficos se organizan en una cuadrícula de dos columnas.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        variables_numericas (list): Lista de nombres de las columnas numéricas a analizar.
        variable_categorica (str, opcional): Nombre de la columna categórica para agrupar las variables numéricas. Por defecto es None.

    Returns:
        None. Muestra los gráficos generados.
    """
    num_vars = len(variables_numericas)
    num_cols = 2
    num_rows = math.ceil(num_vars / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle("Gráficos de Caja para Variables Numéricas", fontsize=16)
    
    axs = axs.flatten()

    for i, variable in enumerate(variables_numericas):
        if variable_categorica:
            sns.boxplot(x=variable_categorica, 
                        y=variable, 
                        hue = variable_categorica, 
                        data=df, 
                        ax=axs[i],
                        whis=whis,
                        palette=palette)
            axs[i].set_title(f'Boxplot de {variable} por {variable_categorica}', fontsize=14)
            axs[i].set_xlabel(variable_categorica, fontsize=12)
            axs[i].set_ylabel(variable, fontsize=12)
        else:
            sns.boxplot(x=df[variable], 
                        ax=axs[i],
                        whis=whis)
            axs[i].set_title(f'Boxplot de {variable}', fontsize=14)
            axs[i].set_xlabel(variable, fontsize=12)
            axs[i].set_ylabel('Valores', fontsize=12)

    # Eliminar ejes vacíos si el número de variables no es múltiplo de 2
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Ajustar el diseño para evitar solapamientos
    plt.tight_layout()
    plt.show()

def variabilidad(df):
    """
    Calcula la desviación estándar, la media y el coeficiente de variación (CV) para cada columna numérica de un DataFrame.
    El coeficiente de variación se expresa como un porcentaje y se calcula como:
        CV = (desviación estándar / media) * 100
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas numéricas a analizar.
    Returns:
        pd.DataFrame: DataFrame con las columnas 'std' (desviación estándar), 'mean' (media) y 'CV' (coeficiente de variación) para cada columna numérica del DataFrame original.
    """
    df_var = df.describe().loc[["std", "mean"]].T  # Crear un nuevo DataFrame con medidas de desviación estándar y media
    df_var["CV"] = (df_var["std"] / df_var["mean"]) * 100  # Crear una columna con el coeficiente de variación
    return df_var


def plot_hist(df, columnas, bins, kde, variable_categorica=None):
    """
    Genera histogramas para las columnas especificadas de un DataFrame.
    Si se proporciona una variable categórica, los datos se diferencian por categoría en un mismo histograma.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columnas (list): Lista de nombres de las columnas numéricas a graficar.
        bins (list): Lista con el número de bins para cada histograma.
        kde (list): Lista de valores booleanos que indican si se debe incluir la estimación de densidad (KDE) para cada histograma.
        variable_categorica (str, opcional): Nombre de la columna categórica para diferenciar los datos en los histogramas.

    Returns:
        None. Muestra los histogramas generados.
    """
    # Cerrar cualquier figura abierta previamente para evitar duplicados
    plt.close('all')  

    # Validar que 'bins' y 'kde' tienen la misma longitud que 'columnas'
    if len(bins) != len(columnas) or len(kde) != len(columnas):
        raise ValueError("Los parámetros 'bins' y 'kde' deben tener la misma longitud que 'columnas'.")

    # Definir el número de columnas para la disposición de los subplots
    ncols = 2
    # Calcular el número de filas necesarias
    nrows = math.ceil(len(columnas) / ncols)

    # Crear la figura y los subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5 * nrows))
    # Asegurarse de que 'axs' sea un arreglo unidimensional para facilitar la iteración
    axs = axs.flatten()

    # Iterar sobre las columnas para crear cada histograma
    for i, col in enumerate(columnas):
        # Crear el histograma para la columna actual
        sns.histplot(
            data=df,
            x=col,
            hue=variable_categorica if variable_categorica else None,
            kde=kde[i],
            bins=bins[i],
            ax=axs[i]
        )
        # Establecer el título del subplot
        axs[i].set_title(f"Histograma de {col}" + (f" por {variable_categorica}" if variable_categorica else ""))

    # Desactivar los subplots vacíos (si los hay)
    for j in range(len(columnas), len(axs)):
        axs[j].axis("off")

    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout()
    # Mostrar los histogramas
    plt.show()

def app_mannwhitney(df, var_num, var_cat, categoria1, categoria2):
    resultados = {}

    for var in var_num:
        # Definir los grupos basados en la variable categórica
        grupo1 = df[df[var_cat] == categoria1][var]
        grupo2 = df[df[var_cat] == categoria2][var]

        # Calcular el estadístico U y el valor p
        u_stat, p_valor = mannwhitneyu(grupo1, grupo2, alternative='two-sided')

        # Calcular los tamaños de las muestras
        n1 = len(grupo1)
        n2 = len(grupo2)

        # Calcular la media y desviación estándar de U
        mu_u = n1 * n2 / 2
        sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        # Calcular el estadístico Z
        z = (u_stat - mu_u) / sigma_u

        # Calcular el tamaño del efecto r
        r = z / np.sqrt(n1 + n2)

        # Almacenar los resultados
        resultados[var] = {'U_stat': u_stat, 'p_valor': p_valor, 'r': r}

    return resultados