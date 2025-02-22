# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:28:03 2025

@author: David Araque
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Cargar el archivo CSV
archivo = "Data_soil_nailing.csv"
df = pd.read_csv(archivo, encoding="ISO-8859-1")

# 🔹 Verificar si el DataFrame tiene datos
if df.empty:
    print("⚠️ El archivo CSV está vacío o no se cargó correctamente.")
else:
    print(f"✅ Se cargaron {df.shape[0]} filas y {df.shape[1]} columnas.")

# 🔹 Información general
print("\n🔹 Primeras 5 filas:")
print(df.head())

print("\n🔹 Información del dataset:")
print(df.info())

print("\n🔹 Valores nulos en cada columna:")
print(df.isnull().sum())

# 🔹 Limpieza de datos
df = df.dropna(how="all")  # Eliminar filas completamente vacías
df = df.dropna(axis=1, how="all")  # Eliminar columnas completamente vacías

# 🔹 Verificar si después de la limpieza aún hay datos
if df.empty:
    print("⚠️ El DataFrame quedó vacío después de eliminar valores nulos. Revisa los datos originales.")
else:
    print(f"✅ Después de la limpieza: {df.shape[0]} filas y {df.shape[1]} columnas.")

# 🔹 Eliminar espacios en blanco y caracteres especiales en strings
df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

# 🔹 Convertir datos a numéricos si es posible
df = df.apply(pd.to_numeric, errors="coerce")

# 🔹 Eliminar duplicados
print("\n🔹 Filas duplicadas antes de limpiar:", df.duplicated().sum())
df = df.drop_duplicates()

# 🔹 Estadísticas de variables numéricas
if not df.select_dtypes(include=["number"]).empty:
    print("\n🔹 Estadísticas de variables numéricas:")
    print(df.describe())
else:
    print("⚠️ No hay columnas numéricas para mostrar estadísticas.")

# 🔹 Histograma de variables numéricas
df_num = df.select_dtypes(include=["number"])
if not df_num.empty:
    df_num.hist(figsize=(20, 15), bins=30)
    plt.suptitle("Distribución de Variables Numéricas", fontsize=16)
    plt.show()
else:
    print("⚠️ No hay columnas numéricas para graficar.")
    
# 🔹 Visualización de variables categóricas
for col in df.select_dtypes(include=["object"]).columns:
    plt.figure(figsize=(10, 5))
    top_values = df[col].value_counts().index[:10]  # 🔹 Filtrar solo las 10 categorías más frecuentes
    sns.countplot(y=df[col], order=top_values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"Top 10 Categorías de {col}")
    plt.show()

# 🔹 Matriz de correlación (solo variables numéricas sin constantes)
df_numerico = df.select_dtypes(include=["number"])
df_numerico = df_numerico.dropna(axis=1, how="all")  # 🔹 Elimina columnas donde TODAS las filas son NaN
if not df_numerico.empty:
    plt.figure(figsize=(15, 12))
    sns.heatmap(df_numerico.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación (Sin Variables Constantes ni NaN)")
    plt.show()
else:
    print("⚠️ No hay suficientes columnas numéricas para calcular la correlación.")

# 🔹 Detección de outliers con IQR
def detectar_outliers_iqr(df):
    outliers = {}
    for col in df.select_dtypes(include=["number"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers_col = df[(df[col] < limite_inferior) | (df[col] > limite_superior)][col]
        if not outliers_col.empty:
            outliers[col] = outliers_col
    return outliers

outliers_detectados = detectar_outliers_iqr(df)

# 🔹 Eliminar outliers creando una copia del DataFrame
df_filtrado = df.copy()
for col in outliers_detectados.keys():
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_filtrado = df_filtrado[(df_filtrado[col] >= limite_inferior) & (df_filtrado[col] <= limite_superior)]

df = df_filtrado

# 🔹 Guardar archivo limpio
if not df.empty:
    df.to_csv("Data_soil_nailing_clean.csv", index=False, encoding="utf-8")
    print("\n✅ Archivo limpio guardado correctamente como 'Data_soil_nailing_clean.csv'.")
else:
    print("⚠️ No se guardó el archivo porque el DataFrame está vacío.")



