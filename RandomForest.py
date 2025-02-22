# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:36:32 2025

@author: David Araque
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, accuracy_score


# Cargar los datos
df = pd.read_csv('Data_soil_nailing_clean.csv')

# 🔹 Definir variables predictoras (X) y variable objetivo (y)
X = df.select_dtypes(include=["number"]).drop(columns=["id","FS global", "FS BC", "FS SL"], errors="ignore")  # Features numéricas
y = df[["FS global", "FS BC", "FS SL"]]  # Reemplaza "target" con tu variable objetivo

# 🔹 Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Datos preparados:")
print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

# Elegir el modelo (Clasificación o Regresión)
modelo_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))  # Para regresión
# modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Para clasificación

# Entrenar el modelo
modelo_rf.fit(X_train, y_train)

print("\n✅ Modelo entrenado correctamente.")

# Hacer predicciones
y_pred = modelo_rf.predict(X_test)

# Evaluar rendimiento
if y.dtypes[0] == "object":  # ✅ Corregido para múltiples variables objetivo
    accuracy = accuracy_score(y_test.values.ravel(), y_pred.argmax(axis=1))
    print(f"\n🔹 Precisión del modelo (Accuracy): {accuracy:.2f}")
else:
    r2_scores = [r2_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y.columns)]
    for col, r2 in zip(y.columns, r2_scores):
        print(f"\n🔹 R² para {col}: {r2:.2f}")

# Obtener la importancia de cada variable en cada modelo de salida
importancias = np.array([modelo.feature_importances_ for modelo in modelo_rf.estimators_])

# Promediar la importancia entre todas las variables objetivo
importancia_promedio = np.mean(importancias, axis=0)

# Crear DataFrame con importancias
importancia_vars = pd.DataFrame({"Variable": X.columns, "Importance": importancia_promedio})
importancia_vars = importancia_vars.sort_values(by="Importance", ascending=False)

# Definir las columnas a excluir
columnas_excluir = ["id", "Es suficiente", "Installion method", "Soil type"]

# Filtrar el DataFrame de importancia de variables
importancia_filtrada = importancia_vars[~importancia_vars["Variable"].isin(columnas_excluir)]

# Graficar la importancia de variables filtradas
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Variable", data=importancia_filtrada, palette="viridis")
plt.title("Variable Importance in Random Forest Multitarget")
plt.show()

# Guardar el modelo
joblib.dump(modelo_rf, "modelo_entrenado.pkl")
