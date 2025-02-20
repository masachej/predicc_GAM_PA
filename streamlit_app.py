import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Cargar el modelo GAM entrenado ---
modelo_path = "modelo_GAM.pkl"

if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)  # Cargar el modelo si el archivo existe
else:
    st.error("Error: No se encontró el archivo del modelo 'modelo_GAM.pkl'. Asegúrate de subirlo al directorio correcto.")

# --- Título principal ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

# --- Explicación de la herramienta ---
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: 
- **Toneladas Caña Molida (TCM)**
- **Rendimiento**
- **Toneladas de Jugo**.

La predicción se realiza mediante un modelo de Machine Learning basado en Generalized Additive Models (GAM).
""")

# --- Entradas del usuario ---
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Función de predicción ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    data = np.array([[tcm, rendimiento, toneladas_jugo]])  # Convertir a formato adecuado
    prediction_log = gam.predict(data)  # Hacer la predicción en escala logarítmica
    prediction = np.expm1(prediction_log)  # Deshacer la transformación logarítmica
    return prediction[0]  # Devolver la predicción

# --- Botón de predicción ---
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        st.success(f"La predicción de producción es: {result:.2f} sacos.")

# --- Ejecutar en Streamlit ---
# Guarda este código como 'streamlit_app.py'
# Ejecuta con: streamlit run streamlit_app.py
