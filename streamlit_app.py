import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import base64  # Para mostrar la imagen en base64

# --- Rutas de archivos ---
modelo_path = "modelo_GAM.pkl"
scaler_path = "scaler.pkl"
logo_path = "logom.png"  # Asegúrate de que el logo esté en la carpeta

# --- Cargar el modelo GAM ---
if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)
else:
    st.error("El archivo 'modelo_GAM.pkl' no se encuentra en el directorio.")
    st.stop()

# --- Cargar el escalador ---
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("El archivo 'scaler.pkl' no se encuentra en el directorio.")
    st.stop()

# --- Función para hacer predicciones ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        data = np.array([[tcm, rendimiento, toneladas_jugo]])  # Crear array con los valores de entrada
        data_scaled = scaler.transform(data)  # Escalar los datos
        prediction_log = gam.predict(data_scaled)  # Hacer la predicción en logaritmo
        prediction = np.expm1(prediction_log)  # Deshacer la transformación logarítmica
        return prediction[0]
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
        return None

# --- Mostrar el logo ---
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="300"></div>', 
        unsafe_allow_html=True
    )
else:
    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# --- Título de la aplicación ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:  
- **Toneladas Caña Molida (TCM)**  
- **Rendimiento (kg/TCM)**  
- **Toneladas de Jugo**  
""")

# --- Entradas del usuario ---
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Botón de predicción ---
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.success(f"La predicción de producción es: {result:.2f} sacos.")

