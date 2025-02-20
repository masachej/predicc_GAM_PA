import numpy as np
import streamlit as st
import joblib
import os
from pygam import LinearGAM

# --- Cargar el modelo GAM ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    modelo_gam = joblib.load(modelo_path)
else:
    st.error("El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# --- Cargar el escalador ---
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("El archivo del escalador no se encuentra. Verifica la ruta.")

# --- Función para hacer la predicción ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        # Preparar los datos de entrada
        data = np.array([[tcm, rendimiento, toneladas_jugo]])
        data_scaled = scaler.transform(data)  # Escalar los datos
        prediction_log = modelo_gam.predict(data_scaled)  # Predicción en escala log
        prediction = np.expm1(prediction_log)  # Revertir transformación logarítmica
        return prediction[0]
    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {e}")
        return None

# --- Configuración de la interfaz en Streamlit ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

st.write("Ingrese los valores para obtener una predicción de la producción de azúcar en sacos.")

# --- Entrada de datos ---
tcm = st.number_input("Ingrese Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Botón para hacer la predicción ---
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.success(f"La predicción de producción es: {result:.2f} sacos.")
