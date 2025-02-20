import streamlit as st
import joblib
import numpy as np
import os

# --- Cargar el modelo GAM ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)
else:
    st.error("❌ El archivo 'modelo_GAM.pkl' no se encuentra en el directorio.")
    st.stop()

# --- Función para hacer predicciones ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        data = np.array([[tcm, rendimiento, toneladas_jugo]])  # Datos de entrada
        prediction_log = gam.predict(data)  # Predicción en logaritmo
        prediction = np.expm1(prediction_log)  # Convertir de log a valores reales
        return prediction[0]
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
        return None

# --- Interfaz en Streamlit ---
st.title("Predicción de Producción de Azúcar con GAM")
st.subheader("Ingrese los valores para obtener una predicción:")

# --- Entradas del usuario ---
tcm = st.number_input("Toneladas Caña Molida (TCM)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Botón de predicción ---
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("⚠️ Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.success(f"✅ Predicción de producción: {result:.2f} sacos.")
