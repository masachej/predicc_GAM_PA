import numpy as np
import streamlit as st
import joblib
import os
import base64
from pygam import LinearGAM

# --- Aplicar monkey patch para evitar el error de np.int ---
if not hasattr(np, 'int'):
    np.int = int

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
        prediction_log = modelo_gam.predict(data_scaled)  # Hacer la predicción en escala logarítmica
        
        # --- Deshacer la transformación logarítmica ---
        prediction = np.expm1(prediction_log)  # np.expm1(x) = exp(x) - 1
        
        return prediction[0]  # Devolver la predicción como un solo valor
    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {e}")
        return None

# --- Cargar el logo si está disponible ---
logo_path = "logom.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="300"></div>',
        unsafe_allow_html=True
    )
else:
    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# --- Títulos de la aplicación ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
- **Toneladas Caña Molida (TCM)**
- **Rendimiento (kg/TCM)**
- **Toneladas de Jugo**
""")

st.write("Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.")

# --- Entrada de datos ---
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# --- Botón para hacer la predicción ---
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.success(f"La predicción de producción es: {result:.2f} sacos.")

