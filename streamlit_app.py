import streamlit as st
import numpy as np
import joblib
import os
import base64
from pygam import LinearGAM

# Cargar el modelo GAM con cacheo adecuado
@st.cache_resource
def load_model():
    modelo_path = "modelo_GAM.pkl"
    if os.path.exists(modelo_path):
        return joblib.load(modelo_path)
    else:
        st.error("No se encontró el archivo del modelo GAM. Verifique la ruta.")
        return None

# Cargar el escalador con cacheo adecuado
@st.cache_data
def load_scaler():
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        st.error("No se encontró el archivo del escalador. Verifique la ruta.")
        return None

modelo_gam = load_model()
scaler = load_scaler()

# Función para realizar la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    if modelo_gam is None or scaler is None:
        return None
    
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    data_scaled = scaler.transform(data)
    prediction = modelo_gam.predict(data_scaled)
    return prediction[0]

# Cargar el logo
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

# Título principal
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar con GAM")

st.write("""
Este aplicativo permite predecir la producción de azúcar utilizando un modelo de Generalized Additive Model (GAM).
""")

st.write("Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.")

# Entrada de datos
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# Botón para hacer la predicción
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        if result is not None:
            st.write(f"La predicción de producción es: {result:.2f} sacos.")

