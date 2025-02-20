import streamlit as st
import numpy as np
import joblib  # Para cargar el modelo y el escalador
import os  # Para verificar la existencia del archivo
import base64  # Para codificar la imagen en base64
from pygam import LinearGAM  # Importar la clase de modelo GAM

# Cargar el modelo GAM
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    modelo_gam = joblib.load(modelo_path)
else:
    st.error("No se encontró el archivo del modelo GAM. Verifique la ruta.")

# Cargar el escalador
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("No se encontró el archivo del escalador. Verifique la ruta.")

# Función para realizar la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    # Convertir los datos a un array numpy
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    # Escalar los datos de entrada
    data_scaled = scaler.transform(data)
    # Hacer la predicción con el modelo GAM
    prediction = modelo_gam.predict(data_scaled)
    return prediction[0]  # Devolver la predicción

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

# Explicación del aplicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar utilizando un modelo de Generalized Additive Model (GAM).
""")

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
""")

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
        st.write(f"La predicción de producción es: {result:.2f} sacos.")
