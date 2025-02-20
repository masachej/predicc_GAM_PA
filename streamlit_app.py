import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Para cargar el modelo y el escalador
import os  # Para verificar la existencia del archivo
import base64  # Para codificar la imagen en base64

# Cargar el modelo GAM
modelo_path = 'modelo_GAM.pkl'
gam = joblib.load(modelo_path)

# Función para realizar la predicción
def hacer_prediccion(tcm, rendimiento, toneladas_jugo):
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    prediccion_log = gam.predict(data)  # Predicción en el espacio logarítmico
    prediccion = np.expm1(prediccion_log)  # Transformar de nuevo al espacio original
    return prediccion[0]

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

# Título secundario
st.subheader("Predicción de la Producción de Azúcar")

# Texto explicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: 
- Toneladas Caña Molida (TCM)
- Rendimiento
- Toneladas de Jugo

La predicción se realiza mediante un modelo GAM entrenado con datos históricos de producción del Ingenio Azucarero Monterrey C.A.
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
        resultado = hacer_prediccion(tcm, rendimiento, toneladas_jugo)
        st.write(f"La predicción de producción es: {resultado:.2f} sacos.")
