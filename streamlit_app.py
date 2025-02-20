import streamlit as st
import tensorflow as tf
import numpy as np
import joblib  # Para cargar el escalador
import os  # Para verificar la existencia del archivo
import base64  # Para codificar la imagen en base64

# Cargar el modelo GAM
from pygam import LinearGAM

# Cargar el modelo
modelo_path = 'modelo_GAM.pkl'
gam = joblib.load(modelo_path) if os.path.exists(modelo_path) else None

# Cargar el escalador (asegúrate de tener el archivo 'scaler.pkl' en el mismo directorio)
scaler = joblib.load('scaler.pkl')

# Función para realizar la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    # Escalar los datos de entrada usando el mismo escalador
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    data_scaled = scaler.transform(data)  # Escalar los datos de entrada
    
    # Aquí es donde se corrige el problema con `np.int` reemplazándolo por `int`
    # Esto es un parche temporal antes de que `pygam` maneje correctamente la deprecación
    try:
        # Realizamos la predicción en escala logarítmica
        prediction_log = gam.predict(data_scaled)  # Hacer la predicción en escala logarítmica
        prediction = np.expm1(prediction_log)  # Convertir de logaritmo a escala original
        return prediction[0]  # Devolver la predicción
    except AttributeError as e:
        if "np.int" in str(e):
            # Parche manual para corregir la deprecación de np.int
            data_scaled = data_scaled.astype(float)  # Asegúrate de que los datos sean de tipo float
            prediction_log = gam.predict(data_scaled)
            prediction = np.expm1(prediction_log)
            return prediction[0]
        else:
            raise e  # Si el error es diferente, lo volvemos a lanzar

# Cargar el logo
logo_path = "logom.png"  # Cambia a la ruta correcta si es necesario
if os.path.exists(logo_path):
    # Codificar la imagen en base64
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Usar HTML para centrar el logo
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

# Texto explicativo sobre la utilidad del aplicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de Jugo.
La herramienta es útil para los profesionales e ingenieros azucareros de la empresa, facilitando la toma de decisiones informadas basadas en datos.
""")

st.write("""
La predicción se realiza mediante un algoritmo de Machine Learning, utilizando un modelo de Red Neuronal Artificial (ANN) entrenada con datos históricos diarios de producción azucarera del Ingenio Azucarero Monterrey C.A.
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
        st.write(f"La predicción de producción es: {result:.2f} sacos.")  # Mostrar la predicción

