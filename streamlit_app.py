import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import base64
import os

# Función para cargar el modelo y el escalador
@st.cache(allow_output_mutation=True)  # Se usa para almacenar el modelo cargado en caché
def load_model():
    # Cargar el modelo ANN (ejemplo con TensorFlow)
    model = tf.keras.models.load_model('ANN_modelo_PPA.h5')
    
    # Cargar el escalador
    scaler = joblib.load('scaler.pkl')
    
    return model, scaler

# Cargar el modelo y el escalador al inicio
model, scaler = load_model()

# Función para hacer la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    # Preparar los datos de entrada
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    
    # Escalar los datos
    data_scaled = scaler.transform(data)
    
    # Realizar la predicción
    prediction = model.predict(data_scaled)
    
    return prediction[0][0]

# Cargar el logo
logo_path = "logom.png"  # Ruta al logo (si tienes)
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="300"></div>', unsafe_allow_html=True)
else:
    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# Título de la aplicación
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

# Explicación de la herramienta
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de Jugo.
La herramienta es útil para los profesionales e ingenieros azucareros de la empresa, facilitando la toma de decisiones informadas basadas en datos.
""")

# Entradas de datos
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

