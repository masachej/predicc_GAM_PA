import os
import numpy as np
import streamlit as st
import joblib
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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

# --- Función para realizar la predicción ---
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        # Preparar los datos de entrada
        data = np.array([[tcm, rendimiento, toneladas_jugo]])
        data_scaled = scaler.transform(data)  # Normalizar con el escalador
        
        # Predecir en escala logarítmica
        prediction_log = modelo_gam.predict(data_scaled)
        
        # Deshacer la transformación logarítmica
        prediction = np.expm1(prediction_log[0])  # exp(x) - 1

        return prediction  # Devolver la predicción
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

# --- Título de la Aplicación ---
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
""")

# --- Entradas del usuario ---
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.1, value=10.0, step=0.1)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.1, value=100.0, step=0.1)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.1, value=50.0, step=0.1)

# --- Realizar la predicción ---
if st.button("Realizar Predicción"):
    result = make_prediction(tcm, rendimiento, toneladas_jugo)
    
    if result is not None:
        st.success(f"La predicción de producción es: **{result:.2f} sacos**.")

        # Calcular métricas de evaluación si el usuario ingresa un valor real
        produccion_real = st.number_input("Si conoce el valor real de producción, ingréselo para evaluar el modelo (opcional)", min_value=0.0, step=1.0)
        
        if produccion_real > 0:
            mse = mean_squared_error([produccion_real], [result])
            rmse = np.sqrt(mse)
            r2 = r2_score([produccion_real], [result])

            st.write(f"### 📊 Métricas de Evaluación")
            st.write(f"- **Error Cuadrático Medio (MSE):** {mse:.2f}")
            st.write(f"- **Raíz del Error Cuadrático Medio (RMSE):** {rmse:.2f}")
            st.write(f"- **Coeficiente de Determinación (R²):** {r2:.2f}")
