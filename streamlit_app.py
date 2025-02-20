import streamlit as st
import joblib
import numpy as np
import os
from pygam import LinearGAM  # Asegurarse de que se importa correctamente

# Ruta del modelo
modelo_path = "modelo_GAM.pkl"

# Verificar si el modelo existe y cargarlo
gam = None
if os.path.exists(modelo_path):
    try:
        gam = joblib.load(modelo_path)
        if not hasattr(gam, 'predict'):
            raise ValueError("El modelo cargado no es un modelo GAM válido.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.error("El archivo del modelo no se encontró. Asegúrate de que 'modelo_GAM.pkl' esté en el directorio correcto.")

# Título de la aplicación
st.title("Predicción de Producción de Azúcar con GAM")
st.subheader("Ingrese los valores para obtener la predicción")

# Entradas de datos con validación básica
tcm = st.number_input("Toneladas Caña Molida (TCM)", min_value=0.0, value=50.0, step=0.1)
rendimiento = st.number_input("Rendimiento (kg/TCM)", min_value=0.0, value=10.0, step=0.1)
toneladas_jugo = st.number_input("Toneladas de Jugo (ton)", min_value=0.0, value=100.0, step=0.1)

# Validar que las entradas sean mayores que cero
if tcm <= 0 or rendimiento <= 0 or toneladas_jugo <= 0:
    st.error("Los valores deben ser mayores que cero.")
else:
    # Botón para realizar la predicción
    if st.button("Realizar Predicción"):
        if gam is None:
            st.error("No se puede realizar la predicción porque el modelo no está cargado.")
        else:
            try:
                # Preparar datos de entrada
                X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])
                
                # Realizar predicción con el modelo GAM
                y_pred_log = gam.predict(X_nuevo)  # Predicción en escala logarítmica
                y_pred = np.expm1(y_pred_log)  # Convertir de logaritmo a escala original
                
                # Mostrar la predicción
                st.success(f"La predicción de producción es: {y_pred[0]:.2f} sacos.")
            except Exception as e:
                st.error(f"Ocurrió un error al hacer la predicción: {e}")
