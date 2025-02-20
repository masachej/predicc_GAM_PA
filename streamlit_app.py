import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import base64
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM

# Cargar el modelo GAM
modelo_path = "modelo_GAM.pkl"  # Asegúrate de que el archivo está en el directorio correcto
if os.path.exists(modelo_path):
    modelo_gam = joblib.load(modelo_path)
else:
    st.error("El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# Cargar el escalador
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("El archivo del escalador no se encuentra. Verifica la ruta.")

# Función para hacer la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    try:
        # Preparar los datos de entrada
        data = np.array([[tcm, rendimiento, toneladas_jugo]])
        data_scaled = scaler.transform(data)  # Escalar los datos
        prediction_log = modelo_gam.predict(data_scaled)  # Predicción en logaritmo

        # Deshacer la transformación logarítmica
        prediction = np.expm1(prediction_log[0])  # Exp(x) - 1

        return prediction  # Devolver la predicción como un solo valor
    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {e}")
        return None

# Cargar el logo si está disponible
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

# Títulos de la aplicación
st.title("MONTERREY AZUCARERA LOJANA")
st.subheader("Predicción de la Producción de Azúcar")

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
- **Toneladas Caña Molida (TCM)**
- **Rendimiento (kg/TCM)**
- **Toneladas de Jugo**
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
        if result is not None:
            st.success(f"La predicción de producción es: {result:.2f} sacos.")

# --- Si deseas guardar la predicción y las métricas en un archivo CSV ---
# Cargar el nuevo conjunto de datos (si es necesario)
nuevo_df = pd.read_csv('/content/drive/My Drive/Historico_test.csv')  # Reemplaza por la ruta correcta

# Preparar los datos de entrada (X)
X_nuevos = nuevo_df[['Tcm', 'Rendimiento', 'Toneladas_jugo']]  # Ajusta con las columnas reales

# Transformación logarítmica
y_nuevos = nuevo_df['Produccion']
y_nuevos_log = np.log1p(y_nuevos)  # Log(x+1) para evitar valores cero o negativos

# Realizar predicciones con el modelo GAM
y_pred_nuevos_log = modelo_gam.predict(X_nuevos)

# Deshacer la transformación logarítmica en las predicciones
y_pred_nuevos = np.expm1(y_pred_nuevos_log)  # Exp(x) - 1 para obtener las predicciones reales

# Calcular las métricas de evaluación (MSE, RMSE)
mse_global = mean_squared_error(y_nuevos, y_pred_nuevos)
rmse_global = np.sqrt(mse_global)
r2_global = r2_score(y_nuevos, y_pred_nuevos)

# Mostrar las métricas globales
st.write(f"\nMétricas globales:")
st.write(f"Mean Squared Error (MSE): {mse_global:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_global:.2f}")
st.write(f"R² Score: {r2_global:.2f}")

# Agregar las métricas al DataFrame
nuevo_df['Prediccion_Produccion'] = y_pred_nuevos
nuevo_df['MSE'] = (y_nuevos - y_pred_nuevos) ** 2
nuevo_df['RMSE'] = np.sqrt(nuevo_df['MSE'])
nuevo_df['R2'] = (y_pred_nuevos - y_nuevos.mean()) ** 2 / (y_nuevos - y_nuevos.mean()) ** 2

# Guardar el archivo con las métricas y las predicciones
nuevo_df.to_csv('predicciones_y_metricas.csv', index=False)

st.write(f"Las predicciones y métricas se han guardado en el archivo `predicciones_y_metricas.csv`.")
