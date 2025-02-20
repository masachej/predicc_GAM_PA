import numpy as np
import pandas as pd
import streamlit as st
import joblib
import base64
from sklearn.metrics import mean_squared_error, r2_score
import io

# Cargar el modelo GAM
modelo_path = "modelo_GAM.pkl"
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
        prediction = modelo_gam.predict(data_scaled)  # Hacer la predicción
        return prediction[0]  # Devolver la predicción como un solo valor
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

# --- Cargar el nuevo conjunto de datos (debe ser un CSV con las mismas columnas que el entrenamiento) ---
uploaded_file = st.file_uploader("Sube el archivo CSV con los datos de prueba", type=["csv"])
if uploaded_file is not None:
    # Cargar los datos de entrada
    nuevo_df = pd.read_csv(uploaded_file)
    
    # Mostrar las primeras filas para verificar
    st.write("Datos de entrada:", nuevo_df.head())

    # --- Preparar los datos de entrada (X) ---
    X_nuevos = nuevo_df[['Tcm', 'Rendimiento', 'Toneladas_jugo']]  # Ajusta con tus columnas reales de entrada

    # --- Aplicar la transformación logarítmica a la variable objetivo ---
    y_nuevos = nuevo_df['Produccion']
    y_nuevos_log = np.log1p(y_nuevos)  # np.log1p aplica log(x + 1)

    # --- Realizar predicciones con el modelo GAM ---
    y_pred_nuevos_log = modelo_gam.predict(X_nuevos)

    # --- Deshacer la transformación logarítmica en las predicciones ---
    y_pred_nuevos = np.expm1(y_pred_nuevos_log)  # Deshacer la transformación logarítmica (exp(x) - 1)

    # --- Calcular las métricas de evaluación ---
    mse_por_fila = (y_nuevos - y_pred_nuevos) ** 2
    rmse_por_fila = np.sqrt(mse_por_fila)

    # --- Calcular el R² por cada fila ---
    media_y_real = y_nuevos.mean()
    sst_por_fila = (y_nuevos - media_y_real) ** 2  # Total Sum of Squares
    ssr_por_fila = (y_pred_nuevos - media_y_real) ** 2  # Residual Sum of Squares
    r2_por_fila = ssr_por_fila / sst_por_fila

    # --- Agregar las predicciones y las métricas por fila al DataFrame ---
    nuevo_df['Prediccion_Produccion'] = y_pred_nuevos
    nuevo_df['MSE_por_fila'] = mse_por_fila
    nuevo_df['RMSE_por_fila'] = rmse_por_fila
    nuevo_df['R2_por_fila'] = r2_por_fila

    # --- Calcular las métricas globales (MSE, RMSE, R²) ---
    mse_global = mean_squared_error(y_nuevos, y_pred_nuevos)
    rmse_global = np.sqrt(mse_global)
    r2_global = r2_score(y_nuevos, y_pred_nuevos)

    # --- Mostrar las métricas globales ---
    st.write(f"**Métricas globales:**")
    st.write(f"Mean Squared Error (MSE): {mse_global:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_global:.2f}")
    st.write(f"R² Score (Global): {r2_global:.2f}")

    # --- Mostrar la tabla con las métricas de evaluación por fila ---
    st.write("**Tabla con predicciones y métricas por fila:**")
    st.write(nuevo_df[['Tcm', 'Rendimiento', 'Toneladas_jugo', 'Produccion', 'Prediccion_Produccion',
                       'MSE_por_fila', 'RMSE_por_fila', 'R2_por_fila']])

    # --- Descargar el archivo de predicciones y métricas ---
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(nuevo_df)
    st.download_button(
        label="Descargar archivo con predicciones y métricas",
        data=csv,
        file_name='predicciones_y_metricas.csv',
        mime='text/csv'
    )

