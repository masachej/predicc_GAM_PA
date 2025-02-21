import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import base64
from pygam import LinearGAM

# --- Configurar la pestaña e icono usando el logo ---
logo_path = "logom.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    st.set_page_config(
        page_title="Predicción de Producción | MONTERREY",
        page_icon=f"data:image/png;base64,{encoded_image}",
        layout="centered"
    )
else:
    st.set_page_config(
        page_title="Predicción de Producción | MONTERREY",
        page_icon="🌱",
        layout="centered"
    )

# --- Cargar el modelo GAM entrenado ---
modelo_path = "modelo_GAM.pkl"
if os.path.exists(modelo_path):
    gam = joblib.load(modelo_path)
else:
    st.error("El archivo del modelo GAM no se encuentra. Verifica la ruta.")

# --- Cargar el logo si está disponible ---
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_image}" width="220" style="border-radius: 15px; margin-bottom: 20px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Aplicar estilos CSS personalizados ---
st.markdown("""
    <style>
        body {
            background-color: #f7f9fb;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 38px;
            color: #1a73e8;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            color: #333333;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .stButton>button {
            background-color: #8bc34a;  /* Verde suave */
            color: white;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 12px;
            transition: 0.3s;
            border: none;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #689f38;  /* Verde más oscuro */
            transform: scale(1.05);
        }
        .stNumberInput>div>div>input {
            font-size: 16px;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #1a73e8;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            background-color: #eaf3ff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            color: #1a73e8;
            margin-top: 30px;
            border: 2px solid #1a73e8;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stNumberInput>div>div>input:focus {
            border-color: #0050b3;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Encabezado ---
st.markdown('<div class="title">MONTERREY AZUCARERA LOJANA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicción de Producción de Azúcar con GAM</div>', unsafe_allow_html=True)

st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave:
- **Toneladas Caña Molida (TCM)**
- **Rendimiento (kg/TCM)**
- **Toneladas de Jugo**

La predicción se realiza mediante un **Modelo Aditivo Generalizado (GAM)**, un enfoque de machine learning que permite capturar relaciones no lineales entre las variables de entrada y la producción de azúcar. Este modelo ha sido entrenado con datos históricos del Ingenio Azucarero Monterrey C.A. y proporciona estimaciones basadas en patrones observados en la producción diaria.
""")

# --- Selección del método de entrada de datos ---
opcion = st.radio("¿Cómo deseas ingresar los datos?", 
                   options=["Ingresar datos manualmente", "Subir archivo CSV o XLS"])

if opcion == "Ingresar datos manualmente":
    # --- Entrada de datos manual ---
    col1, col2, col3 = st.columns(3)
    with col1:
        tcm = st.number_input("Ingrese Tcm:", min_value=0.0, format="%.2f")
    with col2:
        rendimiento = st.number_input("Ingrese Rendimiento:", min_value=0.0, format="%.2f")
    with col3:
        toneladas_jugo = st.number_input("Ingrese Toneladas de Jugo:", min_value=0.0, format="%.2f")

    # --- Botón para predecir ---
    if st.button("Predecir Producción"):
        if tcm <= 0.0 or rendimiento <= 0.0 or toneladas_jugo <= 0.0:
            st.warning("⚠️ Por favor, ingrese valores mayores a 0 en todos los campos antes de predecir.")
        else:
            X_nuevo = np.array([[tcm, rendimiento, toneladas_jugo]])
            # Verificar que no haya NaN o Inf en la entrada
            if np.isnan(X_nuevo).any() or np.isinf(X_nuevo).any():
                st.error("Los datos ingresados contienen valores inválidos (NaN o Inf).")
            else:
                y_pred_log = gam.predict(X_nuevo)
                y_pred = np.expm1(y_pred_log)  # Invertir transformación logarítmica
                st.markdown(f'<div class="result-box">⚡ Predicción de Producción: {y_pred[0]:,.2f} sacos</div>', unsafe_allow_html=True)

elif opcion == "Subir archivo CSV o XLS":
    st.write("""
    **Instrucciones para el archivo:**
    - El archivo debe ser de tipo **.csv** o **xls**.
    - Debe contener las siguientes columnas: 
        - **TCM**: Toneladas Caña Molida
        - **Rendimiento**: Rendimiento en kg/TCM
        - **Toneladas de Jugo**: Toneladas de jugo obtenidas
    """)
    uploaded_file = st.file_uploader("Sube tu archivo CSV o XLS", type=["csv", "xls", "xlsx"])
    
    if uploaded_file is not None:
        # --- Leer el archivo ---
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
            
        # --- Verificar si tiene las columnas requeridas ---
        required_columns = ["TCM", "Rendimiento", "Toneladas de Jugo"]
        if not set(required_columns).issubset(data.columns):
            st.error("⚠️ El archivo debe contener las columnas: 'TCM', 'Rendimiento', 'Toneladas de Jugo'")
        else:
            st.write("Archivo cargado correctamente. Realizando predicciones...")
            X_nuevo = data[["TCM", "Rendimiento", "Toneladas de Jugo"]].values
            
            # Verificar que los datos no contengan NaN o Inf
            if np.isnan(X_nuevo).any() or np.isinf(X_nuevo).any():
                st.error("El archivo contiene valores faltantes o inválidos (NaN o Inf). Por favor, verifica tus datos.")
            else:
                y_pred_log = gam.predict(X_nuevo)
                y_pred = np.expm1(y_pred_log)
                data["Predicción de Producción"] = y_pred
                st.write(data.head())
                
                @st.cache
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df(data)
                st.download_button(
                    label="Descargar archivo con predicciones",
                    data=csv,
                    file_name="predicciones_produccion.csv",
                    mime="text/csv"
                )
