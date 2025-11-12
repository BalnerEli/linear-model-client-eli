import streamlit as st
import requests
import numpy as np

# URL Render
url = "https://elibalner-linear-model-serving-1.onrender.com/v1/models/linear:predict"

st.set_page_config(page_title="Linear Model Client - Eli", page_icon="📈")

st.title("Linear Model Client - Eli Balderas")
st.write("Fórmula del modelo (entrenamiento):")
st.latex(r"y = 3.14x + 2.71")

# Campo para ingresar los valores
data_input = st.text_input("Valores separados por coma:", "0, 1, 2, 3")
btnPredict = st.button("Predict")

def predict(values):
    # Convertir a arreglo NumPy
    x = np.array(values, dtype=np.float32).reshape(-1, 1)
    data = {"instances": x.tolist()}

    response = requests.post(url, json=data)
    return response

if btnPredict:
    try:
        # Convertir el texto en lista de floats
        values = [float(v.strip()) for v in data_input.split(",") if v.strip()]
        prediction = predict(values)

        # Mostrar respuesta
        st.success("Predicciones obtenidas:")
        st.json(prediction.json())
    except Exception as e:
        st.error(f"Error al predecir: {e}")

st.write("---")
st.caption("API backend en Render: " + url)
