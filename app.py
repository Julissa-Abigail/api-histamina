from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  # <- Añade esto si no estaba

app = Flask(__name__)

# Cargar modelo y escalador
modelo = joblib.load("modelo_histamina.pkl")
scaler = joblib.load("scaler_histamina.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    acidez = data.get("acidez")

    # Escalar correctamente usando nombre de columna
    input_df = pd.DataFrame([{ "acidez": acidez }])
    dato_transformado = scaler.transform(input_df)

    # Predicción y probabilidad
    prediccion = modelo.predict(dato_transformado)[0]
    probabilidad = modelo.predict_proba(dato_transformado)[0][1]

    return jsonify({
        "prediccion": int(prediccion),
        "probabilidad": round(probabilidad, 2)
    })
