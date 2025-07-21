import os
from flask import Flask, request
import joblib

# --- Configura la app ---
app = Flask(__name__)

# Carga modelo y escalador
modelo = joblib.load("modelo_histamina.pkl")
escalador = joblib.load("scaler_histamina.pkl")

# Ruta principal para predicción
@app.route("/predict", methods=["POST"])
def predict():
    datos = request.get_json()
    variables = [datos["acidez"]]  # 👈 solo usamos acidez como entrada
    variables_esc = escalador.transform([variables])
    prediccion = modelo.predict(variables_esc)
    return {"prediccion": float(prediccion[0])}

# Ejecuta la app en el puerto de Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
