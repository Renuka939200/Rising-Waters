from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ==========================
# Load Model
# ==========================

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

os.makedirs("static/graphs", exist_ok=True)


# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# ABOUT
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# CONTACT
# ==========================

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ==========================
# PREDICTION
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        annual_rainfall = float(request.form["annual_rainfall"])
        monsoon_rainfall = float(request.form["monsoon_rainfall"])
        river_level = float(request.form["river_level"])
        cloud_coverage = float(request.form["cloud_coverage"])
        humidity = float(request.form["humidity"])
        temperature = float(request.form["temperature"])

        data = np.array([[
            annual_rainfall,
            monsoon_rainfall,
            river_level,
            cloud_coverage,
            humidity,
            temperature
        ]])

        scaled_data = scaler.transform(data)

        model_name = type(model).__name__

        if model_name == "KNeighborsClassifier":
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
        else:
            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0]

        confidence = round(max(probability) * 100, 2)

        # --------------------------
        # Dynamic Weather Graph
        # --------------------------

        plt.figure(figsize=(10,5))

        features = [
            "Annual\nRainfall",
            "Monsoon\nRainfall",
            "River\nLevel",
            "Cloud\nCoverage",
            "Humidity",
            "Temperature"
        ]

        values = [
            annual_rainfall,
            monsoon_rainfall,
            river_level,
            cloud_coverage,
            humidity,
            temperature
        ]

        colors = [
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#e74c3c",
            "#1abc9c"
        ]

        plt.bar(features, values, color=colors)

        plt.title("Current Weather Parameters")

        plt.ylabel("Values")

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        plt.savefig("static/graphs/current_input.png")

        plt.close()

        # --------------------------
        # Prediction
        # --------------------------

        if prediction == 1:

            result = "⚠️ HIGH FLOOD RISK"

            status = "danger"

        else:

            result = "✅ NO FLOOD RISK"

            status = "safe"

        return render_template(

            "result.html",

            prediction=result,

            confidence=confidence,

            status=status,

            annual=annual_rainfall,

            monsoon=monsoon_rainfall,

            river=river_level,

            cloud=cloud_coverage,

            humidity=humidity,

            temperature=temperature

        )

    except Exception as e:

        return f"<h2>Error</h2><p>{e}</p>"


# ==========================

if __name__ == "__main__":

    app.run(debug=True)