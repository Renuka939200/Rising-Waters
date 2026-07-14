# 🌊 Rising Waters

## Intelligent Flood Prediction and Early Warning System

Rising Waters is a Machine Learning-based flood prediction system designed to provide early flood risk assessments using historical weather parameters. The application predicts the likelihood of flooding using multiple classification algorithms and provides an easy-to-use web interface built with Flask.

## Features

- 🌧️ Flood risk prediction
- 🤖 Multiple ML algorithms
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - XGBoost
- 📊 Automatic best model selection
- 📈 Confidence score
- 🌐 Flask web application
- ☁️ IBM Cloud deployment ready
- 📱 Responsive user interface

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- HTML5
- CSS3
- JavaScript

## Project Structure

```
Rising_Waters/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
├── model/
├── static/
└── templates/
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Run the Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## Author

Developed as part of the **Rising Waters** Flood Prediction Project using Machine Learning and Flask.
