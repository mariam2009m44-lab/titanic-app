# Titanic AI — Survival Predictor

An end-to-end Machine Learning project that predicts passenger survival on the RMS Titanic, with an interactive AI chat interface in Arabic and English.

## Overview

This project goes beyond a simple ML notebook — it's a complete pipeline from raw data to a deployed web application with an intelligent chat interface.

Users can type natural language like:
- "امرأة 25 سنة في الدرجة الأولى"
- "راجل 40 سنة درجة تالتة"
- "woman 25 first class"

And the AI will:
1. Parse the text using NLP
2. Predict survival using a trained Random Forest model
3. Explain the reasoning behind the prediction

## Features

- 7 ML models trained and compared
- Hyperparameter tuning with GridSearchCV
- End-to-end Pipeline to prevent data leakage
- AI Chat Interface (Arabic + English)
- Modern UI with animated ship, waves, and stars
- Explainable AI — shows reasons behind each prediction
- Fully responsive (works on mobile)

## Tech Stack

- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- ML Models: scikit-learn, XGBoost
- Web Framework: Flask
- Frontend: HTML5, CSS3, JavaScript
- Deployment: GitHub Codespaces

## Model Performance

| Model | Accuracy | Precision | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.724 |
| KNN | 0.816 | 0.800 | 0.744 |
| SVM | 0.816 | 0.821 | 0.736 |
| Decision Tree | 0.816 | 0.790 | 0.748 |
| Random Forest (Tuned) | 0.816 | 0.833 | 0.732 |
| Bagging | 0.799 | 0.754 | 0.731 |
| XGBoost | 0.799 | 0.746 | 0.735 |

Best Model: Random Forest with n_estimators=200, max_depth=20, min_samples_split=10

## Project Structure

- app.py - Flask application with AI chat
- ai_chat.py - Terminal-based AI chat
- train_model.py - Script to train and save the model
- requirements.txt - Python dependencies
- titanic_model.pkl - Trained Random Forest model
- templates/index.html - Chat interface
- static/style.css - Styling and animations

## How to Run

1. Clone the repo: git clone https://github.com/mariam2009m44-lab/titanic-app.git
2. Install dependencies: pip install -r requirements.txt
3. Train the model: python train_model.py
4. Run the app: python app.py
5. Open http://localhost:5000

## How It Works

1. User types a natural language description
2. NLP parser extracts features (age, sex, class)
3. Random Forest model predicts survival probability
4. Explanation engine generates a human-readable reason

## Dataset

- Source: Kaggle Titanic Dataset
- Training samples: 891
- Test samples: 418
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## Author

Mariam - https://github.com/mariam2009m44-lab
