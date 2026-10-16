from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked']

    data = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex': sex,
        'Embarked': embarked
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]
    confidence = max(probability) * 100

    if prediction == 1:
        result = "SURVIVED"
        message = "This passenger would have survived the tragedy."
        color = "#2ecc71"
        icon = "OK"
    else:
        result = "NOT SURVIVED"
        message = "This passenger would not have survived."
        color = "#e74c3c"
        icon = "X"

    return render_template(
        'index.html',
        result=result,
        message=message,
        confidence=f"{confidence:.1f}",
        color=color,
        icon=icon
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
