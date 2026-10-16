from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re

app = Flask(__name__)
ml_model = joblib.load('titanic_model.pkl')

def parse_text(text):
    text_lower = text.lower()
    features = {'Pclass': 3, 'Sex': 'male', 'Age': 30, 'SibSp': 0, 'Parch': 0, 'Fare': 8.0, 'Embarked': 'S'}
    if any(w in text_lower for w in ['امرأة', 'أنثى', 'بنت', 'سيدة', 'female', 'woman', 'girl', 'lady']):
        features['Sex'] = 'female'
        features['Fare'] = 30.0
    if any(w in text_lower for w in ['أولى', 'اولى', 'first', '1st']):
        features['Pclass'] = 1
        features['Fare'] = 80.0
    elif any(w in text_lower for w in ['تانية', 'ثانية', 'second', '2nd']):
        features['Pclass'] = 2
        features['Fare'] = 20.0
    ages = re.findall(r'\d+', text)
    if ages:
        age = int(ages[0])
        if age <= 80:
            features['Age'] = age
    return features

def predict_and_explain(text):
    features = parse_text(text)
    df = pd.DataFrame([features])
    pred = ml_model.predict(df)[0]
    prob = ml_model.predict_proba(df)[0]
    conf = max(prob) * 100
    reasons = []
    if features['Sex'] == 'female':
        reasons.append("النساء نسبة نجاتهم كانت 74%")
    else:
        reasons.append("الرجال نسبة نجاتهم كانت 19%")
    if features['Pclass'] == 1:
        reasons.append("الدرجة الأولى نسبتهم 63%")
    elif features['Pclass'] == 2:
        reasons.append("الدرجة الثانية نسبتهم 47%")
    else:
        reasons.append("الدرجة الثالثة نسبتهم 24%")
    reason_text = " و ".join(reasons)
    if pred == 1:
        header = f"احتمال نجاة الراكب ده {conf:.1f}%"
        status = "survived"
    else:
        header = f"احتمال نجاة الراكب ده ضعيف (ثقة {conf:.1f}% في التوقع)"
        status = "died"
    return {
        'text': f"{header}\nالسبب: {reason_text}",
        'status': status,
        'confidence': round(conf, 1),
        'features': features
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'empty'})
    result = predict_and_explain(user_msg)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
