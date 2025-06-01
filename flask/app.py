import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

# Use all features
dataset_X = dataset.iloc[:, :-1].values  # All columns except the last one (Outcome)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get all 8 features in the correct order
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    float_features = [float(request.form[feature]) for feature in features]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
