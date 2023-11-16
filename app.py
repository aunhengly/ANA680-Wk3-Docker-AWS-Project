from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
filename = 'file_WineQ.pkl'
model = pickle.load(open(filename, 'rb'))

# Initialize an empty list to store predictions
predictions = []

@app.route('/')
def index():
    return render_template('index.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    citric_acid = request.form['citric_acid']
    sulphates = request.form['sulphates']
    alcohol = request.form['alcohol']
    
    pred = model.predict(np.array([[citric_acid, sulphates, alcohol,]]))
    result = f"Citric Acid: {citric_acid}, Sulphates: {sulphates}, Alcohol: {alcohol} => Prediction: {pred[0]}"
    
    # Append the result to the list of predictions
    predictions.append(result)
    
    # Display the results and clear the form
    return render_template('index.html', predict=pred[0], predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)
