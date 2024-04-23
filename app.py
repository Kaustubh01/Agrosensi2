import pickle

from flask import Flask, render_template, request

# Create a Flask application
app = Flask(__name__)

with open('models/random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/crop-recommendation', methods=['POST','GET'])
def crop_recommendation():
    if request.method == 'POST':

        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make prediction
        prediction = rf_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction_mapping = {
            1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
            6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
            11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon',
            16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
            21: 'jute', 22: 'coffee'
        }

        print(f'data: {N,P, K, temperature,humidity, ph, rainfall}')
        print(f'prediction: {prediction_mapping[prediction[0]]}')

        return render_template('result.html', prediction= prediction_mapping[prediction[0]])

    return render_template('crop_recommendation.html')


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
