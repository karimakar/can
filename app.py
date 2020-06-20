from flask import Flask, render_template, request
import pickle
import numpy as np
# load the xboostclassfier model estimator
filename = 'estimator.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])
        mean_perimeter = float(request.form['mean_perimeter'])
        mean_area = float(request.form['mean_area'])
        mean_smoothness = float(request.form['mean_smoothness'])

        data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness ]])
        my_prediction = classifier.predict(data)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)