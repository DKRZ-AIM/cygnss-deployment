import os
import requests
from flask import Flask, request, render_template, redirect
import pickle
from werkzeug.utils import secure_filename 

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    if request.method == 'POST':
        print(os.getcwd()) 
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        #res = requests.post("http://localhost:8080/predictions/mnist", files={'data': open('test_data/4.png', 'rb')})
        #res = requests.post("http://localhost:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
        res = requests.post("http://torchserve-mar:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
        prediction = res.json()

    return render_template('index.html', prediction_text=f'Predicted Number: {prediction}')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
