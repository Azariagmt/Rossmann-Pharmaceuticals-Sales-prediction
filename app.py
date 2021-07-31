from flask import Flask, request, render_template
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized
from werkzeug.utils import secure_filename
import os
# TODO: refactor into own preprocess component
import pandas as pd
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(NotFound)
def page_not_found_handler(e: HTTPException):
    return '<h1>404.html</h1>', 404


@app.errorhandler(Unauthorized)
def unauthorized_handler(e: HTTPException):
    return '<h1>401.html</h1>', 401


@app.errorhandler(Forbidden)
def forbidden_handler(e: HTTPException):
    return '<h1>403.html</h1>', 403


@app.errorhandler(RequestTimeout)
def request_timeout_handler(e: HTTPException):
    return '<h1>408.html</h1>', 408


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/', secure_filename(f.filename))
        f.save(file_path)
        print(f)
        df = pd.read_csv(file_path)
        print(df.head())

        # TODO: feed into sklearn pipeline
        # TODO: make prediction
        # TODO: send prediction output back

        # Make prediction
        # shutil.rmtree('./uploads/zz')
        # os.mkdir('./uploads/zz')

    # TODO: should return prediction data points
    data = {
        "x": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
        "y": [23, 44, 55, 77, 55, 62]
    }
    return data


@app.route('/analysis')
def predict():
    return render_template('file-upload.html')

# TODO: refactor analysis route to prediction
# TODO: add analysis route.... analysis for existing data using powerbi

if __name__ == '__main__':
    os.environ.setdefault('Flask_SETTINGS_MODULE', 'helloworld.settings')
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    port = int(os.environ.get("PORT", 33507))
    app.run(debug=True)
