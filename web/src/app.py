from flask import Flask, abort, render_template, request
from keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf


app = Flask(__name__)

global model, graph
model = load_model('./src/model.h5')
graph = tf.get_default_graph()


@app.route('/')
def index():
    return render_template('layout.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method != 'POST':
        print('NOT POST REQUEST!')
        return abort(400)

    img = request.files.get('img')
    if img is None:
        return abort(400)

    try:
        pil_img = Image.open(BytesIO(img.stream.read())).convert('L')
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)

        data = np.asarray(pil_img, dtype=float)
        data = data.reshape(1, 28, 28, 1)
        data = data.astype('float32')
        data /= 255

        with graph.as_default():
            classes = model.predict_classes(data, batch_size=128)

    except Exception:
        return abort(400)

    return render_template('number.html', number=classes[0])
