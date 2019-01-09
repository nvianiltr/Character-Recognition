from flask import Flask, request, render_template, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
from keras import backend as K
from keras.models import model_from_yaml
import cv2
import re
import base64
import pickle

app = Flask(__name__)

def load_model():
    yaml_file = open('bin/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    model.load_weights('bin/model.h5')
    print('done')
    return model

def separate_characters(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        letter = img[y:y + h, x:x + w]
        cv2.imwrite(str(i) + '.png', letter)
        list.append(i)

    cv2.destroyAllWindows()
    return list

@app.before_first_request
def initialize():
    global mapping
    mapping = pickle.load(open('bin/mapping.p', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    model = load_model()

    def parseImage(imgData):
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open('output.png','wb') as output:
            output.write(base64.decodebytes(imgstr))

    parseImage(request.get_data())
    x = imread('output.png', mode='L')
    x = np.invert(x)
    imsave('resized.png', x)
    list = separate_characters('resized.png')

    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255

    out = model.predict(x)

    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}

    K.clear_session()
    return True

if __name__ == '__main__':
    app.run(debug=True)