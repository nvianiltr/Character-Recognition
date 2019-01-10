# Main file for running Interactive Character Recognition

from flask import Flask, request, render_template, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import model_from_yaml
import functools
import cv2
import re
import base64
import pickle

app = Flask(__name__)

# sorting function for OpenCV countours with left-to-right direction
def greater(a, b):
    momA = cv2.moments(a)
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv2.moments(b)
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])

    if xa > xb:
        return 1
    if xa == xb:
        return 0
    else:
        return -1

def load_model():
    yaml_file = open('bin/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    model.load_weights('bin/model.h5')
    return model

# separate each character in canvas
def separate_characters(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filenames = []
    contours = sorted(contours, key=functools.cmp_to_key(greater))
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        letter = img[y:y + h, x:x + w]
        filename = 'bin/'+str(i)+'.png'
        cv2.imwrite(filename, letter)
        resize_image(filename)
        filenames.append(filename)

    cv2.destroyAllWindows()
    return filenames

# resizing the character's picture to 280x280 pixels
# so it matches the model's size
def resize_image(filename):
    img = Image.open(filename)
    x, y = img.size
    squared_img = Image.new('RGBA', (280, 280), "black")
    img.thumbnail((280, 280))
    offset = ((280 - x) // 2, (280 - y) // 2)
    squared_img.paste(img, offset)
    squared_img.save(filename)

@app.before_first_request
def initialize():
    global mapping
    mapping = pickle.load(open('bin/mapping.p', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    def parseImage(imgData):
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open('output.png','wb') as output:
            output.write(base64.decodebytes(imgstr))

    parseImage(request.get_data())
    x = imread('output.png', mode='L')
    x = np.invert(x)
    imsave('inverted_output.png', x)
    filenames = separate_characters('inverted_output.png')

    output = ''
    print(filenames)
    confidence = []
    for file in filenames:
        model = load_model()

        x = imread(file, mode="L")
        x = imresize(x,(28,28))
        x = x.reshape(1,28,28,1)
        x = x.astype('float32')
        x /= 255

        out = model.predict(x)
        output += chr(mapping[(int(np.argmax(out, axis=1)[0]))])
        confidence.append(str(max(out[0]) * 100)[:6])
        K.clear_session()

    confidence = [float(i) for i in confidence]
    average_confidence = np.mean(confidence)

    response = {'prediction': output,
                'confidence': average_confidence}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)