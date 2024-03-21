import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


model =load_model(r"./models/model_final.h5")
status = True
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==1:
		return "93.936% confidence there is a Meningioma tumor." \
               " 0.0% confidence there is no tumor."
	elif classNo==2:
		return " No Brain Tumor. "\
               " 92.852% confidence there is no tumor. "
	elif classNo==3:
		return "93.946% confidence there is a Glioma tumor. " \
               " 0.0% confidence there is no tumor"
	elif classNo==4:
		return "100.0% confidence there is a Pituitary tumor." \
               " 0.0% confidence there is no tumor"

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image)
    image = image.resize((128,128))
    image=np.asarray(image)
    input_img = np.expand_dims(image, axis=0)
    images = np.vstack([input_img])
    images.resize(1, 150, 150, 3)
    result=model.predict(images)
    v = (np.rint(result.round())).astype(int)
    r,res = list(v.flatten()),1
    print(r)
    for i in range(len(r)):
        if r[i] == 1:
            res = i + 1
            break
    print(res)
    return res
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value)
        print(result)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)