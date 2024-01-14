import cv2
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
from urllib.request import urlopen

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)

class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('people.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterDynamic(Resource):
    def get(self):
        url = request.args.get('url')
        image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        return {'peopleCount': len(rects)}



api.add_resource(PeopleCounterStatic, '/')

api.add_resource(PeopleCounterDynamic, '/dynamic')


if __name__ == '__main__':
    app.run(debug=True)

