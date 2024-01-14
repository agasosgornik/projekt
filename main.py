import cv2
from flask import Flask
from flask_restful import Resource, Api
import requests

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)

class PeopleCounter(Resource):
    def get(self):
        # load image
        image = cv2.imread('people.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounter, '/')


img2 = requests.get("https://images.pexels.com/photos/1000754/pexels-photo-1000754.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1")
cv2.imread(img2)

if __name__ == '__main__':
    app.run(debug=True)
