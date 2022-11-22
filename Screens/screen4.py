from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
import cv2 as cv
import numpy as np
from model import TensorFlowModel

class Screen4(Screen):
    
    def __init__(self, **kwargs):
        Builder.load_file('Screens/screen4.kv')
        super(Screen4, self).__init__(**kwargs)

    def on_enter(self, *args):
        self.ids.picture1.source = 'DCIM/DigitRecognitionApp/images/test.jpg'
    
    def analyse(self):
        source = 'DCIM/DigitRecognitionApp/images/test.jpg'
        
        original_image = cv.imread(source, cv.IMREAD_COLOR)
        gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        binary_image = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 10)
        contours, hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        images = []
        bounding_boxes = []
        for contour in contours:
            print(contour)
            if cv.contourArea(contour) < 100:
                continue
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            side_length = int(max(w, h) * 1.5)
            img = np.zeros((side_length, side_length), dtype=np.uint8)
            x_new, y_new = int(x + (w - side_length) / 2), int(y + (h - side_length) / 2)
            img = binary_image[y_new : y_new + side_length, x_new : x_new + side_length]
            try:
                img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
            except:
                continue
            img = img / 255
            images.append(img)
            bounding_boxes.append((x, y, w, h))
        if len(images) > 0:
            model = TensorFlowModel()
            model.load('model.tflite')
            predictions = model.pred(np.array(images))
            numbers = np.argmax(predictions, axis=1)
            print(numbers)
            for (x, y, w, h), number in zip(bounding_boxes, numbers):
                location = (x + int(w / 2), y - 10)
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 255, 0)
                cv.putText(original_image, str(number), location, font, fontScale, color, 2, cv.LINE_AA)
        cv.imwrite('DCIM/DigitRecognitionApp/images/analyzed.jpg', original_image)
        self.ids.picture1.source = 'DCIM/DigitRecognitionApp/images/analyzed.jpg'
