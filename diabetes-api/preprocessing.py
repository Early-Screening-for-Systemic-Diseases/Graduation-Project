import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class TonguePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def remove_shadows(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_image(self, img):
        img = np.array(img).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Shadow removal
        img = self.remove_shadows(img)

        # Convert to float for color correction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float32')

        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)

        # Clip values and convert back to uint8
        lab = np.clip(lab, 0, 255).astype('uint8')
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # segmentation step...
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 40, 50]); upper1 = np.array([20, 255, 255])
        lower2 = np.array([160, 40, 50]); upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            x = max(0, x-10); y = max(0, y-10)
            img = img[y:y+h+20, x:x+w+20]

        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return preprocess_input(img.astype('float32'))
