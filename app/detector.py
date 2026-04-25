import cv2
import numpy as np

class FruitDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")
        self.classes = open("model/obj.names").read().strip().split("\n")

    def detect(self, image):
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        results = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    results.append({
                        "label": self.classes[class_id],
                        "confidence": float(confidence)
                    })

        return results
