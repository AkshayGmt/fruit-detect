import cv2
import numpy as np
config_path = "yolov3.cfg"
weights = "yolov3.weights"
classes = open("obj.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights)


cap = cv2.VideoCapture('fruit-47166.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            object_img = img[y:y+h, x:x+w]
            cv2.imwrite( f'object.jpg', img)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# Set a threshold for the minimum confidence level required for detection

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
# Load YOLO object detection model and its configuration file

config_path = "yolov3.cfg"
weights = "yolov3-helmet.weights"
labels = open("helmet.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights)
def model_output(path_name):
    image = cv2.imread(path_name)
    h,w = image.shape[:2]
        # Create a blob from the input frame and pass it through the network

    blob = cv2.dnn.blobFromImage(image, 1/255.0,(416,416), swapRB = True, crop = False)
    
    net.setInput(blob)  # Sets the new input value for the network
    
    ln = net.getLayerNames()
    print(ln)
    print('New value',net.getLayerNames())
    #ln is a list comprsisng all models in config file
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # Run forward pass through YOLO model

    layer_outputs = net.forward(ln)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            # Get class ID and confidence
            scores = detection[5:]
            # Extract the class ID and confidence level of the current detection

            class_id = np.argmax(scores)
            #print(class_id)
            confidence = scores[class_id]
            # Filter out detections that don't meet the minimum confidence threshold

            if confidence>CONFIDENCE:
            # Extract the coordinates of the bounding box for the object

                box = detection[:4]*np.array([w,h,w,h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def detection_recognition(path_name):
    image = cv2.imread(path_name)
    boxes, confidences, class_ids = model_output(path_name)
    # Apply non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    font_scale = 1
    thickness= 1
    if len(idxs)>0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x,y), (x+w, y+h), color = (255,20,147), thickness = thickness)
            object_img = image[y:y+h, x:x+w]
            
            # Display image with boxes
            cv2.imwrite( f'object1.jpg', object_img)

            text = f"{labels[class_ids[i]]}:{confidences[i]:.2f}"
            #Text  size
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness = thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            #rectangle box create for object
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color = (255,20,147), thickness = cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            #text apply in image
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            #image showing
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

detection_recognition("object.jpg")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support
import cv2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import tensorflow as tf
import os
import pathlib
import PIL
from tensorflow.keras.preprocessing.image import ImageDataGenerator 



import os
train_path = 'Training/'
test_path = 'Test/'


batch_size = 100
image_width = 100
image_height = 100
random_state = 100
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split = 0.2,
    subset = "training",
    seed = random_state,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

print(train_data)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split = 0.2,
    subset = "validation",
    seed = random_state,
    image_size = (image_height, image_width),
    batch_size = batch_size
)
# Found 67692 files belonging to 131 classes.
# Using 13538 files for validation.
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size = (image_height, image_width),
    batch_size = batch_size
)
print(test_data)

class_names = train_data.class_names
print(class_names)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data = train_data.cache().shuffle(random_state).prefetch(buffer_size = AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size = AUTOTUNE)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)]
)
data_preprocessing = tf.keras.applications.resnet.preprocess_input
base_model = tf.keras.applications.resnet.ResNet50(
    input_shape = (image_height, image_width, 3),
    include_top = False,
    weights = "imagenet"
)


base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names))
#Creating final model
inputs = tf.keras.Input((100, 100, 3))
x = data_augmentation(inputs)
x = data_preprocessing(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.05)

model.compile(
    optimizer = optimizer,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5)
history = model.fit(train_data, validation_data = validation_data, batch_size = 100, epochs = 1, callbacks=[early_stopping])

result = model.evaluate(test_data)
print(f"Loss: {result[0]}")
print(f"Accuracy: {result[1] * 100}")

metrics = pd.DataFrame(model.history.history)
print(metrics)

model.evaluate(test_data)

D = {'Apple Braeburn': 0,
 'Apple Crimson Snow': 1,
 'Apple Golden 1': 2,
 'Apple Golden 2': 3,
 'Apple Golden 3': 4,
 'Apple Granny Smith': 5,
 'Apple Pink Lady': 6,
 'Apple Red 1': 7,
 'Apple Red 2': 8,
 'Apple Red 3': 9,
 'Apple Red Delicious': 10,
 'Apple Red Yellow 1': 11,
 'Apple Red Yellow 2': 12,
 'Banana': 13,
 'Banana Lady': 14,
 'Banana Red': 15,
 'Lemon': 16,
 'Lemon Meyer': 17,
 'Orange': 18}

key_list = list(D.keys())
val_list = list(D.values())

pred = np.argmax(model.predict(test_data),axis=-1)
from sklearn.metrics import classification_report
#print(classification_report(test_data.classes,pred))

single = cv2.imread("1_100.jpg")
single = cv2.resize(single,(100,100))
single = single.reshape((100, 100, 3))
plt.imshow(single)

val = np.argmax(model.predict(single.reshape(1,100,100,3)),axis=-1)
pos = val_list.index(val)
print(f"The give image is a/an {key_list[pos]}")








