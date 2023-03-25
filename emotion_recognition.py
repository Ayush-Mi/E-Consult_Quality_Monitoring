import tensorflow as tf
import cv2
import numpy as np
import json

class EmotionRecognition:
    def __init__(self,):
        self.model = tf.keras.models.load_model('./model/')
        with open('./class_labels.json', 'rb') as fp:
            self.labels = json.load(fp)
       
    def preprocessing(self,array):
        resized = cv2.resize(array,(48,48))
        gray = np.expand_dims(resized,axis=0)
        return gray 

    def get_label(self,i):
        label = list(self.labels.keys())[list(self.labels.values()).index(i)]
        return label
    
    def pred_emotion(self,image_array):
        img = self.preprocessing(image_array)
        pred = self.model(img)
        if np.max(pred)<0.7:
            return 'unidentified'
        else:
            pred = np.argmax(pred)
            pred = self.get_label(pred)
            return pred