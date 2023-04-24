# Realtime E-Consult Quality Monitoring
This project is about getting realtime feedback of a client while consulting virtually using video conference. It uses realtime emotion detection to observe and monitor the behavior of a person in a video by classifying them into seven classes: Happy, Sad, Angry, Surprised, Disgust, Fear and Neutral.

![](https://github.com/Ayush-Mi/Realtimg_E-Consult_Quality_Monitoring/blob/main/static/demo.gif)

## Application
This can be easily integrated with any video calling application to monitor the facial expression of the customer/client in realtime. The code can also be modified to capture the expressions of multiple people participating in video conference to monitor the impact of a presentation or talk.

## Architecture
![](https://github.com/Ayush-Mi/Realtimg_E-Consult_Quality_Monitoring/blob/main/images/Architecture.png)


## Face Detection

The face detection model is simply a open source pretrained haar cascade frontal face detection model by openCV. More information about this model can be found [here](https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html).

## Emotion Recognition

For the emotion recognition part, I finetuned a keras version of mobilenet model which was pretrained on imagenet dataset. I used the open source facial emotion recogntion dataset on [kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). It had ~35k grayscale images of size 48 by 48 with each belonging to one of the seven labels: Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise.

#### Training

The model was trained on Macbook M1pro for 100 epochs with each epoch having an average time of ~85 seconds. I model had ~3.2M trainable parameters and was trained using Adam optimizer with categorical crossentropy as loss function. 

![](https://github.com/Ayush-Mi/Realtimg_E-Consult_Quality_Monitoring/blob/main/images/model_arch.png)

As we can see in the below graph, the model clearly overfits the training data with the increasing different between train and val loss after ~23rd epoch. Few reasons for this could be the data quality at hand i.e with grayscale image of size 48 by 48 it is difficult to detect unique patters for specific classes.

Accuracy Plot | Train Loss Plot
:---: | :---:
![](https://github.com/Ayush-Mi/Realtimg_E-Consult_Quality_Monitoring/blob/main/images/cnn_accuracy_plot.png) | ![](https://github.com/Ayush-Mi/Realtimg_E-Consult_Quality_Monitoring/blob/main/images/cnn_loss_plot.png)

## Flask App

The python flask app here is just for the purpose of demonstration of the use case. It is basically what an end product would look like with client and consultant connecting with each other over a video call. In the demo shown above it takes two gifs and predicts emotion of the gif labelled as client but the same can be replicated to use a webcam as an input or reading an image from the web browser directly.

## Requirements

The whole project uses python 3.8 and below mentioned libraries:

`pip intsll pandas==1.4.4`

`pip intsll numpy==1.23.2`

`pip install tensorflow==2.9.2`

`pip install tensorboard`

`pip intall re==2.2.1`

`pip install flask`

## How to run

- Download the required [dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from the kaggle and use train.ipynb notebook to train your own emotion recogition model
- Once the model is trained run the app.py which starts a local HTML server. This page can be visited on 127.0.0.0 on your local browser.

## References

- Dataset was taken from [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- Pretrained Frontal Face recognition model from [openCV](https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html)
- Multiple tutorials for building Flask app.
- Gifs takes google

