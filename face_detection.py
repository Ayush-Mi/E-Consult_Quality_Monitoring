import cv2

class FaceDetection:
    def __init__(self,):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def detect_face(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)

        face_coordinates = []
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_coordinates.append(image[y:y + h, x:x + w])

        return face_coordinates