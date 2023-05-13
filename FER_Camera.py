import json
from keras.models import model_from_json
import cv2
import numpy as np

json_file = open("emotion_model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ( "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised")

def image_predict(image):
    return emotions[np.argmax(model.predict(image))]

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        # ret,test_img=self.video.read()# captures frame and returns boolean value and captured image  
          
        
        # gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
        # Find haar cascade to draw bounding box around face
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (1280, 720))
        # if not ret:
        #     break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)


        try:
            num_faces = face_haar_cascade.detectMultiScale(gray_frame, 1.32, 5)

            for (x,y,w,h) in num_faces:  
                # cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                # roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
                # roi_gray=cv2.resize(roi_gray,(48,48))  
                # img = roi_gray.reshape((1,48,48,1))
                # img = img /255.0

                # max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
                # predicted_emotion = emotions[max_index]  

                # cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotions[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except:
            pass

        resized_img = cv2.resize(frame, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()