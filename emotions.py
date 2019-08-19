import cv2
import numpy as np

from keras.models import load_model
from statistics import mode
from dataset import get_labels
from tool import draw_text
from tool import draw_bounding_box
from tool import apply_offsets
from preprocessor import preprocess_input

use_webcam = True

emotion_model_path = 'emotion_model.hdf5'


emotion_labels = get_labels('fer2013')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

emotion_target_size = emotion_classifier.input_shape[1:3]


emotion_window = []


cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)


cap = None
if (use_webcam == True):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('dinner.mp4')




while cap.isOpened():

    
    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)



    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, (20, 40))
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > 10:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()



        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)


    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)




    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
