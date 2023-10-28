import numpy as np
import cv2
import tensorflow as tf
# from tensorflow import keras
from keras.models import load_model
# from keras.utils import load_img, img_to_array
# from keras.preprocessing import image


# from keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.mobilenet_v2 import decode_predictions
# from keras.applications.mobilenet_v2 import MobileNetV2


class_labels = ['matang', 'mentah']
model = load_model('NASNetMobile.h5')
cap = cv2.VideoCapture(0)

coffee = cv2.CascadeClassifier('cascade.xml')


while True :
    #capture Frame by frame
    ret, frame = cap.read()

    # our Operation on the frame 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coffee_bean = coffee.detectMultiScale(gray, 1.1, 5)

    # # Make a prediction


    for x,y,w,h in coffee_bean :
        kotak = cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 1)
        resized_frame = cv2.resize(kotak, (128, 128))
        expanded_frame = np.expand_dims(resized_frame, axis=0)
        normalized_frame = tf.keras.applications.nasnet.preprocess_input(expanded_frame)
        predictions = model.predict(normalized_frame)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        label = f"Class: {class_labels[predicted_class]}, Confidence: {confidence:.2f}"
        cv2.putText(kotak,label, (x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(0, 255, 0), 1)
    
    cv2.imshow('Klasifikasi Kematangan Kopi', frame)
    if cv2.waitKey(1) & 0xFF==('q'):
        break

cap.release()
cv2.destroyAllWindows()