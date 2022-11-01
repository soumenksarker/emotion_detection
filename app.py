import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
import tempfile
from keras_preprocessing import image
from keras_preprocessing.image import img_to_array
from time import sleep
from PIL import Image
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
st.title("emotions_detection_app")
#st.text("Build with Streamlit and OpenCV")
if "photo" not in st.session_state:
	st.session_state["photo"]="not done"

c2, c3 = st.columns([2,1])
def change_photo_state():
	st.session_state["photo"]="done"
@st.cache
def load_image(img):
	im = Image.open(img)
	return im
activities = ["Detect_emotion","About"]
choice = st.sidebar.selectbox("Select Activty",activities)
camera_photo = c2.camera_input("Take a photo", on_change=change_photo_state)
uploaded_photo = c2.file_uploader("Upload Image",type=['jpg','png','jpeg'], on_change=change_photo_state)
if choice == 'Detect_emotion':
    st.subheader("see your inner feels") 
    if st.session_state["photo"]=="done":
        if uploaded_photo:
            our_image= load_image(uploaded_photo)
        elif camera_photo:
            our_image= load_image(camera_photo)
        elif uploaded_photo==None and camera_photo==None:
            our_image= load_image('image.jpg')
        frame = np.array(our_image.convert('RGB'))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        labels = []
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y-10)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        st.image(frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
        
        # cap.release()
        # cv2.destroyAllWindows()




        # while True:
        #     _, frame = cap.read()
            