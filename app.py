import streamlit as st 
import cv2
import numpy as np
import face_recognition 
from PIL import Image

st.title("Face Recognition")

image_train = st.sidebar.file_uploader("Upload an image to train")
detect_image = st.sidebar.file_uploader("Upload an image to test")

if image_train is not None and detect_image is not None:
    try:
        train_image = np.array(Image.open(image_train))
        test_image = np.array(Image.open(detect_image))

        if train_image.size == 0 or test_image.size == 0:
            st.write("One or both images are empty. Please upload valid images.")
        else:
            st.sidebar.image(train_image)
            st.sidebar.image(test_image)

            image_encodings_train = face_recognition.face_encodings(train_image)[0]
            image_encodings_test = face_recognition.face_encodings(test_image)[0]
            image_location_test = face_recognition.face_locations(test_image)

            results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
            dst = face_recognition.face_distance([image_encodings_test], image_encodings_train)

            if results:
                for (top, right, bottom, left) in image_location_test:
                    output_image = cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                st.image(output_image)
                st.write("Faces match")
            else:
                st.write("Faces don't match")

                st.subheader("Please provide your details")
                name = st.text_input("Name")
                email = st.text_input("Email")
                number = st.text_input("Number")
                address = st.text_area("Address")

                if st.button("Get Details"):
                    st.write("Details submitted successfully:")
                    st.write(f"Name: {name}")
                    st.write(f"Email: {email}")
                    st.write(f"Number: {number}")
                    st.write(f"Address: {address}")
    except Exception as e:
        st.write(f"Error: {e}")
else:
    st.write("Upload both images to perform face recognition")
