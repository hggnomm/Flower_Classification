import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%"
    return outcome

uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Tạo đường dẫn lưu file tạm
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Lưu file
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Hiển thị ảnh
    st.image(uploaded_file, width=300)

    # Phân loại ảnh và hiển thị kết quả
    result = classify_images(temp_path)
    st.write(result)
