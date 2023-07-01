import numpy as np
import tensorflow as tf
import streamlit as st
from keras.utils import custom_object_scope
import cv2
from keras.models import load_model
# model=load_model('dogbreed(1).h5')
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * self.scale_factor

# Register the custom layer
with custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
 model=load_model('Satellite_Image_Classification.h5')
CLASS_NAMES =['cloudy','desert','green_area','water']
st.title('Satellite Image Prediction')
st.markdown('Upload the Image')


dog_image= st.file_uploader("Upload image")
submit= st.button('Predict')
if submit:
    if dog_image is not None:
        file_bytes= np.asarray(bytearray(dog_image.read()),dtype= np.uint8)
        opencv_image= cv2.imdecode(file_bytes,1)


# Displaying the image
        st.image(opencv_image,channels="BGR")
        # Resizing the image
        opencv_image=cv2.resize(opencv_image,(224,224))
        # Convert image to 4 Dimenension
        opencv_image.shape=(1,224,224,3)
        # Make Prediction
        Y_pred=model.predict(opencv_image)
        
        print(Y_pred)
        st.success(str("The image is "+   CLASS_NAMES[np.argmax(Y_pred)]))