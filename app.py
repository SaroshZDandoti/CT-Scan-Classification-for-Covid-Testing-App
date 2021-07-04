# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:14:41 2021

@author: saniy
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow.keras
from PIL import Image, ImageOps


st.write( """
# Covid Testing based on Chest CT Scans
# """)

st.subheader("Steps to use:")
st.markdown("1. Open the sidebar on the top left")
st.markdown("2. Upload .jpg or .png file of your CT Scan")
with st.sidebar.header( "Upload your CT Scan"):
    imageF = st.sidebar.file_uploader("Upload Image",type = ['jpg','png'])
    
if imageF is not None:
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5', compile = False)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    st.image(imageF)
    image = Image.open(imageF).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # display the resized image
    #image.show()
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    prediction = model.predict(data)
    
    st.markdown("There's {:.2f} percent probability that the scan shows Covid ".format(prediction[0][0]*100))
    st.markdown("There's {:.2f} percent probability that the scan does not show Covid ".format(prediction[0][1]*100))

    df = [[prediction[0][0],prediction[0][1]]]
    
    data = pd.DataFrame(df, columns = ['Covid','Non-Covid'])
   
    st.bar_chart(data)
else:
    st.markdown("Awaiting for CT Scan of chest to be uploaded...")