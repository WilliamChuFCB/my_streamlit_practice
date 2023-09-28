import streamlit as st
import pandas as pd
import torch
from PIL import Image
import io
import numpy as np

# set the title and text
st.title("Object Detection Using YOLO") 

markdown_content = '''
YOLO, or 'You Only Look Once,' is a real-time object detection system designed to identify and classify objects in images and videos efficiently. Unlike traditional methods that scan an image multiple times with different scales and aspect ratios, YOLO looks at the whole image only once. 

In this webpage, you can upload an image, and you will get the result bounding boxes on this image that indicate the locations of objects and their categories.
'''
st.markdown(markdown_content)

# set the widget for uploading image
st.header("Upload an image:")
image_upload = st.file_uploader("upload an image file for object detection", type=["PNG","JPG","JPEG"])

# object detection using YOLOv5
if image_upload is not None:
    # print the information of input image
    st.header("Image input:") 
    st.write("file:", image_upload)

    image_open = Image.open(image_upload) 
    sizels=list(image_open.size)
    sizels.append(3)
    size=tuple(sizels)
    st.write("X:", size)
    st.image(image_open, width=600)

    # run the model
    st.header("Object detection:") 
    st.subheader("Results")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True) 
    results = model(image_open) 

    img_byte_arr = io.BytesIO()
    rendered_img = results.render()
    img_array = np.array(rendered_img)
    img_pil = Image.fromarray(img_array[0])
    img_pil.save(img_byte_arr, format = 'JPEG')
    st.image(img_byte_arr.getvalue(), caption = 'Processed Image', use_column_width=True)

    # print results details
    results.pandas().xyxy[0]
