import streamlit as st
import os

# 1. إجبار السيرفر على تثبيت المكتبة الخفيفة فوراً أول ما يفتح
if os.system("pip install https://github.com/google-ai-edge/tflite-runtime/releases/download/v2.14.0/tflite_runtime-2.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl") != 0:
    os.system("pip install tflite-runtime")

import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

st.title("🩺 تطبيق مرايتي")

# تحميل الموديل
@st.cache_resource
def load_model():
    # لازم يكون الملف اسمه model_small.tflite عندك في GitHub
    interpreter = tflite.Interpreter(model_path="model_small.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_file = st.camera_input("التقط صورة")

    if img_file:
        img = Image.open(img_file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى']
        st.success(f"النتيجة: {classes[np.argmax(prediction)]}")
        st.write(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")

except Exception as e:
    st.error(f"السيرفر لسه معصلج: {e}")
