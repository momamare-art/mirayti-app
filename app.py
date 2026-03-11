import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="مرايتي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي")

# تحميل الموديل باستخدام النسخة الرسمية
@st.cache_resource
def load_my_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model_small.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"مشكلة في الموديل: {e}")
        return None

interpreter = load_my_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_file = st.camera_input("صور مكان الإصابة")

    if img_file:
        # تجهيز الصورة للموديل
        img = Image.open(img_file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # تنفيذ التوقع
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # التصنيفات
        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى']
        res = classes[np.argmax(prediction)]
        
        st.success(f"التشخيص المتوقع: {res}")
        st.info(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
