import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="مرايتي - النسخة السريعة", page_icon="🩺")
st.title("🩺 تطبيق مرايتي")

# تحميل الموديل بدون TensorFlow
@st.cache_resource
def load_tflite_fast():
    # بنستخدم OpenCV هنا عشان هي أخف بكتير
    net = cv2.dnn.readNetFromTFLite("model_small.tflite")
    return net

try:
    net = load_tflite_fast()
    picture = st.camera_input("صور مكان الإصابة")

    if picture:
        img = Image.open(picture).convert('RGB')
        img = np.array(img)
        # تجهيز الصورة للموديل
        blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), (0,0,0), swapRB=True)
        net.setInput(blob)
        output = net.forward()
        
        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى']
        idx = np.argmax(output)
        confidence = output[0][idx] * 100

        st.success(f"التشخيص المتوقع: {classes[idx]}")
        st.info(f"نسبة التأكد: {confidence:.2f}%")

except Exception as e:
    st.error(f"حدث خطأ: {e}")
    st.info("تأكد أن ملف model_small.tflite موجود بجانب هذا الملف")
