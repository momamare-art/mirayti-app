import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="تطبيق مرايتي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي")

# تحميل الموديل باستخدام OpenCV (أخف وأضمن طريقة للسيرفرات المجانية)
@st.cache_resource
def load_model_fast():
    # OpenCV يقدر يقرأ موديلات TFLite كأنها شبكة عصبية دقيقة
    net = cv2.dnn.readNetFromTFLite("model_small.tflite")
    return net

try:
    net = load_model_fast()
    
    img_file = st.camera_input("التقط صورة للإصابة")

    if img_file:
        # تجهيز الصورة
        image = Image.open(img_file).convert('RGB')
        full_img = np.array(image)
        
        # تحويل الصورة لمصفوفة بيفهمها الموديل (224x224)
        blob = cv2.dnn.blobFromImage(full_img, 1.0/255.0, (224, 224), (0,0,0), swapRB=True)
        net.setInput(blob)
        
        # التوقع
        output = net.forward()
        
        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة جلدية أخرى']
        idx = np.argmax(output)
        confidence = output[0][idx] * 100

        st.success(f"النتيجة: {classes[idx]}")
        st.info(f"نسبة التأكد: {confidence:.2f}%")

except Exception as e:
    st.error(f"السيرفر لسه بيعاندنا: {e}")
    st.info("تأكد أن ملف model_small.tflite موجود في GitHub")
