import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعداد الصفحة
st.set_page_config(page_title="مرايتي - تحليل أصلي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي - التشخيص بموديل كولاب")

# تحميل الموديل الصغير اللي إنت رفعته
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_small.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    picture = st.camera_input("صور مكان الإصابة")

    if picture:
        img = Image.open(picture).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # تشغيل التوقع
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # الأمراض بالترتيب (عدلها لو الترتيب في كولاب كان مختلف)
        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى'] 
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success(f"النتيجة المتوقعة: {result}")
        st.info(f"نسبة التأكد: {confidence:.2f}%")
except Exception as e:
    st.error(f"حصلت مشكلة في تحميل الموديل: {e}")

st.warning("تذكر: هذا التطبيق تعليمي ولا يغني عن زيارة الطبيب.")
