import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🩺 تطبيق مرايتي")

# تحميل الموديل
@st.cache_resource
def load_model():
    # التأكد من استخدام اسم الملف الصحيح الموجود في GitHub
    interpreter = tf.lite.Interpreter(model_path="model_small.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_file = st.camera_input("التقط صورة للإصابة")

    if img_file:
        img = Image.open(img_file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى']
        st.success(f"التشخيص المتوقع: {classes[np.argmax(prediction)]}")
        st.write(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
except Exception as e:
    st.error(f"حدث خطأ في التشغيل: {e}")
