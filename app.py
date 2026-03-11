import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="تطبيق مرايتي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي")

# تحميل الموديل باستخدام tflite_runtime
@st.cache_resource
def load_tflite_model():
    try:
        # تأكد أن اسم الملف هو model_small.tflite كما في الصورة
        interpreter = tflite.Interpreter(model_path="model_small.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"فشل تحميل الموديل: {e}")
        return None

interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    picture = st.camera_input("التقط صورة للإصابة")

    if picture:
        img = Image.open(picture).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى']
        res_idx = np.argmax(prediction)
        
        st.success(f"التشخيص المتوقع: {classes[res_idx]}")
        st.info(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
