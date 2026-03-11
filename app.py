import streamlit as st
import numpy as np
from PIL import Image
# استيراد المكتبة الخفيفة
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

st.title("🩺 تطبيق مرايتي")

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_small.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
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
    result = classes[np.argmax(prediction)]
    
    st.success(f"التشخيص: {result}")
    st.write(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
