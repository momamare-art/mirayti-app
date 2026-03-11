import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# تحميل الموديل بتاعك
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_model.h5')

model = load_my_model()

st.title("🩺 تطبيق مرايتي - تشخيصك الأصلي")

picture = st.camera_input("التقط صورة للإصابة")

if picture:
    img = Image.open(picture).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التوقع
    prediction = model.predict(img_array)
    # ملاحظة: تأكد من ترتيب الأمراض كما فعلت في كولاب
    classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة أخرى'] 
    result = classes[np.argmax(prediction)]
    
    st.subheader(f"التشخيص المتوقع: {result}")
    st.write(f"نسبة التأكد: {np.max(prediction)*100:.2f}%")
