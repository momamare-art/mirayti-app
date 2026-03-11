import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="تطبيق مرايتي", page_icon="🩺")

st.markdown("<h1 style='text-align: center;'>🩺 تطبيق مرايتي</h1>", unsafe_allow_html=True)
st.write("قم بالتقاط صورة لمكان الإصابة الجلدية للتحليل")

# وظيفة تحميل الموديل
@st.cache_resource
def load_my_model():
    # تأكد أن هذا الاسم هو نفس اسم الملف في GitHub
    interpreter = tf.lite.Interpreter(model_path="model_small.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_my_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # فتح الكاميرا
    picture = st.camera_input("التقط صورة")

    if picture:
        # معالجة الصورة
        img = Image.open(picture).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # تشغيل الموديل (تعب الـ 8 ساعات)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # تصنيف النتائج (تأكد من الترتيب حسب تدريبك في كولاب)
        classes = ['إكزيما', 'صدفية', 'حب شباب', 'حالة جلدية أخرى']
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # عرض النتيجة
        st.success(f"التشخيص المتوقع: {result}")
        st.info(f"نسبة التأكد: {confidence:.2f}%")
        
        if confidence < 50:
            st.warning("نسبة التأكد ضعيفة، يرجى تحسين الإضاءة والتصوير بوضوح.")

except Exception as e:
    st.error(f"خطأ في تشغيل التطبيق: {e}")
    st.info("تأكد من رفع ملف model_small.tflite في نفس المجلد")

st.markdown("---")
st.caption("ملاحظة: هذا التطبيق استرشادي فقط ولا يعتد به كبديل للكشف الطبي.")
