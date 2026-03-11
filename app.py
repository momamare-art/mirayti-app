import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="تطبيق مرايتي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي")

# الرابط الخاص بك بعد التعديل
API_URL = "https://mo-1975-mirayti-api.hf.space/predict"

picture = st.camera_input("التقط صورة للإصابة")

if picture:
    with st.spinner('جاري التحليل...'):
        try:
            # إرسال الصورة للـ API الموجود على Hugging Face
            files = {'image': picture.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                result = data['result']
                conf = data['confidence']
                
                # عرض التقرير المطلوب في report_textview
                st.markdown("---")
                st.subheader("📝 تقرير الحالة (report_textview)")
                
                report_text = f"""
                نتائج الفحص المبدئي:
                - الحالة المتوقعة: {result}
                - نسبة التأكد: {conf*100:.2f}%
                """
                st.info(report_text)
            else:
                st.error("السيرفر لا يستجيب، تأكد أن Hugging Face Space في حالة Running")
        except Exception as e:
            st.error(f"خطأ في الاتصال: {e}")
