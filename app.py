import streamlit as st
import google.generativeai as genai
from PIL import Image

# ده مفتاحك اللي شغالين بيه من الصبح
genai.configure(api_key="AIzaSyDdZyo-1J-KhHu32tC4uKaDx_v14vPBS6g")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="مرايتي - تحليل جلدي", page_icon="🩺")
st.title("🩺 تطبيق مرايتي - تحليل الأمراض الجلدية")
st.write("التقط صورة واضحة لمكان الإصابة وسيقوم الذكاء الاصطناعي بتحليلها.")

# الكاميرا
picture = st.camera_input("صور مكان الإصابة")

if picture:
    img = Image.open(picture)
    st.image(img, caption="الصورة قيد التحليل")
    
    with st.spinner('انتظر.. جاري استشارة الذكاء الاصطناعي...'):
        # ده البرومبت اللي تعبنا فيه عشان يحلل الأمراض بجد
        prompt = "أنت طبيب جلدي خبير. حلل هذه الصورة بدقة، حدد المرض الجلدي المحتمل (مثل الإكزيما، الصدفية، حب الشباب، إلخ)، واشرح الأعراض وقدم نصائح للعلاج. اجعل الرد باللغة العربية ومنظم."
        
        try:
            response = model.generate_content([prompt, img])
            st.subheader("📋 نتيجة التحليل المبدئي:")
            st.write(response.text)
        except Exception as e:
            st.error(f"حصلت مشكلة في التحليل: {e}")

    st.info("⚠️ تذكر: هذا التحليل استرشادي فقط، يجب زيارة الطبيب للتشخيص النهائي.")
