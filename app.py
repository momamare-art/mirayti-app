import streamlit as st

st.set_page_config(page_title="مرايتي - Mirayti", page_icon="✨")
st.title("✨ تطبيق مرايتي - Mirayti")
st.subheader("أهلاً بك يا محمد! الكاميرا جاهزة:")

picture = st.camera_input("التقط صورة لنفسك")

if picture:
    st.image(picture, caption="زي القمر يا محمد!")
    st.success("الموقع شغال 100%!")
