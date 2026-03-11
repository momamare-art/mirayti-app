import os
import base64
import io
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

# الكي بتاعك - ده اللي بيشغل المخ
genai.configure(api_key="AIzaSyDdZyo-1J-KhHu32tC4uKaOx_v14vPBS6g")
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data"}), 400
            
        # تحويل الصورة من Base64 اللي بيبعتها الموقع
        image_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = "أنت مساعد طبي ذكي في تطبيق 'مرايتي'. حلل هذه الصورة واكتب تقرير مفصل عن الحالة الجلدية، الصحة العامة، والأنيميا بالعربية."
        
        response = model.generate_content([prompt, img])
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # ده عشان يشتغل على الهوست
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
