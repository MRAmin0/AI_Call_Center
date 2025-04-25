from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from flask_cors import CORS
import webbrowser
import threading

app = Flask(__name__)  # از Flask بدون نیاز به static_folder استفاده می‌کنیم
CORS(app)  # برای دسترسی به API از مرورگرهای مختلف

# لود مدل تحلیل احساسات (مدل فارسی)
sentiment_pipeline = pipeline("sentiment-analysis", model="HooshvareLab/bert-fa-base-uncased")

# چت‌بات ساده برای جواب دادن به سوالات متداول
faq = {
    "سفارش من کجاست؟": "سفارش شما در حال پردازش است.",
    "چطور رمز عبورم را تغییر دهم؟": "برای تغییر رمز عبور، به بخش تنظیمات حساب کاربری بروید.",
    "سلام": "سلام! چطور می‌توانم به شما کمک کنم؟",
    "خوبی": "بله، ممنون! چطور می‌توانم به شما کمک کنم؟"
}

@app.route("/")
def home():
    """صفحه اصلی که پیام خوش‌آمدگویی نمایش می‌دهد"""
    return render_template('index.html')  # استفاده از render_template برای بارگذاری فایل HTML از پوشه templates

@app.route("/chat", methods=["POST"])
def chat():
    """مسیر چت که پیام کاربر را می‌گیرد و پاسخ می‌دهد"""
    data = request.json
    user_message = data.get("message", "")

    # تحلیل احساسات
    sentiment = sentiment_pipeline(user_message)[0]
    sentiment_label = "مثبت" if sentiment["label"] == "LABEL_1" else "منفی"
    
    # پیدا کردن پاسخ به سوالات متداول
    response = faq.get(user_message, "متاسفانه نمی‌توانم به این سوال پاسخ دهم.")
    
    return jsonify({
        "response": response,
        "sentiment": sentiment_label,
        "score": sentiment["score"]
    })

def open_browser():
    """این تابع برای باز کردن مرورگر به طور خودکار استفاده می‌شود."""
    webbrowser.open("http://127.0.0.1:5000", new=2)  # new=2 برای باز کردن در یک تب جدید

if __name__ == "__main__":
    # برای اینکه مرورگر خودکار باز شود، سرور را در یک نخ جداگانه اجرا می‌کنیم
    threading.Timer(1, open_browser).start()  # باز کردن مرورگر 1 ثانیه پس از شروع سرور
    app.run(debug=True, port=5000)
