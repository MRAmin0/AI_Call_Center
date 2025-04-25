from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
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

@app.route("/chat", methods=["POST"])
def chat():
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

if __name__ == "__main__":
    app.run(debug=True)
