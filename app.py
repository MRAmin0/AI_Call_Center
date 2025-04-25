from flask import Flask, request, jsonify
from transformers import pipeline, BertTokenizerFast, BertForSequenceClassification
from flask_cors import CORS
import logging

# راه اندازی برنامه Flask
app = Flask(__name__)
CORS(app)

# تنظیمات مدل تحلیل احساسات
sentiment_pipeline = pipeline("sentiment-analysis", model="HooshvareLab/bert-fa-base-uncased")

# چت‌بات ساده برای سوالات متداول
faq = {
    "سفارش من کجاست؟": "سفارش شما در حال پردازش است.",
    "چطور رمز عبورم را تغییر دهم؟": "برای تغییر رمز عبور، به بخش تنظیمات حساب کاربری بروید.",
    "سلام": "سلام! چطور می‌توانم به شما کمک کنم؟",
    "خوبی": "بله، ممنون! چطور می‌توانم به شما کمک کنم؟",
    "مرسی": "خواهش میکنم، اگر کار دیگری داشتید درخدمتم!"

}

# تنظیمات لاگ برای نمایش خطاها
logging.basicConfig(level=logging.INFO)

# تابع چت برای پاسخ‌دهی به کاربران
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        # بررسی وجود پیام
        if not user_message:
            return jsonify({"error": "پیام ارسالی نمی‌تواند خالی باشد!"}), 400

        # تحلیل احساسات
        sentiment = sentiment_pipeline(user_message)[0]
        sentiment_label = "مثبت" if sentiment["label"] == "LABEL_1" else ("منفی" if sentiment["label"] == "LABEL_0" else "خنثی")
        
        # جستجو در سوالات متداول
        response = faq.get(user_message, "متاسفانه نمی‌توانم به این سوال پاسخ دهم.")
        
        # برگشت پاسخ و احساسات
        return jsonify({
            "response": response,
            "sentiment": sentiment_label,
            "score": sentiment["score"]
        })
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "خطای داخلی سرور رخ داده است!"}), 500


# اجرا در صورت اجرا در محیط تولید
if __name__ == "__main__":
    app.run(debug=True)
