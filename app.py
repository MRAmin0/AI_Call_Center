from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from flask_cors import CORS
import torch
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

# مدل GPT2 فارسی
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
model = AutoModelWithLMHead.from_pretrained("HooshvareLab/gpt2-fa")

# لود کردن pipeline تولید متن
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    # تولید پاسخ با GPT2
    prompt = f"کاربر: {user_message}\nبات:"
    response = generator(prompt, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']

    # استخراج فقط جمله بعد از "بات:"
    answer = response.split("بات:")[-1].strip().split("\n")[0]

    return jsonify({
        "response": answer
    })

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
