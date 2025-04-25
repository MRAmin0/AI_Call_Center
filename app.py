from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from flask_cors import CORS
import torch
import threading
import webbrowser

app = Flask(__name__)
CORS(app)

# بارگذاری مدل GPT2 بهینه‌شده فارسی (روی CPU)
tokenizer = AutoTokenizer.from_pretrained("SajjadAyoubi/distilgpt2-fa")
model = AutoModelWithLMHead.from_pretrained("SajjadAyoubi/distilgpt2-fa")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU اجرا

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    prompt = f"کاربر: {user_message}\nبات:"
    output = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.9,
    )

    generated_text = output[0]['generated_text']
    answer = generated_text.split("بات:")[-1].strip().split("\n")[0]

    return jsonify({
        "response": answer
    })

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
