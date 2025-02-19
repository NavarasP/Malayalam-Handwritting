from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import os
import base64
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "static/uploads"
CROPPED_FOLDER = "static/cropped_letters"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)


def predict_text(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "read the content in the image"},
            {"role": "user", "content": [
                {"type": "text", "text": "What's written in the image?"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


@app.route("/api/predict_text", methods=["POST"])
def api_predict_text():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    predictions = predict_text(filepath)
    
    return jsonify({"success": True, "text": predictions})


@app.route("/", methods=["GET", "POST"])
def index():
    # if "username" not in session:
    #     return redirect(url_for("login"))

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        predictions = predict_text(filepath)
        return render_template("index.html", uploaded_image=filepath, predictions=predictions)

    return render_template("index.html", prediction="")


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if username == "admin" and password == "password": 
        session["username"] = username
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
