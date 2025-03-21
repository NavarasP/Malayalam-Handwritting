from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import os
import base64
from openai import OpenAI


load_dotenv()

app = Flask(__name__)
CORS(app)  
bcrypt = Bcrypt(app)  

app.secret_key = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL") 
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

with app.app_context():
    db.create_all()


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")

    if not username or not password or not email:
        return jsonify({"success": False, "message": "All fields are required"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    new_user = User(username=username, password=hashed_password, email=email)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True, "message": "User registered successfully"})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password, password):
        session["user_id"] = user.id
        return jsonify({
            "success": True, 
            "message": "Login successful", 
            "user_id": user.id 
        })
    
    return jsonify({"success": False, "message": "Invalid credentials"}), 401



@app.route("/api/logout", methods=["GET"])
def logout():
    session.pop("user_id", None)
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route("/api/save_note", methods=["POST"])
def save_note():
    data = request.get_json()
    user_id = data.get("user_id") 

    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    content = data.get("content")
    if not content:
        return jsonify({"success": False, "message": "Note content is required"}), 400

    new_note = Note(user_id=user_id, content=content)
    db.session.add(new_note)
    db.session.commit()

    return jsonify({"success": True, "message": "Note saved successfully"})



@app.route("/api/get_notes", methods=["GET"])
def get_notes():
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"success": False, "message": "Invalid User ID"}), 400

    notes = Note.query.filter_by(user_id=user_id).all()
    note_list = [{"id": n.id, "content": n.content, "created_at": n.created_at.isoformat()} for n in notes]

    return jsonify({"success": True, "notes": note_list})





@app.route("/api/get_profile", methods=["GET"])
def get_profile():
    user_id = request.args.get("user_id")  

    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    user = db.session.get(User, user_id)
    print(user.username)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    return jsonify({"success": True, "profile": {"username": user.username, "email": user.email}})



def predict_text(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages = [
    {"role": "system", "content": "You are an expert in reading and extracting Malayalam text from images."},
    {"role": "user", "content": [
        {"type": "text", "text": "Extract and return only the Malayalam text from this image. Do not provide any explanation, translation, or additional details. Strictly output only the Malayalam text as it appears in the image."},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        }}
    ]}
],
        temperature=0.0,
    )
    return response.choices[0].message.content



@app.route("/api/predict_text", methods=["POST"])
def api_predict_text():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    predictions = predict_text(filepath)
    
    return jsonify({"success": True, "text": predictions})




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files: 
            return jsonify({"success": False, "message": "No file uploaded"})

        file = request.files["image"]
        if file.filename == "":  
            return jsonify({"success": False, "message": "No file selected"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        predictions = predict_text(filepath)  
        return jsonify({"success": True, "text": predictions})

    return render_template("index.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
