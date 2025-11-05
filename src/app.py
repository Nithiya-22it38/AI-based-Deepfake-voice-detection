from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from src.infer import predict_file  # your ML inference function

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    file = request.files["audio"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run model prediction
    result = predict_file(filepath)

    return jsonify({
        "prediction": result["label"],       # e.g., REAL or FAKE
        "confidence": result["confidence"],  # e.g., 0.93
        "reason": result["reason"]           # optional explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
