import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision
from PIL import Image
import io

app = Flask(__name__, static_folder=".", static_url_path="")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "."
app.config["MAX_CONTENT_LENGTH"] = (
    200 * 1024 * 1024
)  # 200MB max file size (for model files)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["ALLOWED_MODEL_EXTENSIONS"] = {"pth"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PikachuNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(128 * 128 * 4, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = None
model_path = "pikachu_classifier.pth"


def load_model(path=None):
    """Load or reload the model from a .pth file"""
    global model, model_path
    if path is None:
        path = model_path

    if model is None:
        model = PikachuNet().to(device)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model_path = path
        print(f"Model loaded successfully from {path}")
        return True
    else:
        print(f"Model file {path} not found!")
        return False


if not load_model():
    raise FileNotFoundError(f"Model file {model_path} not found!")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def allowed_model_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_MODEL_EXTENSIONS"]
    )


def preprocess_image(image):
    """Preprocess image for the model"""
    # Resize to 128x128
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    image = image.convert("RGBA")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def predict_image(image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        is_pikachu = probability >= 0.5
    return probability, is_pikachu


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({"error": "Invalid file type. Please upload a PNG or JPG image."}),
            400,
        )

    try:
        file.seek(0)

        image_bytes = file.read()

        if not image_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400

        # Verify and open image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify it's actually an image by trying to load it
            image.verify()
        except Exception as img_error:
            return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400

        image = Image.open(io.BytesIO(image_bytes))

        image_tensor = preprocess_image(image)

        probability, is_pikachu = predict_image(image_tensor)

        # Format result
        result = {
            "is_pikachu": bool(is_pikachu),
            "confidence": round(probability * 100, 2),
            "message": f"{'Pikachu detected!' if is_pikachu else 'Not Pikachu'} (Confidence: {probability * 100:.2f}%)",
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/upload_model", methods=["POST"])
def upload_model():
    """Endpoint to upload and reload a new .pth model file"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_model_file(file.filename):
        return (
            jsonify({"error": "Invalid file type. Please upload a .pth model file."}),
            400,
        )

    try:
        file.seek(0)

        model_bytes = file.read()

        if not model_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400

        # Save the uploaded model to a temporary location first
        temp_model_path = "temp_model.pth"
        with open(temp_model_path, "wb") as f:
            f.write(model_bytes)

        try:
            if load_model(temp_model_path):
                if (
                    os.path.exists(model_path)
                    and model_path != "pikachu_classifier.pth"
                ):
                    # Keep backup of old model
                    backup_path = model_path.replace(".pth", "_backup.pth")
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(model_path, backup_path)

                final_path = secure_filename(file.filename)
                if not final_path.endswith(".pth"):
                    final_path = "pikachu_classifier.pth"
                else:
                    final_path = os.path.join(app.config["MODEL_FOLDER"], final_path)

                # Move temp file to final location
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_model_path, final_path)
                model_path = final_path

                return jsonify(
                    {
                        "success": True,
                        "message": f"Model successfully updated from {file.filename}",
                        "model_path": model_path,
                    }
                )
            else:
                # Clean up temp file if loading failed
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                return (
                    jsonify(
                        {
                            "error": "Failed to load model. The file may be corrupted or incompatible."
                        }
                    ),
                    400,
                )

        except Exception as load_error:
            # Clean up temp file on error
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            return jsonify({"error": f"Error loading model: {str(load_error)}"}), 400

    except Exception as e:
        return jsonify({"error": f"Error processing model file: {str(e)}"}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the currently loaded model"""
    return jsonify(
        {
            "model_path": model_path,
            "model_exists": os.path.exists(model_path),
            "model_size": (
                os.path.getsize(model_path) if os.path.exists(model_path) else 0
            ),
        }
    )


if __name__ == "__main__":
    app.run()
