import os
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gem import fetch_definition_data

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_indices = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Background_without_leaves',
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}

# Load the trained model
model = tf.keras.models.load_model('plant_village_resnet50v2_39.h5')

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files['image']

        if file.filename == '':
            return render_template("index.html", error="No selected file")

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)
                prediction = model.predict(img_array)
                predicted_class = prediction.argmax(axis=-1)[0]

                # Map predicted class index to class label
                predicted_label = class_indices[predicted_class]
                predicted_label = (
                    predicted_label.replace(",", " ")  
                    .replace("_", " ")                 
                    .replace("-", " ")                 
                    .strip()                         
                )
                # Use relative path for display
                image_url = f"/{filepath}"

                # Fetch suggestions for the predicted disease
                suggestions = fetch_definition_data(predicted_label)

                return render_template("index.html", result=predicted_label, suggestions=suggestions, image_url=image_url)
            except Exception as e:
                os.remove(filepath)  # Clean up file even if there's an error
                return render_template("index.html", error=f"Error processing image: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
