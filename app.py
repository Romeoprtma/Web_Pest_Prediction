from flask import Flask, render_template, request
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('cnnPestPrediction.h5', custom_objects={'BatchNormalization': BatchNormalization}, compile=False)

# Mapping from class index to pest name
class_to_pest = {
    0: "Asiatic Rice Borer",
    1: "Brown Plant Hopper",
    2: "Rice Gall Midge",
    3: "Rice Leaf Roller",
    4: "Rice Leafhopper",
    5: "Rice Water Wevil",
    # Add more mappings as needed
}

app = Flask(__name__)

@app.route('/base.html')
def statistics():
    return render_template('base.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction="No file provided")

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction="No file provided")

    try:
        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        # Preprocess the image to match the model's input shape
        image = image.resize((312, 312))
        image = np.array(image)

        if image.shape[-1] == 4:
            image = image[..., :3]  # Convert RGBA to RGB if needed

        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize to [0, 1]

        # Make prediction
        prediction = model.predict(image)

        # Check if prediction is empty or not
        if len(prediction) == 0:
            return render_template('result.html', prediction="Failed to predict class")

        predicted_class = int(np.argmax(prediction, axis=1)[0])

        # Get the name of the pest based on predicted class
        predicted_pest = class_to_pest.get(predicted_class, "Gambar tidak terdeksi")

        return render_template('result.html', prediction=predicted_pest)

    except Exception as e:
        return render_template('result.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
