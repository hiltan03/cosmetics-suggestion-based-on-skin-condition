# from flask import Flask, render_template, request, jsonify
# import numpy as np
# from keras.preprocessing import image
# from keras.models import model_from_json
# import os

# app = Flask(__name__)

# # Load the model
# json_file = open('model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model1.h5")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         if file:
#             img_path = os.path.join("static/uploads", file.filename)
#             file.save(img_path)

#             prediction = classify_skin(img_path)

#             return jsonify({"prediction": prediction, "image_path": img_path})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500  # Send error details

# if __name__ == '__main__':
#     app.run(debug=True)


import os
from flask import Flask, request, render_template, url_for
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PRODUCT_FOLDER = "static/products"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRODUCT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
try:
    with open('model1.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model1.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()


# 🔹 First define upload_file() before using it
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("❌ No file part in request.")
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        print("❌ No file selected.")
        return "No selected file", 400

    print(f"📂 Received file: {file.filename}")

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        file.save(img_path)  # Save the file
        print(f"✅ File saved to {img_path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return "Error saving file", 500

    # Call classify() to get skin type and recommended product filename
    skin_type, product_image = classify(img_path)

    return render_template("result.html", 
                           user_image=img_path, 
                           skin_type=skin_type, 
                           product_image=product_image)

# 🔹 Now define index() AFTER upload_file()
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # Redirect form submission to '/upload'
        return upload_file()  
    return render_template("index.html")


# Function to classify skin type and return corresponding product image filename
def classify(img_file):
    if loaded_model is None:
        print("❌ Model not loaded. Cannot classify image.")
        return "Error: Model not loaded", "default.jpg"

    if not os.path.exists(img_file):
        print(f"❌ Image file not found: {img_file}")
        return "Error: Image file not found", "default.jpg"

    try:
        test_image = load_img(img_file, target_size=(128, 128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        print(f"✅ Image successfully processed: {img_file}")

        result = loaded_model.predict(test_image)

        print(f"🔍 Model prediction result: {result}")

        a = np.round(result[0][0])
        b = np.round(result[0][1])
        c = np.round(result[0][2])
        d = np.round(result[0][3])       

        if a == 1:
            skin_type = "Dry Skin"
            product_image = "dry_skin_products.jpg"
        elif b == 1:
            skin_type = "Normal Skin"
            product_image = "normal_skin_products.jpg"
        elif c == 1:
            skin_type = "Oily Skin"
            product_image = "oily_skin_products.jpg"
        elif d == 1:
            skin_type="default_skin"
            product_image="himalaya.png"            
        else:
            skin_type = "Unknown"
            product_image = "default.jpg"

        return skin_type, product_image

    except Exception as e:
        print(f"❌ Error in classify function: {e}")
        return "oil skin", "himalaya.png"


if __name__ == "__main__":
    app.run(debug=False, port=8000)
