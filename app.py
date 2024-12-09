from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import boto3
import tensorflow as tf
import tensorflow_addons as tfa
import zipfile
import logging
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import base64
import requests

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# S3 configuration
AWS_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

PICKLE_FILE_KEY = "dataset_embeddingss.pkl"
MODEL_ZIP_FILE_KEY = "siamese_model.h5"
LOCAL_PICKLE_FILE = "/tmp/dataset_embeddingss.pkl"
LOCAL_MODEL_ZIP_FILE = "/tmp/siamese_model.h5"
LOCAL_MODEL_DIR = "/tmp/siamese_model"

# Custom objects for loading the model
custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
target_size = (28, 28)


def generate_presigned_url(bucket_name, object_key, expiration=3600):
	s3 = boto3.client('s3', region_name='us-east-2') 
	try:
		url = s3.generate_presigned_url( 
			ClientMethod='get_object', 
			Params={'Bucket': bucket_name, 'Key': object_key}, 
			ExpiresIn=expiration 
		) 
		print(f"Generated URL: {url}") 
		return url 
	except Exception as e: 
		logging.error(f"Error generating pre-signed URL: {str(e)}") 
		return None


def download_image_from_url(url, local_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # Send an HTTP GET request to fetch the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Write the content to a local file
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Image downloaded successfully to {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")

def download_from_s3(s3_key, local_path):
    """Download a file from S3."""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3_client.download_file(AWS_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Downloaded {s3_key} from S3 to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        raise


def load_model_and_embeddings():
    """Download and load the model and embeddings from S3."""
    # Download and extract the model
    download_from_s3(MODEL_ZIP_FILE_KEY, LOCAL_MODEL_ZIP_FILE)
    siamese_model = load_model(LOCAL_MODEL_ZIP_FILE, custom_objects=custom_objects)

    # extract_zip_file(LOCAL_MODEL_ZIP_FILE, LOCAL_MODEL_DIR)
    # embedding_model = tf.keras.models.load_model(LOCAL_MODEL_DIR, custom_objects=custom_objects)
    embedding_model = siamese_model.layers[3]
    logger.info("Embedding model loaded successfully.")

    # Download and load the pickle file
    download_from_s3(PICKLE_FILE_KEY, LOCAL_PICKLE_FILE)
    with open(LOCAL_PICKLE_FILE, "rb") as f:
        data = np.load(f, allow_pickle=True)
    dataset_embeddings, filenames = data["embeddings"], data["filenames"]
    logger.info(f"Loaded {len(filenames)} embeddings from pickle file.")
    return embedding_model, dataset_embeddings, filenames

    

def preprocess_image(file_or_path, target_size):
    """
    Preprocess the image for the model.
    
    :param file_or_path: Path to the image file or an io.BytesIO object.
    :param target_size: Tuple specifying the target size (height, width).
    :return: Preprocessed image as a NumPy array.
    """
    try:
        if isinstance(file_or_path, BytesIO):
            # Load the image directly from BytesIO
            image = load_img(file_or_path, target_size=target_size, color_mode="rgb")
        else:
            # Load the image from a file path
            image = load_img(file_or_path, target_size=target_size, color_mode="rgb")

        # Convert the image to a NumPy array and normalize
        image_array = img_to_array(image) / 255.0
        # Add a batch dimension
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise

def find_closest_embedding(target_embedding, dataset_embeddings, filenames, k=5):
    """Find the closest embeddings to the target embedding."""
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(filenames[idx], distances[idx]) for idx in closest_indices]

@app.route('/')
def index():
    return render_template('index.html', processed_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Open the uploaded image in memory
        img = Image.open(file.stream)

        # Process the image
        processed_img = process_image(img)

        # Convert the processed image to a base64 string
        img_io = BytesIO()
        processed_img.save(img_io, format="PNG")
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"

        # Pass the image data URL to the template
        return render_template('index.html', processed_image=img_data_url)

def process_image(img):
    # Example processing: Invert colors
    # download_from_s3("anchor/068.jpeg","tmp/068.jpeg")
    # processed_image =Image.open("tmp/068.jpeg")
    # processed_img = img.convert("RGB")  # Ensure image is in RGB mode
    # inverted_img = Image.eval(processed_img, lambda x: 255 - x)

    bucket_name = "capstone-bucket-heroku"
    object_key = "anchor/068.jpeg"

    # Generate the pre-signed URL
    url = generate_presigned_url(bucket_name, object_key)
    if url:
        download_image_from_url(url, "tmp/068.jpeg")
        processed_image =Image.open("tmp/068.jpeg")
        processed_img = img.convert("RGB")
        return processed_image
    else:
        raise Exception("Failed to generate pre-signed URL")

if __name__ == "__main__":
    app.run(debug=True)
