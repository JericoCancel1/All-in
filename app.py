from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

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
    processed_img = img.convert("RGB")  # Ensure image is in RGB mode
    inverted_img = Image.eval(processed_img, lambda x: 255 - x)
    return inverted_img

if __name__ == "__main__":
    app.run(debug=True)
