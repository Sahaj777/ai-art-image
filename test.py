from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

app = Flask(__name__)

# Load the pre-trained style transfer model
style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Define a route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the image generation
@app.route('/generate', methods=['POST'])
def generate():
    # Get the content and style images from the request
    content_image = request.files['content']
    style_image = request.files['style']

    # Load and preprocess the content and style images
    content_image = Image.open(content_image)
    style_image = Image.open(style_image)

    content_image = np.array(content_image.resize((256, 256))) / 255.0
    style_image = np.array(style_image.resize((256, 256))) / 255.0

    # Perform style transfer to generate the art image
    stylized_image = style_model(tf.convert_to_tensor([content_image]), tf.convert_to_tensor([style_image]))[0]
    stylized_image = (np.array(stylized_image) * 255).astype(np.uint8)
    stylized_image = Image.fromarray(stylized_image)

    # Save the generated art image
    output_path = 'static/generated.jpg'
    stylized_image.save(output_path)

    return redirect('/result')

# Define a route to display the generated art image
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    # Set the port dynamically with a default value of 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
