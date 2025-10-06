import tensorflow as tf
import cv2
import numpy as np
import os
import sys

# Image dimensions (must match training)
IMG_WIDTH = 30
IMG_HEIGHT = 30

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python predict.py model.h5 images_folder")

    model_path = sys.argv[1]
    images_folder = sys.argv[2]

    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()

    # Loop through images in folder
    for filename in os.listdir(images_folder):
        img_path = os.path.join(images_folder, filename)
        
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {filename}")
            continue

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=0)  # add batch dimension

        # Predict
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)

        print(f"{filename} -> Predicted category: {predicted_class}")

if __name__ == "__main__":
    main()

