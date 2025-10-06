# Traffic Sign Recognition

This project implements a **Convolutional Neural Network (CNN)** to classify images of traffic signs from the **GTSRB dataset**. The AI can recognize 43 different types of traffic signs based on images and predict their category with high accuracy.

## Overview

The project contains the following main components:

1. **traffic.py**:  
   - **load_data(data_dir)**: Loads images and labels from the dataset.  
     - Reads images from subdirectories (0–42), where each folder represents a traffic sign category.  
     - Resizes images to `(IMG_WIDTH, IMG_HEIGHT)` for neural network input.  
     - Returns a tuple `(images, labels)` as lists of numpy arrays and integers.  
   - **get_model()**: Returns a compiled CNN model.  
     - Input shape: `(IMG_WIDTH, IMG_HEIGHT, 3)`  
     - Output: `NUM_CATEGORIES` units with softmax activation  
     - Includes convolutional layers, pooling layers, dense layers, and optional dropout.  
   - **main function**:  
     - Loads and preprocesses the dataset  
     - Splits the data into training and testing sets  
     - Compiles and trains the CNN  
     - Evaluates performance on the testing set  
     - Optionally saves the trained model to disk  

## Features

- Image preprocessing using OpenCV (`cv2`)  
- Flexible CNN architecture for experimentation  
- Supports saving and loading trained models for reuse  
- Platform-independent data loading using `os.path.join`  

## File structure
```text
traffic-sign-recognition/
│
├── gtsrb/                     # Dataset sample
│   ├── 0/
├── traffic.py                 # Main program for training/evaluating the model
├── README.md                  # Project documentation
├── requirements.txt           # Required Python packages
├── LICENSE                    # License file (MIT License)
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition
```
2. (Optional) Create a virtual environment:
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required packages:
```python
pip install -r requirements.txt
```
Requirements (`requirements.txt`):
```python
numpy
opencv-python
tensorflow
scikit-learn
```
Ensure you have Python 3.8+ installed.

## Usage
1. Train the model and evaluate:
```python
python traffic.py gtsrb
```
2. Save the trained model:
```python
python traffic.py gtsrb model.h5
```
3. Load and use the saved model:
```python
from tensorflow.keras.models import load_model
model = load_model("model.h5")
# Use model.predict() on new images
```
## Experimentation

During development, multiple CNN architectures were tested with different combinations of:
- Number of convolutional layers
- Filter sizes and counts
- Pooling layers and pool sizes
- Dense layers and dropout rates

Observations:
- Adding multiple convolutional layers with small filters improved feature extraction.
- Dropout helped reduce overfitting.
- Pooling layers reduced computation but too much pooling caused loss of spatial information.
- Normalizing input images significantly improved convergence.
Results showed that a CNN with 3 convolutional layers, max pooling, and 2 dense layers achieved the best accuracy on the testing set.

## License
This project is licensed under the MIT License. See the LICENSE
