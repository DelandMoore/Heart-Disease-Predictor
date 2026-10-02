from langchain_core.tools import tool
from tensorflow import keras
from PIL import Image
import numpy as np 
import io

# Load the trained model 
model = keras.models.load_model('best_pneumonia_model.keras')

@tool
def predict_pneumonia(image_path:str) -> str:
    """
    Analyze a chest X-ray image to detect signs of pneumonia 
    only call this tool when a user has uploaded an image 
    return the probability and a plain language label
    """

    # Load uploaded image directly
    image = Image.open(image_path)

    # Preprocessing the image same as how it was preprocessed during training
    image = image.convert('L')
    image = image.resize((28,28))

    # Converting the image to an array
    image = np.array(image)

    # Adding channel dimension and batch dimension
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # Getting prediction and probability
    prediction = model.predict(image, verbose=0)

    probability = float(prediction[0][0])

    # Printing the class 
    if probability >= 0.1:
        print('Prediction: Pneumonia')
    else:
        print('Prediction: Normal')

    # Pneumonia prbaility
    print(f'Pneumonia probaility: {probability:.2f}%')

    # Normal Probability
    print(f'Normal probability: {(1 - probability) * 100:.2f}%')


