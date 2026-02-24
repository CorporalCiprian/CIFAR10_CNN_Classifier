# CIFAR-10 Image Classification with CNN

This project demonstrates a Convolutional Neural Network (CNN) built using TensorFlow and Keras to classify images from the CIFAR-10 dataset into 10 distinct categories.

## Project Structure

* `train.py`: The script that builds, compiles, and trains the CNN model. It saves the trained model locally.
* `demo.py`: An interactive script that loads the locally saved model and makes visual predictions with confidence scores.
* `requirements.txt`: The required Python libraries to run the scripts.

## Setup and Execution

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training the model

2. Train the model to generate the weights file:
```bash
python train.py
```

## Running the demo

3. Run the interactive prediction demo:
```bash
python demo.py
```
