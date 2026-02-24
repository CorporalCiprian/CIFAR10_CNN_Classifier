import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def train_model():
    print("Loading and preprocessing data...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(f"Loaded {len(train_images)} training images.")

    print("\nBuilding the CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("\nStarting training...")
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    print("\nSaving results...")
    model.save('cifar10_model.keras')
    print("Model successfully saved to 'cifar10_model.keras'")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Model Performance')
    plt.savefig('training_history.png')
    print("Plot successfully saved to 'training_history.png'")

if __name__ == "__main__":
    train_model()