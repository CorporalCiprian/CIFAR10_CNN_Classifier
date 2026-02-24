import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_test():
    if not os.path.exists('cifar10_model.keras'):
        print("ERROR: Cannot find 'cifar10_model.keras'. Run the training script first!")
        return

    print("Loading the trained model...")
    model = tf.keras.models.load_model('cifar10_model.keras')

    print("Downloading test data...")
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_images = test_images / 255.0

    while True:
        random_index = random.randint(0, len(test_images) - 1)
        img = test_images[random_index]
        real_label = class_names[test_labels[random_index][0]]

        img_batch = np.expand_dims(img, axis=0)
        predictions = model.predict(img_batch, verbose=0)

        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        print(f"------------------------------------------------")
        print(f"Image index:  {random_index}")
        print(f"True label:   {real_label}")
        print(f"AI predicts:  {predicted_class} ({confidence:.2f}% confidence)")

        plt.figure(figsize=(4, 4))
        plt.imshow(img)

        color = 'green' if predicted_class == real_label else 'red'

        plt.title(f"AI: {predicted_class} ({confidence:.1f}%)\nReal: {real_label}", color=color)
        plt.axis('off')
        plt.show()

        answer = input("Do you want to try another image? (yes/no): ")
        if answer.lower() not in ['y', 'yes']:
            break

if __name__ == "__main__":
    load_and_test()