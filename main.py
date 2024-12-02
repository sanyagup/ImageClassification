import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt


# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize some images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(label_names[y_train[i][0]])
    plt.axis('off')
plt.show()

