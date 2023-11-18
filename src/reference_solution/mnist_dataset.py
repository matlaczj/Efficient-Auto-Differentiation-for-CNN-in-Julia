from tensorflow.keras.datasets import mnist
from algorytmy.reference_solution.config import input_shape

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (
    train_images.reshape(train_images.shape[0], *input_shape).astype("float32") / 255.0
)
test_images = (
    test_images.reshape(test_images.shape[0], *input_shape).astype("float32") / 255.0
)
# transform target to binary classification problem (5 or not 5)
train_labels = (train_labels == 5).astype("int32")
test_labels = (test_labels == 5).astype("int32")

# select first 100 samples
train_images = train_images[:100]
train_labels = train_labels[:100]
test_images = test_images[:100]
test_labels = test_labels[:100]