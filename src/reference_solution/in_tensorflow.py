import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from algorytmy.reference_solution.config import *
from algorytmy.reference_solution.mnist_dataset import train_images, train_labels, test_images, test_labels
import time

message = "Running on GPU" if tf.test.gpu_device_name() else "Running on CPU"
print(message)

model = Sequential(
    [
        Conv2D(
            num_filters_1, kernel_size, activation=activation_1, input_shape=input_shape
        ),
        Flatten(),
        Dense(num_units, activation=activation_1)
    ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.summary()

start = time.time()
history = model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(test_images, test_labels),
)
end = time.time()
print("Time: ", end - start)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
