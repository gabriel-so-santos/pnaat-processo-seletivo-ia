import numpy
from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1] to improve training stability and convergence
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape input data to include a channel dimension (required for Conv2D layers)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# No class balancing is required, as the MNIST dataset is already evenly distributed across classes

model = Sequential([
    Input(shape=(28,28,1)),

    layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(filters=24, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=10, activation='softmax')

    # Dropout was evaluated but did not provide a meaningful improvement for this task
])

model.summary()

# Compile the model using Adam optimizer and sparse categorical crossentropy (no one-hot encoding required)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 4 epochs, one more doesn't provide significant improvements
model.fit(x_train, y_train, epochs=4, batch_size=32, validation_split=0.10, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print(f"\naccuracy: {acc*100:.4f}%\nloss {loss*100:.4f}%")

model.save('model.h5')