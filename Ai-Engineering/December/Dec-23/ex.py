import tensorflow as tf
from tensorflow import keras

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

# Model WITHOUT batch normalization
model_no_bn = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Model WITH batch normalization
model_with_bn = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    # keras.layers.BatchNormalization(),  # After each dense layer
    keras.layers.Dense(128, activation='relu'),
    # keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

# Compile both models
for model in [model_no_bn, model_with_bn]:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train and compare!
history_no_bn = model_no_bn.fit(x_train, y_train, epochs=20, validation_split=0.2)
history_with_bn = model_with_bn.fit(x_train, y_train, epochs=20, validation_split=0.2)

# end of file