import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('flight_data.pkl', 'rb') as f:
    flight_data = pickle.load(f)

no_of_epochs = 5
nodes_in_layer = [8, 16, 32, 64, 128, 256]

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i in nodes_in_layer:
    print(i)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(i, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_data.values, flight_data.values, epochs=no_of_epochs, validation_split = 0.25)
    ax1.plot(hist.history['accuracy'])
    ax1.legend(nodes_in_layer, loc="lower right")
    ax1.grid(True)
    ax1.set_ylim([0.5, 1.1])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.plot(hist.history['loss'])
    ax2.legend(nodes_in_layer, loc="upper right")
    ax2.grid(True)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('SCC Loss')
    ax2.set_ylim([0, 1.4])

plt.show()