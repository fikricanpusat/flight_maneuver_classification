import tensorflow as tf
import pickle

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('flight_data.pkl', 'rb') as f:
    flight_data = pickle.load(f)
with open('validation_data.pkl', 'rb') as f:
    validation_data = pickle.load(f)
with open('validation_flight_data.pkl', 'rb') as f:
    validation_flight_data = pickle.load(f)

no_of_epochs = 10

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_data.values, flight_data.values, epochs=no_of_epochs, validation_split = 0.25)
model.save('shl_maneuver_classification.model')

score = model.evaluate(validation_data, validation_flight_data, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])