import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('flight_data.pkl', 'rb') as f:
    flight_data = pickle.load(f)

no_of_epochs = 5

features_sorted = ['    __VVI-__fpm', '_elev-stick', '_lift-___lb', '_roll-__deg',\
                   'Gload-axial', '    ____Q-rad/s', 'thro1-_part', '_slip-__deg',\
                   '    ailrn-_surf', '    _drag-___lb', '____R-rad/s', '____P-rad/s',\
                   'ruddr-stick', '_Vind-_kias', '    _beta-__deg', 'hpath-__deg',\
                   '____M-lb-ft', 'hding-__mag', 'ruddr-_surf', '____N-_ftlb',\
                   '____L-lb-ft', '____M-_ftlb']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
acc = list()
lss = list()

for i in range(len(features_sorted)):
    print(i)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_data.values, flight_data.values, epochs=no_of_epochs, validation_split = 0.25)
    acc.append(hist.history.get('accuracy')[-1])
    lss.append(hist.history.get('loss')[-1])
    print(hist.history.get('accuracy')[-1])
    print("Removed: " + features_sorted[-1])
    train_data.pop(features_sorted[-1])
    features_sorted.pop()

ax1.plot(acc)
ax1.set_xlabel('Number of Removed Features')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax2.plot(lss)
ax2.set_xlabel('Number of Removed Features')
ax2.set_ylabel('SCC Loss')
ax2.grid(True)
plt.show()