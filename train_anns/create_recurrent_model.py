import pandas as pd
from collections import deque
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

SEQ_LEN = 30
EPOCHS = 3

def preprocess_df(df):
    flight_data = df.pop('flight').values
    for (column, columnData) in df.iteritems():
        df[column] = df[column] /df[column].abs().max()
    df['flight'] = flight_data
    sequential_data = []
    prev = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev.append([n for n in i[:-1]])
        if len(prev) == SEQ_LEN:
            sequential_data.append([np.array(prev), i[-1]])
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)
validation_file_path = "validation.csv"
validation_data = pd.read_csv(validation_file_path)

with open('exclude.pkl', 'rb') as f:
    exclude = pickle.load(f)

for i in range(len(exclude)):
    train_data.pop(exclude[i])
    validation_data.pop(exclude[i])

times = sorted(train_data.index.values)
times_validation = sorted(validation_data.index.values)

train_x, train_y = preprocess_df(train_data)
validation_x, validation_y = preprocess_df(validation_data)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)
with open('validation_rnn.pkl', 'wb') as f:
    pickle.dump(validation_x, f)
with open('validation_flight_rnn.pkl', 'wb') as f:
    pickle.dump(pd.Series(validation_y), f)

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=EPOCHS, validation_data=(validation_x, validation_y))
model.save('rc_maneuver_classification.model')

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
