import tensorflow as tf
import pickle

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('flight_data.pkl', 'rb') as f:
    flight_data = pickle.load(f)

no_of_epochs = 5

optimizers = ['SGD','AdaGrad','RMSprop','Adadelta','Adam','AdaMax','NAdam','FTRL']
activations = [tf.nn.relu, tf.nn.sigmoid, tf.nn.softmax, tf.nn.softplus, tf.nn.softsign,\
              tf.nn.tanh, tf.nn.selu, tf.nn.elu]
losses = ['categorical_crossentropy', 'sparse_categorical_crossentropy', 'kl_divergence',\
          'mean_squared_error','mean_absolute_error','cosine_similarity','huber_loss','categorical_hinge',]

max_acc = 0

for i in range(len(optimizers)):
    for j in range(len(activations)):
        for k in range(len(losses)):
            print(i, ", ", j, ", ", k)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, activation=activations[j]))
            model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
            try:
                model.compile(optimizer=optimizers[i], loss=losses[k], metrics=['accuracy'])
                hist = model.fit(train_data.values, flight_data.values, epochs=no_of_epochs, validation_split = 0.25)
            except:
                print("Problem")
                continue
            if(hist.history.get('accuracy')[-1] > max_acc):
                max_acc = hist.history.get('accuracy')[-1]
                max_acc_opt = optimizers[i]
                max_acc_act = activations[j]
                max_acc_lss = losses[k]
                print(max_acc)

print("Max Accuracy: ", max_acc, " Optimizer: ", max_acc_opt, " Activation: ",\
    max_acc_act, " Loss: ", max_acc_act)