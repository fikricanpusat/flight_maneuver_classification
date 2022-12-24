import tensorflow as tf
import pickle
import numpy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

with open('validation_data.pkl', 'rb') as f:
    validation_data = pickle.load(f)
with open('validation_flight_data.pkl', 'rb') as f:
    validation_flight_data = pickle.load(f)
with open('validation_rnn.pkl', 'rb') as f:
    validation_rnn = pickle.load(f)
with open('validation_flight_rnn.pkl', 'rb') as f:
    validation_flight_rnn = pickle.load(f)
    
stats_list = list()

def show_bar_plot(size_x, size_y, classes, lists, names):
    plt.figure(figsize=(size_x, size_y))
    plt.clf()
    width = 0.25
    ind = numpy.arange(len(classes))
    for j in range(len(lists)):
        plt.bar(ind + j * width, lists[j] , width, label=names[j])
        for i in range(len(lists[j])):
            plt.text(i + width * j, lists[j][i], str(round(lists[j][i], 1)) + "%", ha = 'center', va = 'top', size=10, rotation = 90)
    plt.xticks(ind + 2*width / 2, classes)
    plt.ylabel('[%]')
    plt.legend()
    plt.show()

def evaluate(model, title, validation_x, validation_y):
    global stats_list
    classifications = ["Level Flight", "Sustained Turn","Ascend","Descend","Reverse",\
        "Immelman","Aileron Roll","Split-S","Chandelle","Lazy-8"]
    pred = model.predict(validation_x)
    clas = list()
    count = 0
    for i in range(len(pred)):
        clas.append(numpy.argmax(pred[i]))
        if clas[-1] != validation_flight_data.values[i]:
            count = count + 1
    confusion_mtx = confusion_matrix(validation_y.values, clas)

    plt.figure(figsize = (7, 7))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    tick_marks = numpy.arange(10)
    plt.xticks(tick_marks, classifications, rotation=90)
    plt.yticks(tick_marks, classifications)
    plt.title(title)
    print(confusion_mtx)
    thresh = confusion_mtx.max() / 2.
    for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
        plt.text(j, i, confusion_mtx[i, j],
            horizontalalignment="center",
            color="white" if confusion_mtx[i, j] > thresh else "black")
    plt.ylabel('True Maneuver')
    plt.xlabel('Predicted Maneuver')

    acc_list = list()
    pre_list = list()
    rec_list = list()

    for i in range(len(classifications)):
        TP = confusion_mtx[i][i]
        FP = sum([column[i] for column in confusion_mtx]) - confusion_mtx[i][i]
        FN = sum(confusion_mtx[i]) - confusion_mtx[i][i]
        TN = sum(sum(confusion_mtx)) - sum(confusion_mtx[i]) -\
            sum([column[i] for column in confusion_mtx]) + confusion_mtx[i][i]
        acc_list.append(100 * ((TP + TN) / (TP + TN + FP + FN)))
        pre_list.append(100 * (TP / (TP + FP)))
        rec_list.append(100 * (TP / (TP + FN)))
        print(classifications[i])
        print("accuracy = " + str(acc_list[-1]))
        print("precision = " + str(pre_list[-1]))
        print("recall = " + str(rec_list[-1]))
        print("\n")

    lst = list()
    lst.extend([acc_list, pre_list, rec_list])
    types=list()
    types.extend(["Accuracy", "Precision", "Recall"])

    show_bar_plot(15, 3, classifications, lst, types)

    stats_list.append(sum(acc_list) / len(acc_list))
    stats_list.append(sum(pre_list) / len(pre_list))
    stats_list.append(sum(rec_list) / len(rec_list))
    score = model.evaluate(validation_x, validation_y, verbose=0)
    stats_list.append(100 * score[1])
    print('\n\nValidation loss:', score[0])
    print('Validation accuracy:', score[1])
    print("\n\n")

shl = tf.keras.models.load_model('shl_maneuver_classification.model')
dnn = tf.keras.models.load_model('dnn_maneuver_classification.model')
rnn = tf.keras.models.load_model('rcn_maneuver_classification.model')

print("Single Hidden Layer:")
evaluate(shl, "Single Hidden Layer", validation_data, validation_flight_data)

print("Deep Neural Network:")
evaluate(dnn, "Deep Neural Network", validation_data, validation_flight_data)

print("Recurrent Neural Network:")
evaluate(rnn, "Recurrent Neural Network", validation_rnn, validation_flight_rnn)

lst = list()
lst.extend([stats_list[0:4], stats_list[4:8], stats_list[8:12]])
types=list()
types.extend(["SHLNN", "DNN", "RNN"])
stat_names=list()
stat_names.extend(["Maneuver Accuracy", "Precision", "Recall", "Total Accuracy"])

show_bar_plot(10, 5, stat_names, lst, types)
