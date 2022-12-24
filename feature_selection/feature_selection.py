import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pickle

train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)
train_data = train_data.iloc[np.random.permutation(len(train_data))]
flight_data = train_data.pop('flight')
for (column, columnData) in train_data.iteritems():
    train_data[column] = train_data[column] / train_data[column].abs().max()

validation_file_path = "validation.csv"
validation_data = pd.read_csv(validation_file_path)
validation_data = validation_data.iloc[np.random.permutation(len(validation_data))]
validation_flight_data = validation_data.pop('flight')
for (column, columnData) in validation_data.iteritems():
    validation_data[column] = validation_data[column] / validation_data[column].abs().max()

cor_matrix_pearson = train_data.corr(method='pearson').abs()
cor_matrix_kendall = train_data.corr(method='kendall').abs()
cor_matrix_spearman = train_data.corr(method='spearman').abs()
plt.figure(figsize = (15, 5))
plt.subplot(1, 3, 1)
plt.title("Pearson Correlation Matrix")
plt.imshow(cor_matrix_pearson)
plt.subplot(1, 3, 2)
plt.title("Kendall Correlation Matrix")
plt.imshow(cor_matrix_kendall)
plt.subplot(1, 3, 3)
plt.title("Spearman Correlation Matrix")
plt.imshow(cor_matrix_spearman)
plt.show()

print("\n\nHighly Correlated Features\n");
for i in range(len(cor_matrix_pearson)):
    for j in range(i):
        if cor_matrix_pearson.iloc[i, j] > 0.75 and cor_matrix_kendall.iloc[i, j] > 0.75\
            and cor_matrix_spearman.iloc[i, j] > 0.75 and i != j:
            print(str(i) + " - " + train_data.columns[i] + " & " + str(j) + " - "\
                + train_data.columns[j] + " & " + str("%.3f" % cor_matrix_pearson.iloc[i, j]) + " & "\
                    + " & " + str("%.3f" % cor_matrix_kendall.iloc[i, j])\
                        + " & " + str("%.3f" % cor_matrix_spearman.iloc[i, j]) + "\\\\")

X = train_data.values
y = flight_data.values

selector = SelectKBest(score_func=f_classif, k='all').fit(X,y)
x_new = selector.transform(X)
scores = selector.scores_

plt.figure(figsize = (10, 5))
plt.xlabel("Flight Feature", fontsize=20)
plt.ylabel("ANOVA Score", fontsize=20)
plt.xticks(rotation=90)
plt.bar(train_data.columns.values, scores / sum(scores), color ='maroon', width = 0.75)

print("\n\nANOVA Scores Sorted")
anova_scores = dict()
for i in range(len(scores)):
    anova_scores[train_data.columns.values[i]] = scores[i]
sorted_scores = dict(sorted(anova_scores.items(), key=lambda item:item[1], reverse=True))
print(sorted_scores)
print("\n\n");

extract = list()
for i in range(len(cor_matrix_pearson)):
    for j in range(i):
        if cor_matrix_pearson.iloc[i, j] > 0.75 and cor_matrix_kendall.iloc[i, j] > 0.75\
            and cor_matrix_spearman.iloc[i, j] > 0.75 and i != j:
                
                if anova_scores[train_data.columns[i]] > anova_scores[train_data.columns[j]]:
                    extract.append(train_data.columns[j])
                else:
                    extract.append(train_data.columns[i])

extract = [*set(extract)]

for i in range(len(extract)):
    train_data.pop(extract[i])
    validation_data.pop(extract[i])

number_of_features = 3
lst = list(range(0, len(train_data.columns)))
els = [list(x) for x in it.combinations(lst, number_of_features)]

ann_acc_scores = dict()
ann_lss_scores = dict()

for i in range(len(train_data.columns.values)):
    ann_acc_scores[train_data.columns.values[i]] = 0
    ann_lss_scores[train_data.columns.values[i]] = 0

for i in range(len(els)):
    print("Complete: " + str(i) + " / " + str(len(els) - 1))
    temp = train_data.iloc[:, els[i]]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(temp.values, flight_data.values, epochs=2, validation_split=0.2)
    for j in range(number_of_features):
        print(train_data.columns.values[els[i][j]])
        ann_acc_scores[train_data.columns.values[els[i][j]]] += hist.history.get('accuracy')[-1]
        ann_lss_scores[train_data.columns.values[els[i][j]]] += hist.history.get('loss')[-1]

count = 0
for i in range(len(els)):
    for j in range(number_of_features):
        if els[i][j] == 0:
            count = count + 1

plt.figure(figsize = (10, 5))
plt.xlabel("Flight Feature", fontsize=20)
plt.ylabel("ANN Average Accuracy", fontsize=20)
plt.xticks(rotation=90)
ann_scores_list = [x / count for x in list(ann_acc_scores.values())]
plt.ylim([min(ann_acc_scores) - 0.1, 1])
plt.bar(range(len(ann_acc_scores)), ann_scores_list, align='center')
plt.xticks(range(len(ann_acc_scores)), list(ann_acc_scores.keys()))

plt.figure(figsize = (10, 5))
plt.xlabel("Flight Feature", fontsize=20)
plt.ylabel("ANN Avarage Loss", fontsize=20)
plt.xticks(rotation=90)
ann_loss_list = [x / count for x in list(ann_lss_scores.values())]
plt.bar(range(len(ann_lss_scores)), ann_loss_list, align='center')
plt.xticks(range(len(ann_lss_scores)), list(ann_lss_scores.keys()))

print("\n\ANN Avg Acc Scores Sorted")
sorted_avg_acc = dict(sorted(ann_acc_scores.items(), key=lambda item:item[1], reverse=True))
print(sorted_avg_acc)
print("\n\n")

print("\n\ANN Avg Loss Scores Sorted")
sorted_avg_lss = dict(sorted(ann_lss_scores.items(), key=lambda item:item[1], reverse=True))
print(sorted_avg_lss)
print("\n\n")

with open('extract.pkl', 'wb') as f:
    pickle.dump(extract, f)

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    
with open('flight_data.pkl', 'wb') as f:
    pickle.dump(flight_data, f)

with open('validation_data.pkl', 'wb') as f:
    pickle.dump(validation_data, f)

with open('validation_flight_data.pkl', 'wb') as f:
    pickle.dump(validation_flight_data, f)
