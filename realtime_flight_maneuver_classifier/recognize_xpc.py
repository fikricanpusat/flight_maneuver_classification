import socket
import struct
import numpy
import pickle
import pandas as pd
import tensorflow as tf
import xpc

UDP_IP = ""
UDP_PORT = 49005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

model = tf.keras.models.load_model('dnn_maneuver_classification.model')

train_file_path = "train.csv"
train_data = pd.read_csv(train_file_path)
with open('exclude.pkl', 'rb') as f:
    exclude = pickle.load(f)
train_data.pop("flight")

classifications = ["Level Flight", "Sustained Turn","Ascend","Descend","Reverse",\
    "Immelman","Aileron Roll","Split-S","Chandelle","Lazy-8"]

try:
    while True:
            data, addr = sock.recvfrom(365)
            floatdata = struct.unpack('=bbbbbiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffff', data)

            data_sample = numpy.zeros((2, 33))
            data_sample[0] = str(floatdata[6]) , \
                str(floatdata[17]) , str(floatdata[19]) , str(floatdata[20]) , str(floatdata[21]) , \
                str(floatdata[24]) , str(floatdata[25]) , str(floatdata[26]) , \
                str(floatdata[33]) , str(floatdata[34]) , str(floatdata[35]) , \
                str(floatdata[42]) , str(floatdata[43]) , str(floatdata[44]) , \
                str(floatdata[51]) , str(floatdata[52]) , str(floatdata[53]) , \
                str(floatdata[60]) , str(floatdata[61]) , str(floatdata[62]) , str(floatdata[63]) , \
                str(floatdata[69]) , str(floatdata[70]) , str(floatdata[71]) , str(floatdata[72]) , str(floatdata[76]) , \
                str(floatdata[78]) ,\
                str(floatdata[87]) , str(floatdata[88]) , str(floatdata[89]) , str(floatdata[90]) , str(floatdata[91]) , str(floatdata[92])

            for (column, columnData) in train_data.iteritems():
                data_sample[0][train_data.columns.get_loc(column)] = data_sample[0][train_data.columns.get_loc(column)] / train_data[column].abs().max()

            exc_list = list()
            for i in range(len(exclude)):
                exc_list.append(train_data.columns.get_loc(exclude[i]))
            data_sample = numpy.delete(data_sample, exc_list, 1)

            prediction = model.predict(data_sample)
                      
            print("\nComputer Says:")
            print("Level Flight: ", prediction[0][0], "Turning: ", prediction[0][1], "Ascent: ", prediction[0][2], "Descent: ", prediction[0][3], "Reverse: ", prediction[0][4], "Immelman: ", prediction[0][5])

            if numpy.amax(prediction[0]) < 0.80:
                print("   - Unidentified.")
                xpc_text = "Unidentified"
            else:
                print("   - ", classifications[numpy.argmax(prediction[0])])
                xpc_text = classifications[numpy.argmax(prediction[0])]

            with xpc.XPlaneConnect(xpHost = '127.0.0.1') as client:
                client.sendTEXT("ANN Says: " + xpc_text, 200, 570)
            
except KeyboardInterrupt:
    print("Ended!")
    sock.close()