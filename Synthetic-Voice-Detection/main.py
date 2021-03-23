from keras.models import load_model
from preprocess import *
import os
model_path = "models/swishnetwide-91.8199%.h5"

model = load_model(model_path)


def threshold(result_array):
    result = 0 #0 means fake, 1 means real
    if result_array[0][0] > result_array[0][1]:
        return 0
    else:
        return 1

test_file_directory = "test_file/"
for filename in os.listdir(test_file_directory):
    features = extract_feature(test_file_directory+filename)
    # if features.shape(1) > 87:
    #     features[1] = features[1][:87]
    # print(features.shape)
    predict_array = model.predict(features)
    result = threshold(predict_array)
    if result == 0:
        print(filename, "is fake")
    else:
        print(filename, "is real")