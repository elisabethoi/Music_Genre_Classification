import os
from time import sleep
from dataclasses import replace
import keras
from keras.models import Sequential
#import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import splitfolders
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from tensorflow.keras.optimizers import Adam

#############################################################################################
############ DNN (dense neural network)

# df1 = pd.read_csv(
#     'C:/Users/elio9/OneDrive/Web/DAT/NN2/music-genre-recognition/Data/features_3_sec.csv')
# # print(df1.head())

# df2 = pd.read_csv(
#     'C:/Users/elio9/OneDrive/Web/DAT/NN2/music-genre-recognition/Data/features_30_sec.csv')
# # print(df2.head())

# # print(df1.shape)
# # print(df2.shape)


# # print(df1.dtypes)
# # print(df2.dtypes)

# df1 = df1.drop(labels='filename', axis=1)

# class_list = df1.iloc[:, -1]
# print(class_list)

# # Feature Extraction
# # LabelEncoder() -> to convert categorical text data into numerical data
# convertor = preprocessing.LabelEncoder()
# y = convertor.fit_transform(class_list)
# print(y)

# print(df1.iloc[:, :-1])

# # Scaling the Features
# scaler = StandardScaler()
# X = scaler.fit_transform(np.array(df1.iloc[:, :-1], dtype=float))
# print(type(X))
# print(X)
# # Training and Testing Set
# #X_train, X_val, y_train, y_test = train_test_split(X, y, test_size= 0.33)


# # Takes % of data for a test ds
# test_ds_ratio = 0.05
# x_test = []
# y_test = []

# for _ in range(int(len(X)*test_ds_ratio)):
#     idx_test = np.random.randint(0, len(X))
#     x_test.append(X[idx_test])
#     y_test.append(y[idx_test])
#     X = np.delete(X, idx_test, 0)
    
# print(len(y_test))
# print(len(x_test))

# test_data = {
#         "x_test": x_test,
#         "y_test": y_test
# }


# #x_test = X[np.random.choice(X,len(X)*0.05, replace=False),:]

# # Building the model
# def trainModel(model, ds, epochs, optimizer):
#     x_train = ds.get("x_train")
#     x_val = ds.get("x_val")
#     y_train = ds.get("y_train")
#     y_val = ds.get("y_val")

#     batch_size = 128
#     model.compile(optimizer=optimizer,
#                   loss='sparse_categorical_crossentropy',
#                   metrics='accuracy')
#     return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
#                      batch_size=batch_size)


# # Training and Testing Set with k-fold cross validation
# kfold = KFold(n_splits=10)

# # dense layer -> neuronal network layer
# # started with 512 neurons
# # dropout layer = 20% -> one in 5 inputs will be randomly excluded (ausgeschlossen) from each update cycle


# model = Sequential([
#     keras.layers.Dense(512, activation="relu"),
#     # Dropout layer prevents overfitting
#     keras.layers.Dropout(0.2),

#     keras.layers.Dense(256, activation="relu"),
#     keras.layers.Dropout(0.2),

#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dropout(0.2),

#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dropout(0.2),

#     # softmax -> Output mit dem höchsten Wert
#     keras.layers.Dense(10, activation="softmax"),
# ])




# for train_index, val_index in kfold.split(X):
#     print("TRAIN:", train_index, "VALIDATION:", val_index)
#     x_train, x_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]
    
#     print(len(x_train))
#     print(len(y_train))

#     data = {
#         "x_train": x_train,
#         "x_val": x_val,
#         "y_train": y_train,
#         "y_val": y_val
#     }
#     # adam gave us the best results, so we use the adam optimizer
#     model_history = trainModel(
#         model=model, ds=data, epochs=200, optimizer="adam")


# print(model.summary())




# def plotValidate(history):
#     print("Validation Accuracy", max(history.history["val_accuracy"]))
#     pd.DataFrame(history.history).plot(figsize=(12, 6))
#     plt.show()


# plotValidate(model_history)

# x_test = test_data.get("x_test")
# y_test = test_data.get("y_test")

# test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=128)
# print("Test Loss: ", test_loss)
# test_acc = test_acc*100
# print("Best Test Accuracy: ", test_acc)






# plotValidate(model_history)

###########################################################################
################### CNN


splitfolders.ratio('Data/images_original', output="datasets", seed=1337, ratio=(.8, 0.1, 0.1)) 

train_dir = "datasets/train/"
train_datagenerator = ImageDataGenerator(rescale=1./255)
train_generator = train_datagenerator.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

val_dir = "datasets/val/"
val_datagenerator = ImageDataGenerator(rescale=1./255)
val_generator = val_datagenerator.flow_from_directory(val_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)


# 5 Convolutional layers
# a Dropout layer to avoid over-fitting 
# Dense layer with Softmax activation

def GenreModel(input_shape=(288,432,4), classes=10):  
    # sourcery skip: inline-immediately-returned-variable
    X_input = Input(input_shape)

    X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(16,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(32,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Flatten()(X)
    

    X = Dense(X.shape[1], activation='relu', name='fc1')(X)
    X = Dropout(0.1)(X)
    X = Dense(256, activation='relu', name='fc3')(X)
    X = Dropout(0.1)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs=X_input,outputs=X,name='GenreModel')
    return model



model = GenreModel()
opt = Adam(learning_rate=0.001)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy']) 

model.fit_generator(train_generator, epochs=250, validation_data=val_generator)


print(model.summary())


test_dir = "datasets/test/"
test_datagenerator = ImageDataGenerator(rescale=1./255)
test_generator = test_datagenerator.flow_from_directory(test_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)


model.predict_generator(test_generator)















