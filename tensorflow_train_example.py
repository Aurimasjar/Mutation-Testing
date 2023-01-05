# import pandas as pd
# from keras import models
# from tensorflow.keras import layers

# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# import mlrun.frameworks.tf_keras as mlrun_tf_keras
# import mlrun
#
#
# def train(
#     data, label_column = 'pazeidimai', epochs = 10,
#     mlrun_wrapper = lambda model, x_test, y_test: None
# ):
#     x = data.drop(label_column, axis = 1)
#     y = data[label_column]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, stratify=y)
#
#     model = models.Sequential()
#     model.add(layers.Dense(128, input_dim=x_train.shape[1], activation='relu'))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(32, activation='relu'))
#     model.add(layers.Dense(16, activation='relu'))
#     model.add(layers.Dense(8, activation='relu'))
#     model.add(layers.Dense(4, activation='relu'))
#     model.add(layers.Dense(2, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     mlrun_wrapper(model, x_test, y_test)
#
#     model.fit(
#         x_train, y_train,
#         class_weight=(1 - pd.Series(y_train).value_counts() / len(y_train)).to_dict(),
#         epochs=epochs
#     )
#
#     return model
#
#
# def set_artifacts(context, label_column, model_name, model, x_test, y_test):
#     x_test[label_column] = y_test
#     context.log_dataset('test_set', df = x_test, index = False, format = 'csv')
#     mlrun_tf_keras.apply_mlrun(model = model, model_name = model_name, context = context)
#
#
# def train_remote(context: mlrun.MLClientCtx, dataset: mlrun.DataItem, model_name, epochs = 10, label_column = 'pazeidimai'):
#     train(
#         dataset.as_df(), label_column, epochs,
#         lambda model, x_test, y_test: set_artifacts(context, label_column, model_name, model, x_test, y_test),
#     )