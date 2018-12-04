import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

###############################################################################

def preprocess(vocab_len, random_state, test_size):
    # Get directory of this file
    path = os.path.dirname(os.path.abspath("dem_v_rep_classifier.py"))

    # Load dataset
    data = pd.read_csv(path+"/data/raw/ExtractedTweets.csv")

    # Inspect format of data
    print(data[:5])

    # Ensure Democrats and Republicans are equally represented in the data
    print("The number of tweets: ", len(data.Party))
    print("The number of tweets from Democrats: ", list(data.Party).count("Democrat"))
    print("The number of tweets from Republicans: ", list(data.Party).count("Republican"))

    # Assign my input and labels to variables for preprocessing
    labels = data["Party"]
    tweets = data["Tweet"]

    # Create a tokenizer for my text
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_len, lower=True)
    tokenizer.fit_on_texts(tweets)
    # Assign each unique token a value and turn the tweets into a sequence of these values
    X = tokenizer.texts_to_sequences(tweets)
    # Democrat=0; Republican=1
    Y = LabelEncoder().fit_transform(labels)

    # Section off my testing set from my training set
    train_data, test_data, train_labels, test_labels = train_test_split(X,
                                                                        Y,
                                                                        random_state=random_state,
                                                                        test_size=test_size)

    # Find the tweet with the most words and set that at max_len for padding
    max_len = len(max(X, key=len))

    # Pad data so each list is the same length
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding="post",
                                                            maxlen=max_len)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                            padding="post",
                                                            maxlen=max_len)


    np.save(path+"/data/processed/train_data.npy", train_data)
    np.save(path+"/data/processed/test_data.npy", test_data)
    np.save(path+"/data/processed/train_labels.npy", train_labels)
    np.save(path+"/data/processed/test_labels.npy", test_labels)

    return train_data, test_data, train_labels, test_labels

def std_model(vocab_len, L1_size, dropout, epochs, batch_size):

    # Define model
    model = keras.Sequential([
    keras.layers.Embedding(vocab_len, L1_size), # Embedding layer outputs 3D matrix
    keras.layers.GlobalMaxPooling1D(), # Pooling layer outputs 2D matrix
    keras.layers.Dropout(dropout), # Dropout layer to prevent overfitting
    keras.layers.Dense(L1_size, activation=tf.nn.relu), # Standard MLP dense layer with relu activation
    keras.layers.Dense(1, activation=tf.nn.sigmoid) # Binary output layer with sigmoid activation
    ])
    # Print a summary of my model
    model.summary()

    # Choose my loss function, optimizer, and metrics

    model.compile(optimizer=tf.train.AdamOptimizer(),
    loss="binary_crossentropy",
    metrics=["accuracy"])

    # Portion out a validation set to ensure I am not overitting my data
    x_val = train_data[:int(len(train_data)/2)]
    partial_x_train = train_data[int(len(train_data)/2):]
    y_val = train_labels[:int(len(train_labels)/2)]
    partial_y_train = [train_labels[int(len(train_labels)/2):]]

    # Train my model with the training and validation data
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=True)

    # Evaluate my metrics with the testing data
    results = model.evaluate(test_data, test_labels)
    print("Final loss is: ", results[0], "\n", "Accuracy is: ", results[1])

    # Prepare results for plotting
    history_dict = history.history
    history_dict.keys()
    val_loss = history_dict['val_loss']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    acc = history_dict['acc']
    epochs = range(1, len(acc) + 1)

    # Plot my training and validation loss over the number of epochs
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Loss over epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Plot my training and validation accuracy over the number of epochs
    plt.plot(epochs, acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def conv_model(vocab_len, L1_size, window, dropout, epochs, batch_size):

    # Define convolutional model
    model = keras.Sequential([
    keras.layers.Embedding(vocab_len, L1_size), # Embedding layer outputs 3D matrix
    keras.layers.Conv1D(L1_size, window, activation=tf.nn.relu), # The added convolutional layer for 5-grams
    keras.layers.GlobalMaxPooling1D(), # Pooling layer outputs 2D matrix
    keras.layers.Dropout(dropout), # Dropout layer to prevent overfitting
    keras.layers.Dense(L1_size, activation=tf.nn.relu), # Standard MLP dense layer with relu activation
    keras.layers.Dense(1, activation=tf.nn.sigmoid) # Binary output layer with sigmoid activation
    ])
    # Print a summary of my model
    model.summary()

    # Choose my loss function, optimizer, and metrics

    model.compile(optimizer=tf.train.AdamOptimizer(),
    loss="binary_crossentropy",
    metrics=["accuracy"])

    # Portion out a validation set to ensure I am not overitting my data
    x_val = train_data[:int(len(train_data)/2)]
    partial_x_train = train_data[int(len(train_data)/2):]
    y_val = train_labels[:int(len(train_labels)/2)]
    partial_y_train = [train_labels[int(len(train_labels)/2):]]

    # Train my model with the training and validation data
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=True)

    # Evaluate my metrics with the testing data
    results_with_conv = model.evaluate(test_data, test_labels)
    print("Final loss is: ", results_with_conv[0], "\n", "Accuracy is: ", results_with_conv[1])

    # Prepare loss and accuracy data for plotting
    history_dict = history.history
    history_dict.keys()
    val_loss = history_dict['val_loss']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    acc = history_dict['acc']
    epochs = range(1, len(acc) + 1)

    # Plot my training and validation loss over the number of epochs
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Loss over epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Plot my training and validation accuracy over the number of epochs
    plt.plot(epochs, acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

###############################################################################

vocab_len = 50000
random_state = 99
test_size = 0.25

L1_size = 128
window = 5
dropout = 0.2
epochs = 5
batch_size=650

train_data, test_data, train_labels, test_labels = preprocess(vocab_len, random_state, test_size)
std_model(vocab_len, L1_size, dropout, epochs, batch_size)
conv_model(vocab_len, L1_size, window, dropout, epochs, batch_size)



'''
These graphs indicate minor overfitting, but overall better results even with a
single convolutional layer.  The first two show the results of training
without a convolutional layer and the second two show the results with a single
convolutional layer. The first shows that the loss between the validation and training
set intersect at about 2.5 epochs.  The training loss continues to fall and the
validation loss begins to increase. The second graph indicates the change in accuracy
over the 5 epochs.  After the validation and training accuracy intersect, the training
accuracy continues to maximize and the validation accuracy plateaus. The seconds two graphs
show that adding a convolutional layer to the network did indeed increase the accuracy
by over 3%.  With all hyperparameters and parameters  otherwise the same,the network's
accuracy was increased, but the graphs indicate that the overfitting was exacerbated.


'''
