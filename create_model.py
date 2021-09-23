import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

SEED = 21

def load_datasets():
    """
    Load and shape the train and test data from the CIFAR 10 Dataset
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)
    class_num = y_test.shape[1]
    return x_train, y_train, x_test, y_test, class_num

def create_convolutional_layer(model, x_train, num_channels, pooling):
    """
    Creates convolutional layers with 32 channels of size 3x3
    Uses the Rectified Linear Unit (ReLU) activation function
    Uses dropouts to remove some connections between layers to prevent overfitting
    Batch normalisation normalises the inputs for the next layer so activations are created with the same distribution
    Also uses pooling layers so the classifier can learn relevant and important patterns in the images
    """

    if x_train is not None:
        # Done for the first layer, specifying shape of training data
        model.add(Conv2D(num_channels, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
    else:
        model.add(Conv2D(num_channels, (3, 3), activation='relu', padding='same'))
        if pooling:
            model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

def add_convolutional_layers(model, x_train):
    """
    Adds all the layers with an increasing filter size so network can learn more complex representations
    """

    create_convolutional_layer(model, x_train, 32, False)
    create_convolutional_layer(model, None, 64, True)
    create_convolutional_layer(model, None, 64, True)
    create_convolutional_layer(model, None, 128, False)

def flatten_data(model):
    """
    The data in the model needs to be flattened before going through the densely connected layers
    """

    model.add(Flatten())
    model.add(Dropout(0.2))

def create_densely_connected_layer(model, num_neurons, last):
    """
    Creates the fully connected layers / Artificial Neural Network (ANN)
    Analyses the input features and combine into attributes that assist classification
    Forms collections of neurons that represent different parts of objects in pictures
    """

    if not last:
        model.add(Dense(num_neurons, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    else:
        model.add(Dense(num_neurons))
        # Use the Softmax activation function to select the neuron with the highest probability as the output
        # This means classifying that image as the class that it mose likely belongs to
        model.add(Activation('softmax'))

def add_densely_connected_layers(model, class_num):
    """
    Adds the fully connected layers to the model
    """

    create_densely_connected_layer(model, 256, False)
    create_densely_connected_layer(model, 128, False)
    create_densely_connected_layer(model, class_num, True)

def compile_model(model, epochs, optimiser):
    """
    Compiles the final model and can print the summary
    """

    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    while True:
        summary = input("Would you like to view the summary? y/n: ")
        if summary.lower() == "y":
            print(model.summary())
            break
        elif summary.lower() == "n":
            break
        print("Please enter a valid choice.")

def train_model(model, x_train, y_train, x_test, y_test, epochs):
    """
    Trains and evaluates the model
    """

    numpy.random.seed(SEED)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def save_model(model):
    """
    Saves the model to a file name specified by user
    """

    file_name = input("Please input the model name: ")
    file_name = file_name + ".h5"
    model.save(file_name)
    print("Saved model to disk")

def main():
    x_train, y_train, x_test, y_test, class_num = load_datasets()
    model = Sequential()
    epochs = 2

    add_convolutional_layers(model, x_train)
    flatten_data(model)
    add_densely_connected_layers(model, class_num)
    compile_model(model, epochs=epochs, optimiser='adam')
    train_model(model, x_train, y_train, x_test, y_test, epochs)
    save_model(model)

if __name__ == "__main__":
    main()