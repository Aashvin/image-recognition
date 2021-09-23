import sys
import numpy
from keras.models import load_model, Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot
import random


def model_load(model_name: str):
    """
    Loads the specified model
    """

    print("Loading the model...")
    model = load_model(model_name)

    while True:
        summary = input("Would you like to see the summary of the model? y/n: ").lower()
        if summary == 'y':
            model.summary()
            break
        elif summary == 'n':
            break
        
        print("Please choose a valid option. y/n")

    return model

def process_data(model):
    """
    Loads the CIFAR 10 dataset and processes it to the correct shapes
    """

    print("Loading and processing the train/test data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

    scores = model.evaluate(x_test, y_test, verbose=0)

    while True:
        metrics = input("Would you like to see the accuracy metric? y/n: ").lower()
        if metrics == 'y':
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            break
        elif metrics == 'n':
            break
        
        print("Please choose a valid option. y/n")

    return x_test, y_test

def predict_new(model, x_test, y_test):
    """
    Predicts the contents of x_test using the model
    """

    for x in x_test:
        x = x.reshape(1, 32, 32, 3)

    predict_x = model.predict(x_test)
    y_test = numpy.argmax(predict_x, axis=1)

    return x_test, y_test

def get_classes():
    """
    Function to get the classes of the CIFAR 10 dataset
    """

    classes =   {
                    0: 'plane',
                    1: 'car',
                    2: 'bird',
                    3: 'cat',
                    4: 'deer',
                    5: 'dog',
                    6: 'frog',
                    7: 'horse',
                    8: 'ship',
                    9: 'truck'
                }

    return classes

def random_classification(classes, x_test, y_test, pure):
    """
    Picks and classifies a random picture
    If pure is true, the picture chosen is random from the set of all pictures
    If pure is false, the picture chosen is of the specified class
    """

    rand_num = random.randint(0, len(x_test) - 1)

    if pure:
        print("The pictured object in index " + str(rand_num) + " of the test set is a: " + classes[y_test[rand_num]])
    else:
        print("The pictured object is a random " + classes[y_test[rand_num]])

    pyplot.imshow(x_test[rand_num])
    pyplot.show()

def get_class_option(classes):
    """
    Gets the user input for a specified class of the CIFAR 10 dataset
    """

    for item in classes:
        print("    " + classes[item])

    while True:
        choice_val = input("Please type in a category from the list above: ").lower()
        if choice_val in classes.values():
            break

        for item in classes:
            print("    " + classes[item])
        print("That was not a valid choice, please select a valid option.")
    
    choice_key = list(classes.keys())[list(classes.values()).index(choice_val)]

    return choice_key, choice_val

def number_classifications(classes, y_test):
    """
    Gets the number of pictures classified as a specified class
    """

    choice_key, choice_val = get_class_option(classes)
    print("There are " + str(list(y_test).count(choice_key)) + " pictures classified as " + choice_val)

def selected_class_random_image(classes, x_test, y_test):
    """
    Creates a new set of data consisting of only the specified class
    """

    choice_key, choice_val = get_class_option(classes)
    new_x_test, new_y_test = [], []
    for index in range(len(x_test)):
        if y_test[index] == choice_key:
            new_x_test.append(x_test[index])
            new_y_test.append(y_test[index])

    random_classification(classes, new_x_test, new_y_test)

def get_run_choice():
    """
    Gets the user input for the main menu
    """

    while True:
        try:
            choice = int(input("Please enter your choice: "))
        except ValueError:
            print("That was not a valid choice, please enter a number between 1 and 3.")
        else:
            if choice > 0 and choice < 5:
                break
            print("That was not a valid choice, please enter a number between 1 and 3.")

    return choice

def run(classes, x_test, y_test):
    """
    Runs the main program
    """

    while True:
        print("What would you like to do? Options are:")
        print("    1: Get a random image and its classification from the test set")
        print("    2: Get the number of images classified as a particular classification")
        print("    3: Get a random image of a specified classification")
        print("    4: Exit")
        choice = get_run_choice()

        if choice == 1:
            random_classification(classes, x_test, y_test, True)
        elif choice == 2:
            number_classifications(classes, y_test)
        elif choice == 3:
            selected_class_random_image(classes, x_test, y_test, False)
        else:
            break

def main():
    if len(sys.argv) != 2:
        print("Please enter the filename as the first and only argument when running the file")
        print("Include the .h5 extension")
    else:
        model = model_load(sys.argv[1])
        x_test, y_test = process_data(model)
        x_test, y_test = predict_new(model, x_test, y_test)
        classes = get_classes()
        run(classes, x_test, y_test)
    

if __name__ == "__main__":
    main()