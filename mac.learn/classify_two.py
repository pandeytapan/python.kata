# Load the libraries
import numpy as np
import matplotlib.pyplot as plt # for plotting
from sklearn import datasets # for loading the iris dataset
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.tree import DecisionTreeClassifier, plot_tree # for using the decision tree classifier
from sklearn.metrics import accuracy_score # for evaluating the model
import json # for saving the test data


# Write a functoin that will print the sepal length and width
# for a given species
def plot_sepal_length_width(species_name, sepal_length, sepal_width):
    plt.scatter(sepal_length, sepal_width, label = species_name)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Sepal Length vs Width for {}".format(species_name))
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the iris dataset
iris = datasets.load_iris()
# Explain the dataset
# print(iris.DESCR)
feature = iris.data
label = iris.target

print("Feature shape: ", feature.shape)
print("Label shape: ", label.shape)

# Split the data into training and testing sets
# for each feature, 70% of the data is used for training and 30% for testing
# you'll get back 4 arrays: feature_train, feature_test, label_train, label_test
# two arrays for the features and two arrays for the labels
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3, random_state=42)

# Save the feature_test data to a json file
# with open("test_feature_iris_data.json", "w") as f:
#     json.dump(feature_test.tolist(), f)
# feature_test = None

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth = 5, random_state = 42)

# Train the classifier on the training data that includes the features and the labels
clf.fit(feature_train, label_train)

# Load the label test data i.e. the features from the json file 
with open("test_feature_iris_data.json", "r") as f:
    feature_test = np.array(json.load(f))

# Extract the target labels from the test data
species_names = iris.target_names

for index, species in enumerate(species_names):
    print(index, species)
    # Get sepal length and width from the species
    sepal_length = feature[label == index][:, 0]
    sepal_width = feature[label == index][:, 1]
    # Plot the sepal length and width for given species in a function
    plot_sepal_length_width(species, sepal_length, sepal_width)


