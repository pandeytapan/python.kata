from sklearn.tree import DecisionTreeClassifier

# Example data : [size, weight] and type of fruit
data = [[120, 80], [100, 60], [130, 90], [150, 100]]

# Example label : Type oof fruit. Apple or Orange
labels = ['Apple', 'Apple', 'Orange', 'Orange']

# Create a model
model = DecisionTreeClassifier()
# train the model over the data
model.fit(data, labels)

# Predict the type of fruit for given size and weight
# Do this repeatedly for different values of size and weight
while True:
    size = int(input("Enter size of fruit (in cms): "))
    weight = int(input("Enter weight of fruit (in gms): "))
    # Model accepts a list of lists as input
    fruit = model.predict([[size, weight]])
    print(f"The predicted fruit is {fruit}")

    # Ask user if he wants to continue
    choice = input("Do you want to continue? (y/n): ")
    if choice == 'n':
        break
