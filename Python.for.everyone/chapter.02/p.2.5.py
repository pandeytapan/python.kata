num_one = int(input("Enter first number: "))
num_two = int(input("Enter second number: "))

print("Summation:%20d" % (num_one + num_two))
print("Difference:%19d" % (num_one - num_two))
print("Product: %21d" % (num_one * num_two))
print("Average: %21.2f" % (num_one / num_two))
print("Distance: %20d" % abs(num_one - num_two))
print("Maximum: %21d" % max(num_one, num_two))
print("Minimum: %21d" % min(num_one, num_two))