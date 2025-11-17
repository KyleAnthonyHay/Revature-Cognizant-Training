def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
message1 = greet("John")
message2 = greet("Jane", "Hi")
print(message1)  # Output: Hello, John!
print(message2)  # Output: Hi, Jane!

print("--------------------------------")

def sum_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total
result = sum_numbers(1, 2, 3, 4, 5)
print(result)  
# Output: 15

print("--------------------------------")
def print_person_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_person_info(name="Alice", age=30, city="Wonderland")
# Output:
# name: Alice
# age: 30
# city: Wonderland