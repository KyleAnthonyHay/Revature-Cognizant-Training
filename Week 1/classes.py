class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Creating instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Accessing attributes
print(person1.name)  # Output: Alice
print(person2.age)   # Output: 30

# Modifying attributes
person1.age = 26
person2.name = "Robert"

print(person1.age)   # Output: 26
print(person2.name)  # Output: Robert



