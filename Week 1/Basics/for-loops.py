person = {"name": "Alice", "age": 25, "city": "New York"}

print("--------------------------------")
for key in person:
        print(key)
print("--------------------------------")
for value in person.values():
    print(value)
print("--------------------------------")
# Iterating over key-value pairs
for key, value in person.items():
        print(f"{key}: {value}")