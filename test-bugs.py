# Test Python file with bugs for Grepal to demolish!

# Bug 1: Division by zero (Classic)
def divide(a, b):
    return a / b  # No check for b == 0

# Bug 2: Modifying list during iteration
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    if item % 2 == 0:
        my_list.append(item * 2)  # Dangerous!

# Bug 3: Mutable default argument
def add_item(item, target_list=[]):
    target_list.append(item)  # This will bite you!
    return target_list

# Bug 4: Using print without f-strings (inefficient)
name = "Alice"
age = 25
print("Hello, my name is " + name + " and I am " + str(age) + " years old")

# Bug 5: Comparing with None using ==
def check_value(value):
    if value == None:  # Should use 'is None'
        return "No value"
    return value

# Bug 6: Unused imports and variables
import os
import sys
import json

unused_variable = "I'm just taking up space"

# Bug 7: Infinite recursion potential
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # No check for negative numbers!

# Bug 8: SQL injection waiting to happen (if this was real SQL)
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Dangerous!
    return query

print("Grepal is going to have a field day with this code!")
