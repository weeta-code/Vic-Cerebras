#!/usr/bin/env python3
"""
Test script to verify Grepal's improved bug detection capabilities
"""

# Test case 1: JavaScript - Off-by-one error
js_code_with_bugs = '''
function processArray(arr) {
    let results = [];
    for (let i = 0; i <= arr.length; i++) {  // Bug: off-by-one error
        results.push(arr[i] * 2);
        arr.push(arr[i]);  // Bug: modifying array during iteration
    }
    return results;
}

function getUserName(users, id) {
    return users[id].name;  // Bug: potential null reference
}
'''

# Test case 2: Python - Division by zero and list modification
python_code_with_bugs = '''
def calculate_average(numbers):
    total = sum(numbers)
    return total / 0  # Bug: division by zero

def process_data(data=[]):  # Bug: mutable default argument
    for item in data:
        data.append(item * 2)  # Bug: modifying list during iteration
    return data

def get_item(items, index):
    return items[5]  # Bug: hardcoded index without bounds check
'''

# Test case 3: Mixed JavaScript/Python (similar to user's example)
mixed_code = '''
// JavaScript part
for (let i = 0; i <= array.length; i++) {
    console.log(array[i]);
    array.push(i);
}

# Python part  
def divide(a, b):
    return a / 0

list = [1, 2, 3]
for item in list:
    list.append(item)
'''

print("Test cases for improved Grepal bug detection:")
print("\n1. JavaScript with off-by-one, array modification, and null reference bugs")
print(js_code_with_bugs)

print("\n2. Python with division by zero, mutable defaults, and unsafe access")  
print(python_code_with_bugs)

print("\n3. Mixed JavaScript/Python code (similar to user's example)")
print(mixed_code)

print("\nTo test these:")
print("1. Start the Grepal server: python server/main.py")
print("2. Open VSCode with the Grepal extension")
print("3. Create test files with the above code")
print("4. Watch the server logs for improved debugging output")
print("5. Check that bugs are detected by both embeddings similarity AND enhanced fallback analysis")