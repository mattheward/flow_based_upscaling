test = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

i = 2
x_neighbor = next((num for num in test[1 - 1] if num > i), None)
print(x_neighbor)
