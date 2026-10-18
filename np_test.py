import numpy as np

mat1 = np.array([[1, 2, 3], [4, 5, 6]])

vec1 = np.array([5, 3])

vec2 = np.array([1, 2, 3])


# print(vec1 @ mat1)


arr1 = [1, 2, 3, 4, 5]

arr2 = [(arr1, 1), (vec1, 2), (vec2, 3), (arr1, 4)]

a, b = tuple(zip(*arr2))

print(a)
print(b)

# np.random.shuffle(arr1)

# print(arr1)