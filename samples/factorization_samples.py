from matrixlib.core import Matrix
from matrixlib.factorization import Factorization

# Define your matrix (measurements)
measurements1 = Matrix([
    [25, 60],  # 25°C, 60% humidity
    [24, 65]   # 24°C, 65% humidity
])

# Perform LU Decomposition using the Factorization class
L, U = Factorization.lu_decomposition(measurements1)

# Display the result
print("L Matrix (Lower Triangular):")
for row in L:
    print(row)

print("\nU Matrix (Upper Triangular):")
for row in U:
    print(row)

"""L Matrix (Lower Triangular):
[1, 0]
[0.96, 1]

U Matrix (Upper Triangular):
[25, 60]
[0, 4]
"""