from matrixlib.properties import MatrixProperties
from matrixlib.core import Matrix

# This is a companion matrix example:
# [0 1 0 0]
# [0 0 1 0]
# [0 0 0 1]
# [1 2 3 4]
companion = Matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 2, 3, 4]  # Last row contains coefficients
])

# Create MatrixProperties instance
properties = MatrixProperties(companion)

# Test if it's a companion matrix
print(properties.is_companion())  # Should print True

# For comparison, here's a non-companion matrix
not_companion = Matrix([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])
print(MatrixProperties(not_companion).is_companion())

companion = Matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 2, 3, 4]  # Coefficients row
])
print("Is companion matrix?", MatrixProperties(companion).is_companion())