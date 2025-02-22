from matrixlib.core import Matrix
from matrixlib.visualization import Visualize

# Sample data
data = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [5, 6, 7, 8],
    [8, 7, 6, 5]
]

# Create a Matrix instance
matrix = Matrix(data)

# Compute the correlation matrix
correlation_matrix = matrix.correlation()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualizing the correlation matrix
visualizer = Visualize()
visualizer.to_heatmap(correlation_matrix)
