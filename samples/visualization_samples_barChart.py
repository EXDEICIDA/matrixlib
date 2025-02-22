from matrixlib.core import Matrix
from matrixlib.visualization import Visualize




data = [
    [1, 3, 5, 7],
    [2, 4, 6, 8],
    [5, 2, 7, 1]
]

matrix = Matrix(data)
visualizer = Visualize()
visualizer.to_barchart(matrix)