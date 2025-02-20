from matrixlib.core import Matrix
from matrixlib.distances import Distance


dist = Distance()
# These could represent features of something (like temperature and humidity)
measurements1 = Matrix([
    [25, 60],  # 25°C, 60% humidity
    [24, 65]   # 24°C, 65% humidity
])

measurements2 = Matrix([
    [26, 58],
    [25, 63]
])

# Find out how similar these measurements are
similarity = dist.calculate(measurements1, measurements2, metric='euclidean')
print(f"Similarity of measurements: {similarity}")

# For more different scales, you might want minkowski with p=3
minkowski = dist.calculate(measurements1, measurements2, metric='minkowski', p=3)
print(f"Minkowski distance (p=3): {minkowski}")

#Similarity of measurements: 3.1622776601683795
#Minkowski distance (p=3): 2.6207413942088964

print(measurements1.get_rank())
