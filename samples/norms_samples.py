from matrixlib.core import Matrix
from matrixlib.norms import Norm

# Create your measurement matrices
measurements1 = Matrix([
    [25, 60],  # 25°C, 60% humidity
    [24, 65]  # 24°C, 65% humidity
])

measurements2 = Matrix([
    [26, 58],
    [25, 63]
])

# Calculate Frobenius norm (overall magnitude of measurements)
frob1 = Norm.frobenius(measurements1)
frob2 = Norm.frobenius(measurements2)
print(f"Frobenius norms: {frob1:.2f}, {frob2:.2f}")

# Calculate max norm (maximum absolute row sum)
max1 = Norm.max_norm(measurements1)
max2 = Norm.max_norm(measurements2)
print(f"Max norms: {max1:.2f}, {max2:.2f}")

# Calculate p-norm with p=1 (Manhattan distance)
p1_norm1 = Norm.p_norm(measurements1, p=1)
p1_norm2 = Norm.p_norm(measurements2, p=1)
print(f"1-norms: {p1_norm1:.2f}, {p1_norm2:.2f}")

# Calculate condition number (measure of matrix sensitivity)
cond1 = Norm.condition_number(measurements1)
cond2 = Norm.condition_number(measurements2)
print(f"Condition numbers: {cond1:.2f}, {cond2:.2f}")
