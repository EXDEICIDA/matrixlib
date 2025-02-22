# Operation Class Documentation

## Overview
The `Operation` class provides a comprehensive collection of matrix operations implemented recursively to handle N-dimensional matrices. This class is part of the matrixlib package and includes fundamental matrix operations, advanced linear algebra computations, and statistical methods.

## Table of Contents
- [Basic Matrix Operations](#basic-matrix-operations)
  - [Addition](#addition)
  - [Subtraction](#subtraction)
  - [Multiplication](#multiplication)
  - [Transpose](#transpose)
- [Advanced Operations](#advanced-operations)
  - [Determinant](#determinant)
  - [Inverse](#inverse)
  - [Adjugate](#adjugate)
  - [Trace](#trace)
- [Special Products](#special-products)
  - [Dot Product](#dot-product)
  - [Hadamard Product](#hadamard-product)
  - [Matrix Exponentiation](#matrix-exponentiation)
- [Row Operations](#row-operations)
  - [Swap Rows](#swap-rows)
  - [Scale Row](#scale-row)
- [Statistical Operations](#statistical-operations)
  - [Correlation](#correlation)

## Basic Matrix Operations

### Addition
```python
@staticmethod
def add(a: Matrix, b: Matrix) -> Matrix
```

#### Description
Performs element-wise addition of two matrices of the same shape.

#### Parameters
- `a`: First Matrix object
- `b`: Second Matrix object

#### Returns
- Matrix object containing the sum

#### Mathematical Expression
For matrices A and B:
```
C = A + B where cᵢⱼ = aᵢⱼ + bᵢⱼ
```

#### Example
```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = Operation.add(A, B)  # Results in [[6, 8], [10, 12]]
```

### Subtraction
```python
@staticmethod
def subtract(a: Matrix, b: Matrix) -> Matrix
```

#### Description
Performs element-wise subtraction of two matrices of the same shape.

#### Parameters
- `a`: First Matrix object
- `b`: Second Matrix object

#### Returns
- Matrix object containing the difference

#### Mathematical Expression
```
C = A - B where cᵢⱼ = aᵢⱼ - bᵢⱼ
```

### Multiplication
```python
@staticmethod
def multiply(a: Matrix, b: Matrix) -> Matrix
```

#### Description
Performs matrix multiplication using recursive implementation.

#### Parameters
- `a`: First Matrix object
- `b`: Second Matrix object

#### Returns
- Matrix object containing the product

#### Mathematical Expression
```
C = A × B where cᵢⱼ = Σₖ(aᵢₖ × bₖⱼ)
```

#### Complexity
- Time: O(n³) for n×n matrices
- Space: O(n²)

## Advanced Operations

### Determinant
```python
@staticmethod
def determinant(matrix: Matrix) -> float
```

#### Description
Calculates the determinant using recursive Laplace expansion.

#### Mathematical Expression
For a 2×2 matrix:
```
|A| = a₁₁a₂₂ - a₁₂a₂₁
```

For larger matrices:
```
|A| = Σⱼ((-1)ⁱ⁺ʲaᵢⱼMᵢⱼ)
```

#### Example
```python
A = Matrix([[1, 2], [3, 4]])
det = Operation.determinant(A)  # Results in -2
```

### Inverse
```python
@staticmethod
def inverse(matrix: Matrix) -> Matrix
```

#### Description
Calculates the inverse matrix using the adjugate method.

#### Mathematical Expression
```
A⁻¹ = (1/|A|) × adj(A)
```

#### Requirements
- Matrix must be square
- Determinant must be non-zero

## Special Products

### Dot Product
```python
@staticmethod
def dot_product(a: Matrix, b: Matrix) -> Matrix
```

#### Description
Computes the dot product of two matrices with matching inner dimensions.

#### Mathematical Expression
For vectors:
```
a·b = Σᵢ(aᵢbᵢ)
```

### Hadamard Product
```python
@staticmethod
def hadamard_product(a: Matrix, b: Matrix) -> Matrix
```

#### Description
Performs element-wise multiplication of two matrices.

#### Mathematical Expression
```
C = A ⊙ B where cᵢⱼ = aᵢⱼ × bᵢⱼ
```

## Statistical Operations

### Correlation
```python
@staticmethod
def correlation(matrix: Matrix) -> Matrix
```

#### Description
Computes the Pearson correlation coefficient matrix.

#### Mathematical Details
1. Column means: μⱼ = (1/n)Σᵢxᵢⱼ
2. Standard deviations: σⱼ = √((1/(n-1))Σᵢ(xᵢⱼ - μⱼ)²)
3. Correlation: ρᵢⱼ = cov(i,j)/(σᵢσⱼ)

#### Example
```python
data = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
correlation_matrix = Operation.correlation(data)
```

## Implementation Details

### Recursive Approach
Many operations use recursive implementations to handle N-dimensional matrices:

```python
def recursive_operation(data):
    if isinstance(data, list):
        return [recursive_operation(sub) for sub in data]
    return operation(data)  # Base case
```

### Error Handling

1. Dimension Mismatch
```python
if a.shape != b.shape:
    raise ValueError("Matrices must have same shape")
```

2. Square Matrix Requirements
```python
if not matrix.is_square_matrix():
    raise ValueError("Operation requires square matrix")
```

3. Singular Matrix
```python
if determinant == 0:
    raise ValueError("Matrix is singular")
```

## Performance Considerations

### Time Complexity
- Addition/Subtraction: O(n²)
- Multiplication: O(n³)
- Determinant: O(n!)
- Inverse: O(n³)
- Correlation: O(n²m) where n = rows, m = columns

### Space Complexity
- Most operations: O(n²)
- Recursive operations: O(n²) + stack space

### Optimization Opportunities
1. Implement parallel processing for large matrices
2. Use numpy for performance-critical applications
3. Implement Strassen's algorithm for multiplication
4. Cache frequently used computations

## Best Practices

### Input Validation
```python
def validate_dimensions(a: Matrix, b: Matrix):
    if a.shape != b.shape:
        raise ValueError("Invalid dimensions")
```

### Numerical Stability
```python
if abs(value) < 1e-10:  # Use appropriate tolerance
    # Handle near-zero values
```

## Testing Recommendations

1. Unit Tests
   - Test with known matrices
   - Test edge cases
   - Verify mathematical properties
   - Test error conditions

2. Performance Tests
   - Large matrices
   - Complex operations
   - Memory usage

3. Numerical Tests
   - Test with ill-conditioned matrices
   - Verify precision
   - Check for numerical stability

## Examples

### Basic Operations
```python
# Create matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

# Addition
C = Operation.add(A, B)

# Multiplication
D = Operation.multiply(A, B)

# Transpose
E = Operation.transpose(A)
```

### Advanced Operations
```python
# Calculate determinant
det = Operation.determinant(A)

# Find inverse
inv = Operation.inverse(A)

# Matrix exponentiation
squared = Operation.matrix_exponentiation(A, 2)
```

### Statistical Analysis
```python
# Create data matrix
data = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Calculate correlation matrix
corr = Operation.correlation(data)
```

## References

1. Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU Press.
2. Press, W. H. (2007). Numerical recipes: The art of scientific computing. Cambridge university press.
3. Strang, G. (2016). Introduction to linear algebra. Wellesley-Cambridge Press.
4. Horn, R. A., & Johnson, C. R. (2012). Matrix analysis. Cambridge university press.