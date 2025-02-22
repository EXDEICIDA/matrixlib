# Matrix Norms (Norm Class)
## Overview

The `Norm` class provides a comprehensive collection of matrix norm computations and related utilities. This class implements various types of matrix norms, including standard norms (Frobenius, spectral, nuclear), induced p-norms, and specialized norms like the Schatten norm. The implementation relies on the `Matrix` class from `matrixlib.core` for basic matrix operations.

## Class Methods

### Standard Norms

#### `frobenius(matrix: Matrix) -> float`
Computes the Frobenius norm (also known as the Hilbert-Schmidt norm) of a matrix.

**Mathematical Definition:**
```
||A||_F = sqrt(sum_{i,j} |a_{ij}|^2)
```

**Properties:**
- Equivalent to the square root of the sum of squares of all matrix elements
- Equivalent to the square root of the sum of squares of singular values
- Invariant under unitary transformations
- Satisfies the submultiplicative property: ||AB||_F ≤ ||A||_F ||B||_F

#### `max_norm(matrix: Matrix) -> float`
Calculates the maximum absolute row sum norm (also known as infinity norm).

**Mathematical Definition:**
```
||A||_∞ = max_i sum_j |a_{ij}|
```

**Properties:**
- Represents the maximum absolute row sum
- Useful in convergence analysis of iterative methods
- Dual norm to the 1-norm

#### `spectral_norm(matrix: Matrix) -> float`
Computes the spectral norm (also known as 2-norm or operator norm).

**Mathematical Definition:**
```
||A||_2 = sqrt(λ_max(A^T A))
```
where λ_max denotes the largest eigenvalue.

**Properties:**
- Equal to the largest singular value
- Invariant under unitary transformations
- Important in stability analysis
- Used in condition number calculations

### Advanced Norms

#### `nuclear_norm(matrix: Matrix) -> float`
Calculates the nuclear norm (also known as trace norm).

**Mathematical Definition:**
```
||A||_* = sum_i σ_i
```
where σ_i are the singular values.

**Properties:**
- Dual norm to the spectral norm
- Convex relaxation of matrix rank
- Used in matrix completion problems
- Important in low-rank approximation

#### `p_norm(matrix: Matrix, p: Union[int, float] = 2) -> float`
Computes the entry-wise p-norm.

**Mathematical Definition:**
```
||A||_p = (sum_{i,j} |a_{ij}|^p)^(1/p)
```

**Requirements:**
- p ≥ 1
- Returns float value representing the norm

**Properties:**
- Generalizes the concept of vector p-norms to matrices
- Special cases:
  - p = 1: Manhattan norm
  - p = 2: Frobenius norm
  - p = ∞: Maximum absolute value

#### `induced_norm(matrix: Matrix, p: Union[int, float] = 2) -> float`
Calculates the induced p-norm.

**Mathematical Definition:**
```
||A||_p = max_{x≠0} ||Ax||_p / ||x||_p
```

**Supported Values:**
- p = 1: Maximum absolute column sum
- p = 2: Spectral norm
- p = ∞: Maximum absolute row sum

**Properties:**
- Represents the maximum "stretching" factor of the matrix
- Satisfies submultiplicative property
- Important in numerical stability analysis

### Specialized Methods

#### `schatten_norm(matrix: Matrix, p: Union[int, float] = 2) -> float`
Computes the Schatten p-norm.

**Mathematical Definition:**
```
||A||_p = (sum_i σ_i^p)^(1/p)
```
where σ_i are singular values.

**Special Cases:**
- p = 1: Nuclear norm
- p = 2: Frobenius norm
- p = ∞: Spectral norm

**Properties:**
- Unitarily invariant
- Important in quantum information theory
- Generalizes standard matrix norms

#### `condition_number(matrix: Matrix, p: Union[int, float] = 2) -> float`
Calculates the condition number with respect to a given norm.

**Mathematical Definition:**
```
κ_p(A) = ||A||_p ||A^{-1}||_p
```

**Properties:**
- Measures numerical stability
- Indicates sensitivity to perturbations
- Returns ∞ for singular matrices
- Important in numerical linear algebra

#### `relative_norm(matrix1: Matrix, matrix2: Matrix, norm_type: str = 'frobenius') -> float`
Computes the relative norm between two matrices.

**Mathematical Definition:**
```
rel_norm = ||A - B|| / ||B||
```

**Requirements:**
- Matrices must have same dimensions
- Supports various norm types through string parameter

## Implementation Details

### Helper Methods

#### `_get_singular_values(matrix: Matrix) -> List[float]`
Internal method for computing singular values using power iteration.

**Algorithm Overview:**
1. Constructs A^T A matrix
2. Implements power iteration method
3. Uses deflation for multiple singular values
4. Returns sorted list of singular values

#### `_matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]`
Internal method for matrix multiplication.

**Features:**
- Standard matrix multiplication algorithm
- Handles arbitrary dimensions
- Used in various norm computations

#### `_matrix_inverse(matrix: List[List[float]]) -> List[List[float]]`
Internal method for matrix inversion using Gauss-Jordan elimination.

**Algorithm Features:**
- Full pivoting
- In-place augmented matrix operations
- Error handling for singular matrices

## Usage Examples

```python
from matrixlib.core import Matrix
from norms import Norm

# Create a sample matrix
matrix = Matrix([[1, 2], [3, 4]])

# Computing various norms
frob_norm = Norm.frobenius(matrix)
spec_norm = Norm.spectral_norm(matrix)
nucl_norm = Norm.nuclear_norm(matrix)

# Computing condition number
cond = Norm.condition_number(matrix)

# Computing relative norm between matrices
matrix2 = Matrix([[1.1, 2.1], [3.1, 4.1]])
rel_norm = Norm.relative_norm(matrix, matrix2)
```

## Performance Considerations

1. **Singular Value Computation**
   - Uses power iteration method
   - Trade-off between accuracy and speed
   - Suitable for moderate-sized matrices

2. **Memory Usage**
   - Creates temporary matrices for some computations
   - Memory complexity: O(n²) for n×n matrices

3. **Computational Complexity**
   - Frobenius norm: O(n²)
   - Spectral norm: O(k·n²) where k is iteration count
   - Nuclear norm: O(k·n³)
   - Condition number: O(n³)

## Error Handling

The class implements robust error handling for:
- Invalid p-norm parameters (p < 1)
- Singular matrices in condition number computation
- Dimension mismatches in relative norm
- Numerical stability issues in singular value computation

## Mathematical Background

Matrix norms are essential in:
1. Numerical analysis
2. Linear algebra computations
3. Optimization problems
4. Error analysis
5. Convergence studies

They satisfy the following properties:
1. Non-negativity: ||A|| ≥ 0
2. Positive definiteness: ||A|| = 0 iff A = 0
3. Homogeneity: ||αA|| = |α| ||A||
4. Triangle inequality: ||A + B|| ≤ ||A|| + ||B||

## References

1. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations
2. Horn, R. A., & Johnson, C. R. (2012). Matrix Analysis
3. Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra