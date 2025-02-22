# Factorization Class Documentation

## Overview
The `Factorization` class provides a comprehensive collection of matrix factorization methods commonly used in linear algebra and numerical computations. This class is part of the matrixlib package and offers implementations of various decomposition techniques including LU, Eigenvalue, SVD, QR, Jordan, and Polar decompositions.

## Table of Contents
- [Dependencies](#dependencies)
- [Class Methods](#class-methods)
  - [LU Decomposition](#lu-decomposition)
  - [Eigenvalue Decomposition](#eigenvalue-decomposition)
  - [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
  - [QR Decomposition](#qr-decomposition)
  - [Jordan Decomposition](#jordan-decomposition)
  - [Polar Decomposition](#polar-decomposition)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Mathematical Background](#mathematical-background)
- [Performance Considerations](#performance-considerations)

## Dependencies
```python
import math
from matrixlib.operations import Operation
from random import random
from matrixlib.core import Matrix
```

## Class Methods

### LU Decomposition
```python
@staticmethod
def lu_decomposition(matrix: Matrix) -> tuple[list, list]
```

#### Description
Performs LU decomposition on a square matrix, factoring it into the product of a lower triangular matrix (L) and an upper triangular matrix (U).

#### Parameters
- `matrix`: A square Matrix object to be decomposed

#### Returns
- Tuple containing:
  - `l`: Lower triangular matrix
  - `u`: Upper triangular matrix

#### Mathematical Background
LU decomposition factors a matrix A into the product L × U where:
- L is a lower triangular matrix with 1's on the diagonal
- U is an upper triangular matrix

#### Usage Example
```python
from matrixlib.core import Matrix
from matrixlib.factorization import Factorization

# Create a matrix
A = Matrix([[4, 3], [6, 3]])

# Perform LU decomposition
L, U = Factorization.lu_decomposition(A)

# L will be like: [[1, 0], [1.5, 1]]
# U will be like: [[4, 3], [0, -1.5]]
```

#### Limitations
- Only works with square matrices
- May be numerically unstable for ill-conditioned matrices
- Does not include pivoting strategies

### Eigenvalue Decomposition
```python
@staticmethod
def eigenvalue_decomposition(matrix: Matrix, max_iterations=100, tolerance=1e-10) -> tuple[list, list]
```

#### Description
Computes the eigenvalues and eigenvectors of a square matrix using the power iteration method.

#### Parameters
- `matrix`: Square Matrix object
- `max_iterations`: Maximum number of iterations for the power method (default: 100)
- `tolerance`: Convergence tolerance (default: 1e-10)

#### Returns
- Tuple containing:
  - `eigenvalues`: List of eigenvalues
  - `eigenvectors`: List of corresponding eigenvectors

#### Mathematical Background
For a square matrix A, an eigenvector v and corresponding eigenvalue λ satisfy:
```
Av = λv
```
The power iteration method is used to find the dominant eigenvalue and corresponding eigenvector.

#### Usage Example
```python
# Create a 2x2 matrix
A = Matrix([[4, 1], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = Factorization.eigenvalue_decomposition(A)

# eigenvalues ≈ [4.618, 2.382]
# eigenvectors will be the corresponding unit vectors
```

### Singular Value Decomposition (SVD)
```python
@staticmethod
def svd(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]
```

#### Description
Performs Singular Value Decomposition, factoring a matrix into the product U × S × V^T.

#### Parameters
- `matrix`: Matrix object to decompose

#### Returns
- Tuple containing:
  - `U`: Left singular vectors (orthogonal matrix)
  - `S`: Diagonal matrix of singular values
  - `V_T`: Transpose of right singular vectors (orthogonal matrix)

#### Mathematical Background
SVD decomposes a matrix A into the product:
```
A = USV^T
```
where:
- U and V are orthogonal matrices
- S is a diagonal matrix containing singular values in descending order

#### Usage Example
```python
# Create a matrix
A = Matrix([[4, 0], [3, -5]])

# Perform SVD
U, S, V_T = Factorization.svd(A)

# S will contain the singular values in descending order
# U and V_T will be orthogonal matrices
```

### QR Decomposition
```python
@staticmethod
def qr_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix]
```

#### Description
Performs QR decomposition using the Gram-Schmidt process, factoring a square matrix into the product of an orthogonal matrix Q and an upper triangular matrix R.

#### Parameters
- `matrix`: Square Matrix object

#### Returns
- Tuple containing:
  - `Q`: Orthogonal matrix
  - `R`: Upper triangular matrix

#### Mathematical Background
QR decomposition represents a matrix A as:
```
A = QR
```
where:
- Q is an orthogonal matrix (Q^T Q = I)
- R is an upper triangular matrix

#### Usage Example
```python
# Create a matrix
A = Matrix([[1, -1], [1, 1]])

# Perform QR decomposition
Q, R = Factorization.qr_decomposition(A)

# Q will be orthogonal
# R will be upper triangular
```

### Jordan Decomposition
```python
@staticmethod
def jordan_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]
```

#### Description
Computes the Jordan canonical form of a matrix, expressing it as P × J × P^(-1).

#### Parameters
- `matrix`: Square Matrix object

#### Returns
- Tuple containing:
  - `P`: Transformation matrix
  - `J`: Jordan form matrix
  - `P_inverse`: Inverse of transformation matrix

#### Mathematical Background
Jordan decomposition expresses a matrix A as:
```
A = PJP^(-1)
```
where:
- J is the Jordan canonical form
- P is the transformation matrix
- Each Jordan block corresponds to an eigenvalue

#### Usage Example
```python
# Create a matrix
A = Matrix([[2, 1], [0, 2]])

# Perform Jordan decomposition
P, J, P_inv = Factorization.jordan_decomposition(A)

# J will contain Jordan blocks
# P will contain generalized eigenvectors
```

### Polar Decomposition
```python
@staticmethod
def polar_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix]
```

#### Description
Computes the polar decomposition, factoring a matrix into the product of a unitary matrix and a positive semi-definite Hermitian matrix.

#### Parameters
- `matrix`: Matrix object

#### Returns
- Tuple containing:
  - `U`: Unitary matrix
  - `P`: Positive semi-definite Hermitian matrix

#### Mathematical Background
Polar decomposition expresses a matrix A as:
```
A = UP
```
where:
- U is unitary (U^T U = I)
- P is positive semi-definite Hermitian

#### Usage Example
```python
# Create a matrix
A = Matrix([[1, -1], [1, 1]])

# Perform polar decomposition
U, P = Factorization.polar_decomposition(A)

# U will be unitary
# P will be positive semi-definite
```

## Error Handling

The class implements various error checks:

1. Square Matrix Requirements:
   - LU Decomposition
   - QR Decomposition
   - Jordan Decomposition
   - Eigenvalue Decomposition

```python
if not matrix.is_square_matrix():
    raise ValueError("Decomposition can only be done on square matrices.")
```

2. Numerical Stability:
   - Polar Decomposition includes regularization for singular matrices
   - SVD handles small numerical errors in eigenvalues

## Performance Considerations

1. Memory Usage:
   - All methods create copies of input matrices to prevent modification
   - Temporary matrices are created during calculations

2. Computational Complexity:
   - LU Decomposition: O(n³)
   - Eigenvalue Decomposition: O(kn²) per iteration, where k is max_iterations
   - SVD: O(mn²) for m×n matrix
   - QR Decomposition: O(n³)
   - Jordan Decomposition: O(n³)
   - Polar Decomposition: O(n³)

3. Optimization Opportunities:
   - Consider using numpy for large matrices
   - Implement parallel processing for large matrices
   - Add pivoting strategies for numerical stability

## Best Practices

1. Input Validation:
   ```python
   # Always check matrix dimensions
   if not matrix.is_square_matrix():
       raise ValueError("Invalid matrix dimensions")
   ```

2. Numerical Stability:
   ```python
   # Use tolerance for floating-point comparisons
   if abs(value) < tolerance:
       # Handle near-zero values
   ```

3. Error Handling:
   ```python
   try:
       result = Factorization.method(matrix)
   except ValueError as e:
       # Handle invalid input
   except Exception as e:
       # Handle numerical errors
   ```

## Contributing

When adding new factorization methods:

1. Follow the existing pattern of static methods
2. Include comprehensive input validation
3. Add appropriate error handling
4. Document mathematical background
5. Provide usage examples
6. Consider numerical stability
7. Add performance optimization where possible

## Testing Recommendations

1. Unit Tests:
   - Test with known matrices and their factorizations
   - Test edge cases (singular matrices, near-singular matrices)
   - Test error conditions
   - Verify mathematical properties (e.g., A = LU, Q is orthogonal)

2. Performance Tests:
   - Test with large matrices
   - Measure execution time
   - Monitor memory usage

3. Numerical Stability Tests:
   - Test with ill-conditioned matrices
   - Verify accuracy of results
   - Check for numerical overflow/underflow

## References

1. Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.). JHU Press.
2. Trefethen, L. N., & Bau III, D. (1997). Numerical linear algebra. SIAM.
3. Horn, R. A., & Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.