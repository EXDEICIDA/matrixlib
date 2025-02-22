# Matrix Properties Testing

This script provides a comprehensive demonstration of various matrix properties and types. It utilizes the `Matrix` and `MatrixProperties` classes to verify different matrix characteristics.

## Usage
Run the script to test various matrix properties:
```bash
python matrix_test.py
```

## Matrix Types and Examples

### 1. Companion Matrix
A **Companion Matrix** is a special form used in characteristic equations.
```python
companion = Matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 2, 3, 4]  # Coefficients row
])
print("Is companion matrix?", MatrixProperties(companion).is_companion())
```

### 2. Triangular Matrices
#### Upper Triangular
A matrix where all elements below the main diagonal are zero.
```python
upper_triangular = Matrix([
    [1, 2, 3],
    [0, 4, 5],
    [0, 0, 6]
])
print("Is upper triangular?", MatrixProperties(upper_triangular).is_upper_triangular())
```
#### Lower Triangular
A matrix where all elements above the main diagonal are zero.
```python
lower_triangular = Matrix([
    [1, 0, 0],
    [4, 5, 0],
    [7, 8, 9]
])
print("Is lower triangular?", MatrixProperties(lower_triangular).is_lower_triangular())
```

### 3. Hessenberg Matrix
A **Hessenberg Matrix** is almost triangular but has nonzero elements just below the main diagonal.
```python
upper_hessenberg = Matrix([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [0, 9, 10, 11],
    [0, 0, 12, 13]
])
print("Is upper Hessenberg?", MatrixProperties(upper_hessenberg).is_upper_hessenberg())
```

### 4. Orthogonal Matrix
A matrix is **Orthogonal** if its transpose is equal to its inverse.
```python
orthogonal = Matrix([
    [0, 1],
    [-1, 0]
])
print("Is orthogonal?", MatrixProperties(orthogonal).is_orthogonal())
```

### 5. Identity and Anti-Identity Matrices
#### Identity Matrix
A square matrix with ones on the diagonal and zeros elsewhere.
```python
identity = Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
print("Is identity?", MatrixProperties(identity).is_identity())
```
#### Anti-Identity Matrix
A matrix with ones on the anti-diagonal (opposite diagonal).
```python
anti_identity = Matrix([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])
print("Is anti-identity?", MatrixProperties(anti_identity).is_anti_identity())
```

### 6. Symmetric and Skew-Symmetric Matrices
#### Symmetric Matrix
A matrix is **Symmetric** if it equals its transpose.
```python
symmetric = Matrix([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
])
print("Is symmetric?", MatrixProperties(symmetric).is_symmetric())
```
#### Skew-Symmetric Matrix
A matrix is **Skew-Symmetric** if its transpose equals its negative.
```python
skew_symmetric = Matrix([
    [0, 2, -3],
    [-2, 0, 1],
    [3, -1, 0]
])
print("Is skew-symmetric?", MatrixProperties(skew_symmetric).is_skew_symmetric())
```





