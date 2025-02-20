# Matrix Library Documentation

A powerful Python library for matrix operations and analysis, supporting N-dimensional matrices with various mathematical operations and property checks.

## Table of Contents
1. [Matrix Class](#matrix-class)
2. [Matrix Operations](#matrix-operations)
3. [Matrix Properties](#matrix-properties)
4. [Examples](#examples)

## Matrix Class

The `Matrix` class provides the foundation for creating and manipulating matrices of any dimension.

### Constructor

```python
Matrix(data: List)
```

Creates a new Matrix instance from a nested list structure.

**Parameters:**
- `data`: A nested list representing the matrix data. Can be N-dimensional.

**Raises:**
- `ValueError`: If matrix data is empty or has inconsistent dimensions
- `TypeError`: If input data is not a list

### Methods

#### `_get_shape(data)`
Returns the dimensions of the matrix as a tuple.

```python
@staticmethod
def _get_shape(data)
```

**Returns:**
- Tuple representing matrix dimensions (e.g., (3,3) for 3x3 matrix)

#### `is_square_matrix()`
Checks if the matrix is square (equal number of rows and columns).

```python
def is_square_matrix() -> bool
```

**Returns:**
- `True` if matrix is square, `False` otherwise

## Matrix Operations

The `Operation` class provides static methods for common matrix operations.

### Addition

```python
@staticmethod
def add(a: Matrix, b: Matrix) -> Matrix
```

Performs element-wise addition of two matrices.

**Parameters:**
- `a`: First matrix
- `b`: Second matrix

**Returns:**
- New Matrix with element-wise sum

**Raises:**
- `ValueError`: If matrices have different shapes

### Subtraction

```python
@staticmethod
def subtract(a: Matrix, b: Matrix) -> Matrix
```

Performs element-wise subtraction of two matrices.

**Parameters:**
- `a`: First matrix
- `b`: Second matrix

**Returns:**
- New Matrix with element-wise difference

**Raises:**
- `ValueError`: If matrices have different shapes

### Multiplication

```python
@staticmethod
def multiply(a: Matrix, b: Matrix) -> Matrix
```

Performs matrix multiplication.

**Parameters:**
- `a`: First matrix
- `b`: Second matrix

**Returns:**
- New Matrix with multiplication result

**Raises:**
- `ValueError`: If inner dimensions don't match

### Scalar Multiplication

```python
@staticmethod
def scalar_multiply(matrix: Matrix, scalar: float) -> Matrix
```

Multiplies every element in the matrix by a scalar value.

**Parameters:**
- `matrix`: Input matrix
- `scalar`: Number to multiply by

**Returns:**
- New Matrix with scaled values

### Transpose

```python
@staticmethod
def transpose(a: Matrix) -> Matrix
```

Transposes the matrix (converts rows to columns and vice versa).

**Parameters:**
- `a`: Input matrix

**Returns:**
- New Matrix with transposed elements

### Determinant

```python
@staticmethod
def determinant(matrix: Matrix) -> float
```

Calculates the determinant of a square matrix.

**Parameters:**
- `matrix`: Square matrix

**Returns:**
- Determinant value

**Raises:**
- `ValueError`: If matrix is not square

### Adjugate

```python
@staticmethod
def adjugate(matrix: Matrix) -> Matrix
```

Calculates the adjugate (transpose of cofactor matrix).

**Parameters:**
- `matrix`: Square matrix

**Returns:**
- Adjugate matrix

### Inverse

```python
@staticmethod
def inverse(matrix: Matrix) -> Matrix
```

Calculates the inverse of a matrix.

**Parameters:**
- `matrix`: Square matrix

**Returns:**
- Inverse matrix

**Raises:**
- `ValueError`: If matrix is singular or not square

## Matrix Properties

The `MatrixProperties` class provides methods to check various matrix properties.

### Constructor

```python
MatrixProperties(matrix: Union[Matrix, List])
```

**Parameters:**
- `matrix`: Matrix object or nested list

### Property Methods

#### Triangular Matrices

```python
def is_upper_triangular() -> bool
def is_lower_triangular() -> bool
```

Check if matrix is upper or lower triangular.

#### Hessenberg Matrices

```python
def is_upper_hessenberg() -> bool
def is_lower_hessenberg() -> bool
```

Check if matrix is upper or lower Hessenberg form.

#### Orthogonal Matrix

```python
def is_orthogonal() -> bool
```

Check if matrix is orthogonal (A^T × A = I).

#### Hollow Matrix

```python
def is_hollow() -> bool
```

Check if matrix is hollow (zero diagonal elements).

## Examples

### Creating a Matrix

```python
# Create a 2x2 matrix
matrix = Matrix([
    [1, 2],
    [3, 4]
])

# Create a 3x3 matrix
matrix_3x3 = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```

### Basic Operations

```python
# Addition
result = Operation.add(matrix_a, matrix_b)

# Multiplication
product = Operation.multiply(matrix_a, matrix_b)

# Transpose
transposed = Operation.transpose(matrix)

# Determinant
det = Operation.determinant(matrix)
```

### Checking Properties

```python
props = MatrixProperties(matrix)

# Check if matrix is upper triangular
is_upper = props.is_upper_triangular()

# Check if matrix is orthogonal
is_orthogonal = props.is_orthogonal()
```