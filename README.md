```markdown
# matrixlib Documentation

## Introduction
**matrixlib** is a comprehensive Python library for advanced matrix operations, linear algebra computations, and data visualization. Designed for both educational and practical use, it provides recursive implementations for N-dimensional matrices and emphasizes clarity, extensibility, and numerical stability. Key features include:

- Matrix operations (addition, multiplication, determinants, inverses)
- Matrix factorizations (LU, SVD, QR, Jordan)
- Norm and distance metric computations
- Matrix property validation (symmetric, orthogonal, triangular)
- Visualization tools (heatmaps, 3D plots, networks)

---

## Core Classes

### 1. `Matrix` Class (Base Class)
*Handles matrix creation, storage, and basic manipulations.*  
*(Assumed to be implemented but not documented in provided files.)*

---

### 2. `Operation` Class  
#### Overview  
Provides fundamental and advanced matrix operations.  

#### Basic Operations  
| Method | Description | Mathematical Expression |  
|--------|-------------|-------------------------|  
| `add(a, b)` | Element-wise addition | \( C = A + B \) |  
| `subtract(a, b)` | Element-wise subtraction | \( C = A - B \) |  
| `multiply(a, b)` | Matrix multiplication | \( C = A \times B \) |  
| `transpose(a)` | Transpose matrix | \( C = A^T \) |  

#### Advanced Operations  
| Method | Description | Formula |  
|--------|-------------|---------|  
| `determinant(matrix)` | Recursive Laplace expansion | \( |A| = \sum_{j} (-1)^{i+j} a_{ij} M_{ij} \) |  
| `inverse(matrix)` | Adjugate method | \( A^{-1} = \frac{1}{|A|} \text{adj}(A) \) |  
| `trace(matrix)` | Sum of diagonal elements | \( \text{tr}(A) = \sum_{i} a_{ii} \) |  

#### Example:  
```python  
A = Matrix([[1, 2], [3, 4]])  
B = Operation.inverse(A)  # Inverse of A  
```

---

### 3. `Norm` Class  
#### Overview  
Computes matrix norms and related metrics.  

#### Key Methods  
| Method | Description | Formula |  
|--------|-------------|---------|  
| `frobenius(matrix)` | Frobenius norm | \( ||A||_F = \sqrt{\sum_{i,j} a_{ij}^2} \) |  
| `spectral_norm(matrix)` | Largest singular value | \( ||A||_2 = \sqrt{\lambda_{\text{max}}(A^T A)} \) |  
| `nuclear_norm(matrix)` | Sum of singular values | \( ||A||_* = \sum_i \sigma_i \) |  

#### Example:  
```python  
matrix = Matrix([[1, 2], [3, 4]])  
frob = Norm.frobenius(matrix)  # ≈ 5.477  
```

---

### 4. `MatrixProperties` Class  
#### Overview  
Validates matrix types and properties.  

#### Key Checks  
| Method | Matrix Type | Example |  
|--------|-------------|---------|  
| `is_upper_triangular()` | Upper triangular | All elements below diagonal are zero |  
| `is_orthogonal()` | Orthogonal | \( Q^T = Q^{-1} \) |  
| `is_symmetric()` | Symmetric | \( A = A^T \) |  

#### Example:  
```python  
sym_matrix = Matrix([[1, 2], [2, 3]])  
print(MatrixProperties(sym_matrix).is_symmetric())  # True  
```

---

### 5. `Factorization` Class  
#### Overview  
Implements matrix decomposition techniques.  

#### Methods  
| Method | Description | Formula |  
|--------|-------------|---------|  
| `lu_decomposition(matrix)` | LU decomposition | \( A = LU \) |  
| `svd(matrix)` | Singular Value Decomposition | \( A = U S V^T \) |  
| `qr_decomposition(matrix)` | QR factorization | \( A = QR \) |  

#### Example:  
```python  
A = Matrix([[4, 3], [6, 3]])  
L, U = Factorization.lu_decomposition(A)  
```

---

### 6. `Distance` Class  
#### Overview  
Computes distance metrics between matrices.  

#### Metrics  
| Method | Description | Formula |  
|--------|-------------|---------|  
| `euclidean(a, b)` | Euclidean distance | \( \sqrt{\sum (a_{ij} - b_{ij})^2} \) |  
| `cosine(a, b)` | Cosine similarity | \( 1 - \frac{\sum a_{ij}b_{ij}}{||A||_F ||B||_F} \) |  

#### Example:  
```python  
dist = Distance().calculate(A, B, metric="euclidean")  
```

---

### 7. `Visualize` Class  
#### Overview  
Creates visual representations of matrices.  

#### Methods  
| Method | Visualization Type |  
|--------|--------------------|  
| `to_heatmap()` | Color-coded heatmap |  
| `to_graph()` | Network graph |  
| `to_surface3d()` | 3D surface plot |  

#### Example:  
```python  
Visualize.to_heatmap(Matrix([[1, 2], [3, 4]]), title="Sample Heatmap")  
```

---

## Advanced Topics

### Performance Considerations  
| Operation | Time Complexity |  
|-----------|-----------------|  
| Matrix Multiplication | \( O(n^3) \) |  
| LU Decomposition | \( O(n^3) \) |  
| Spectral Norm | \( O(kn^2) \) (iterative) |  

### Error Handling  
- **Dimension Mismatch**: Raised for incompatible matrix shapes.  
- **Singular Matrix**: Raised during inversion if determinant is zero.  

### Best Practices  
1. Validate inputs using `validate_dimensions()`.  
2. Use tolerance checks for floating-point comparisons.  
3. Precompute expensive operations (e.g., determinants).  

---

## Comprehensive Example  
```python  
# Create matrices  
A = Matrix([[1, 2], [3, 4]])  
B = Matrix([[5, 6], [7, 8]])  

# Perform operations  
C = Operation.add(A, B)  
D = Factorization.lu_decomposition(C)  
Norm.spectral_norm(D[0])  

# Visualize  
Visualize.to_heatmap(C)  
```

---

## References  
1. Golub & Van Loan, *Matrix Computations*  
2. Horn & Johnson, *Matrix Analysis*  
3. Strang, *Introduction to Linear Algebra*  

## Appendix  
### Dependencies  
- `numpy`, `matplotlib`, `networkx`, `plotly`  

### Contributing Guidelines  
- Follow existing method patterns.  
- Include input validation and error handling.  
``` 

This documentation consolidates all provided class files into a structured, unified format while preserving mathematical rigor and practical examples.
