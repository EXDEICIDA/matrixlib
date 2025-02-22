# Matrix Distance Metrics (Distance Class)
## Overview

The `Distance` class provides a comprehensive collection of distance metrics for matrices. This class implements various types of distance measures, including standard metrics (Euclidean, Manhattan, Cosine) and specialized ecological metrics (Bray-Curtis, Canberra). The implementation is pure Python and relies on the `Matrix` class from `matrixlib.core`.

## Class Architecture

### Class Attributes

```python
METRICS: Dict[str, Callable]
```
A class-level dictionary storing all registered distance metric functions.

### Core Methods

#### `__init__(self)`
Initializes the Distance class and registers all available metrics.

#### `register_metrics(cls) -> None`
Class method that populates the `METRICS` dictionary with all available distance functions.

**Registered Metrics:**
- euclidean
- manhattan
- cosine
- chebyshev
- minkowski

## Distance Metrics

### Standard Metrics

#### `euclidean(a: Matrix, b: Matrix) -> float`
Computes the Euclidean distance between two matrices.

**Mathematical Definition:**
```
d(A,B) = sqrt(sum_{i,j} (a_{ij} - b_{ij})²)
```

**Properties:**
- Metric space properties satisfied
- Translation and rotation invariant
- Most commonly used distance metric
- Sensitive to scale differences

#### `manhattan(a: Matrix, b: Matrix) -> float`
Calculates the Manhattan (L1) distance between matrices.

**Mathematical Definition:**
```
d(A,B) = sum_{i,j} |a_{ij} - b_{ij}|
```

**Properties:**
- Less sensitive to outliers than Euclidean
- Grid-based distance metric
- Not rotation invariant
- Useful in high-dimensional spaces

#### `cosine(a: Matrix, b: Matrix) -> float`
Computes the cosine distance between matrices.

**Mathematical Definition:**
```
d(A,B) = 1 - (sum_{i,j} a_{ij}b_{ij}) / (||A||_F ||B||_F)
```

**Properties:**
- Scale invariant
- Measures angular difference
- Range: [0, 2]
- Useful for high-dimensional data
- Not a proper metric (triangle inequality doesn't hold)

#### `chebyshev(a: Matrix, b: Matrix) -> float`
Calculates the Chebyshev (L∞) distance.

**Mathematical Definition:**
```
d(A,B) = max_{i,j} |a_{ij} - b_{ij}|
```

**Properties:**
- Maximum difference in any dimension
- Useful in games (chess king moves)
- Less sensitive to dimensionality
- Invariant under coordinate transforms

### Advanced Metrics

#### `minkowski(a: Matrix, b: Matrix, p: float = 2) -> float`
Computes the Minkowski distance of order p.

**Mathematical Definition:**
```
d(A,B) = (sum_{i,j} |a_{ij} - b_{ij}|^p)^(1/p)
```

**Properties:**
- Generalizes many distance metrics
- p = 1: Manhattan distance
- p = 2: Euclidean distance
- p = ∞: Chebyshev distance
- Metric space properties for p ≥ 1

#### `normalized_euclidean(a: Matrix, b: Matrix) -> float`
Calculates scale-invariant Euclidean distance.

**Mathematical Definition:**
```
d(A,B) = sqrt(sum_{i,j} (a_{ij} - b_{ij})² / n)
```
where n is the total number of elements.

**Properties:**
- Scale-invariant version of Euclidean distance
- Useful for comparing matrices of different sizes
- Normalized to number of elements

### Ecological Metrics

#### `bray_curtis(a: Matrix, b: Matrix) -> float`
Computes Bray-Curtis dissimilarity.

**Mathematical Definition:**
```
d(A,B) = sum_{i,j} |a_{ij} - b_{ij}| / sum_{i,j} |a_{ij} + b_{ij}|
```

**Properties:**
- Commonly used in ecology
- Bounded between [0,1]
- Sensitive to abundant elements
- Not a true metric (triangle inequality doesn't hold)

#### `canberra(a: Matrix, b: Matrix) -> float`
Calculates Canberra distance.

**Mathematical Definition:**
```
d(A,B) = sum_{i,j} |a_{ij} - b_{ij}| / (|a_{ij}| + |b_{ij}|)
```

**Properties:**
- Highly sensitive to changes near zero
- Useful for sparse data
- Normalized by magnitude
- Each element weighted equally

#### `jaccard(a: Matrix, b: Matrix, threshold: float = 0.0) -> float`
Computes Jaccard distance between matrices.

**Mathematical Definition:**
```
d(A,B) = 1 - |A ∩ B| / |A ∪ B|
```
where sets are defined by elements above threshold.

**Properties:**
- Binary similarity measure
- Range: [0,1]
- Useful for presence/absence data
- Ignores magnitude differences

#### `hamming(a: Matrix, b: Matrix) -> float`
Calculates Hamming distance between matrices.

**Mathematical Definition:**
```
d(A,B) = count(a_{ij} ≠ b_{ij})
```

**Properties:**
- Counts differing positions
- Integer-valued distance
- Useful for categorical data
- Simple to compute and interpret

## Utility Functions

### `validate_matrices(a: Matrix, b: Matrix) -> None`
Validates matrix inputs for compatibility.

**Checks:**
- Input type validation
- Shape compatibility
- Non-emptiness

### `flatten_matrix(matrix: Matrix) -> list`
Converts 2D matrix to 1D list.

### `calculate(cls, a: Matrix, b: Matrix, metric: str = 'euclidean', **kwargs) -> float`
Generic method to calculate any registered distance metric.

### `get_all_distances(self, a: Matrix, b: Matrix) -> Dict[str, float]`
Computes all available distance metrics between two matrices.

### `find_nearest_neighbor(self, target: Matrix, candidates: List[Matrix], metric: str = 'euclidean') -> Tuple[int, float]`
Finds closest matrix from a list of candidates.

## Usage Examples

```python
from matrixlib.core import Matrix
from distance import Distance

# Initialize distance calculator
dist = Distance()

# Create sample matrices
matrix_a = Matrix([[1, 2], [3, 4]])
matrix_b = Matrix([[2, 3], [4, 5]])

# Calculate specific distance
euc_dist = dist.calculate(matrix_a, matrix_b, 'euclidean')

# Calculate all distances
all_distances = dist.get_all_distances(matrix_a, matrix_b)

# Find nearest neighbor
candidates = [Matrix([[1, 1], [1, 1]]), Matrix([[4, 4], [4, 4]])]
nearest_idx, distance = dist.find_nearest_neighbor(matrix_a, candidates)
```

## Performance Considerations

1. **Computational Complexity**
   - Most metrics: O(n×m) for n×m matrices
   - Jaccard distance: O(n×m×log(n×m)) due to set operations
   - Nearest neighbor search: O(k×n×m) for k candidates

2. **Memory Usage**
   - Constant extra space for most metrics
   - Set-based metrics (Jaccard) use O(n×m) extra space

3. **Numerical Stability**
   - Cosine distance handles magnitude normalization
   - Canberra distance handles near-zero values
   - Normalized Euclidean handles scale differences

## Error Handling

The class implements robust error handling for:
- Invalid matrix inputs
- Shape mismatches
- Empty matrices
- Zero magnitude vectors in cosine distance
- Invalid p-values in Minkowski distance
- Division by zero in normalized metrics

## References

1. Cha, S. H. (2007). Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions
2. Deza, M. M., & Deza, E. (2009). Encyclopedia of Distances
3. Faith, D. P., Minchin, P. R., & Belbin, L. (1987). Compositional dissimilarity as a robust measure of ecological distance