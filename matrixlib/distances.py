from matrixlib.core import Matrix
import math
from typing import Callable, Dict, Optional, List, Tuple


class Distance:
    """A class that calculates various distance metrics between matrices.
    Pure implementation without external dependencies."""

    # Dictionary of available distance metrics
    METRICS: Dict[str, Callable] = {}

    def __init__(self):
        # Register all distance metrics
        self.register_metrics()

    @classmethod
    def register_metrics(cls) -> None:
        """Register all available distance metrics."""
        cls.METRICS = {
            'euclidean': cls.euclidean,
            'manhattan': cls.manhattan,
            'cosine': cls.cosine,
            'chebyshev': cls.chebyshev,
            'minkowski': cls.minkowski
        }

    @staticmethod
    def validate_matrices(a: Matrix, b: Matrix) -> None:
        """Validate that matrices are compatible for distance calculation."""
        if not isinstance(a, Matrix) or not isinstance(b, Matrix):
            raise TypeError("Inputs must be Matrix objects")
        if a.shape != b.shape:
            raise ValueError(f"Matrices must have same shape. Got {a.shape} and {b.shape}")
        if not a.data or not b.data:
            raise ValueError("Matrices cannot be empty")

    @staticmethod
    def flatten_matrix(matrix: Matrix) -> list:
        """Convert a matrix into a flattened list."""
        return [x for row in matrix.data for x in row]

    @classmethod
    def calculate(cls, a: Matrix, b: Matrix, metric: str = 'euclidean', **kwargs) -> float:
        """Calculate distance between matrices using specified metric."""
        if metric not in cls.METRICS:
            raise ValueError(f"Unsupported metric: {metric}. Available metrics: {list(cls.METRICS.keys())}")
        return cls.METRICS[metric](a, b, **kwargs)

    @staticmethod
    def euclidean(a: Matrix, b: Matrix) -> float:
        """Calculate Euclidean distance between two matrices."""
        Distance.validate_matrices(a, b)
        squared_diff_sum = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                squared_diff_sum += (x - y) ** 2
        return math.sqrt(squared_diff_sum)

    @staticmethod
    def manhattan(a: Matrix, b: Matrix) -> float:
        """Calculate Manhattan distance between two matrices."""
        Distance.validate_matrices(a, b)
        abs_diff_sum = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                abs_diff_sum += abs(x - y)
        return abs_diff_sum

    @staticmethod
    def cosine(a: Matrix, b: Matrix) -> float:
        """Calculate cosine distance between two matrices."""
        Distance.validate_matrices(a, b)

        # Calculate dot product and magnitudes
        dot_product = 0.0
        mag_a = 0.0
        mag_b = 0.0

        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                dot_product += x * y
                mag_a += x * x
                mag_b += y * y

        mag_a = math.sqrt(mag_a)
        mag_b = math.sqrt(mag_b)

        if mag_a == 0 or mag_b == 0:
            raise ValueError("Cannot calculate cosine distance for zero magnitude vector")

        return 1 - (dot_product / (mag_a * mag_b))

    @staticmethod
    def chebyshev(a: Matrix, b: Matrix) -> float:
        """Calculate Chebyshev distance between two matrices."""
        Distance.validate_matrices(a, b)
        max_diff = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                max_diff = max(max_diff, abs(x - y))
        return max_diff

    @staticmethod
    def minkowski(a: Matrix, b: Matrix, p: float = 2) -> float:
        """Calculate Minkowski distance between two matrices."""
        Distance.validate_matrices(a, b)
        if p <= 0:
            raise ValueError("Order p must be positive")

        sum_diff = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                sum_diff += abs(x - y) ** p
        return sum_diff ** (1 / p)

    def get_all_distances(self, a: Matrix, b: Matrix) -> Dict[str, float]:
        """Calculate all available distance metrics between two matrices."""
        results = {}
        for metric_name in self.METRICS:
            try:
                results[metric_name] = self.calculate(a, b, metric_name)
            except Exception as e:
                results[metric_name] = f"Error: {str(e)}"
        return results

    def find_nearest_neighbor(self, target: Matrix, candidates: List[Matrix],
                              metric: str = 'euclidean') -> Tuple[int, float]:
        """Find the nearest neighbor to target matrix from a list of candidates."""
        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        min_distance = float('inf')
        min_index = -1

        for i, candidate in enumerate(candidates):
            distance = self.calculate(target, candidate, metric)
            if distance < min_distance:
                min_distance = distance
                min_index = i

        return min_index, min_distance

    @staticmethod
    def normalized_euclidean(a: Matrix, b: Matrix) -> float:
        """Calculate normalized Euclidean distance between two matrices.
        Normalizes by the number of elements to make it scale-invariant."""
        Distance.validate_matrices(a, b)
        squared_diff_sum = 0.0
        n_elements = len(a.data) * len(a.data[0])
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                squared_diff_sum += (x - y) ** 2
        return math.sqrt(squared_diff_sum / n_elements)

    @staticmethod
    def bray_curtis(a: Matrix, b: Matrix) -> float:
        """Calculate Bray-Curtis dissimilarity between two matrices.
        Commonly used in ecology and community composition studies."""
        Distance.validate_matrices(a, b)
        numerator = 0.0
        denominator = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                numerator += abs(x - y)
                denominator += abs(x + y)
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def canberra(a: Matrix, b: Matrix) -> float:
        """Calculate Canberra distance between two matrices.
        More sensitive to differences near zero."""
        Distance.validate_matrices(a, b)
        distance = 0.0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                denominator = abs(x) + abs(y)
                if denominator > 0:
                    distance += abs(x - y) / denominator
        return distance

    @staticmethod
    def jaccard(a: Matrix, b: Matrix, threshold: float = 0.0) -> float:
        """Calculate Jaccard distance between two matrices.
        Treats matrix elements as sets based on presence (>threshold) or absence."""
        Distance.validate_matrices(a, b)
        set_a = set((i, j) for i, row in enumerate(a.data)
                    for j, val in enumerate(row) if val > threshold)
        set_b = set((i, j) for i, row in enumerate(b.data)
                    for j, val in enumerate(row) if val > threshold)

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return 1 - (intersection / union if union > 0 else 0)

    @staticmethod
    def hamming(a: Matrix, b: Matrix) -> float:
        """Calculate Hamming distance between two matrices.
        Counts the number of positions at which corresponding elements differ."""
        Distance.validate_matrices(a, b)
        diff_count = 0
        for row_a, row_b in zip(a.data, b.data):
            for x, y in zip(row_a, row_b):
                if x != y:
                    diff_count += 1
        return float(diff_count)
