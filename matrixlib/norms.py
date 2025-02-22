from matrixlib.core import Matrix
import math
from typing import List, Union

"""This is a class for computing various matrix norms"""


class Norm:
    @staticmethod
    def frobenius(matrix: Matrix) -> float:
        """Compute the Frobenius norm (Euclidean norm for matrices)."""
        return math.sqrt(sum(x ** 2 for row in matrix.data for x in row))

    @staticmethod
    def max_norm(matrix: Matrix) -> float:
        """Compute the max norm (largest absolute row sum)."""
        return max(sum(abs(x) for x in row) for row in matrix.data)

    @staticmethod
    def _get_singular_values(matrix: Matrix) -> List[float]:
        """Helper method to compute singular values using power iteration method."""
        data = matrix.data
        n_rows = len(data)
        n_cols = len(data[0])

        # Create transpose of matrix
        transpose = [[data[j][i] for j in range(n_rows)] for i in range(n_cols)]

        # Compute a^t * a for eigenvalue computation
        ata = [[sum(transpose[i][k] * data[k][j]
                    for k in range(n_rows))
                for j in range(n_cols)]
               for i in range(n_cols)]

        def power_iteration(matrix: List[List[float]], max_iter: int = 100) -> float:
            n = len(matrix)
            # Start with random vector
            vector = [1 / math.sqrt(n)] * n

            for _ in range(max_iter):
                # Matrix-vector multiplication
                new_vector = [sum(matrix[i][j] * vector[j]
                                  for j in range(n))
                              for i in range(n)]

                # Normalize
                norm = math.sqrt(sum(x * x for x in new_vector))
                vector = [x / norm for x in new_vector]

            # Compute eigenvalue
            return math.sqrt(abs(sum(sum(matrix[i][j] * vector[j]
                                         for j in range(n)) * vector[i]
                                     for i in range(n))))

        singular_values = []
        remaining_matrix = [row[:] for row in ata]

        # Find top singular values
        for _ in range(min(n_rows, n_cols)):
            sigma = power_iteration(remaining_matrix)
            singular_values.append(sigma)

            # Deflate matrix for next iteration if needed
            if len(singular_values) < min(n_rows, n_cols):
                # This is a simplified deflation - for complete accuracy,
                # you might want to implement full Gram-Schmidt orthogonalization
                pass

        return singular_values

    @staticmethod
    def spectral_norm(matrix: Matrix) -> float:
        """Compute the spectral norm (maximum singular value)."""
        singular_values = Norm._get_singular_values(matrix)
        return math.sqrt(singular_values[0]) if singular_values else 0.0

    @staticmethod
    def nuclear_norm(matrix: Matrix) -> float:
        """Compute the nuclear norm (sum of singular values)."""
        singular_values = Norm._get_singular_values(matrix)
        return sum(math.sqrt(sv) for sv in singular_values)

    @staticmethod
    def p_norm(matrix: Matrix, p: Union[int, float] = 2) -> float:
        """Compute the entry-wise p-norm."""
        if p < 1:
            raise ValueError("p must be greater than or equal to 1")
        return sum(abs(x) ** p for row in matrix.data
                   for x in row) ** (1 / p)

    @staticmethod
    def induced_norm(matrix: Matrix, p: Union[int, float] = 2) -> float:
        """Compute the induced p-norm."""
        if p == 1:
            # Maximum absolute column sum
            n_rows = len(matrix.data)
            n_cols = len(matrix.data[0])
            return max(sum(abs(matrix.data[i][j])
                           for i in range(n_rows))
                       for j in range(n_cols))
        elif p == 2:
            return Norm.spectral_norm(matrix)
        elif p == float('inf'):
            return Norm.max_norm(matrix)
        else:
            raise ValueError("p must be 1, 2, or inf for induced norms")

    @staticmethod
    def _matrix_multiply(a: List[List[float]],
                         b: List[List[float]]) -> List[List[float]]:
        """Helper method for matrix multiplication."""
        n = len(a)
        m = len(b[0])
        p = len(b)
        return [[sum(a[i][k] * b[k][j] for k in range(p))
                 for j in range(m)]
                for i in range(n)]

    @staticmethod
    def _matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
        """Helper method to compute matrix inverse using Gauss-Jordan."""
        n = len(matrix)
        # Augment with identity matrix
        augmented = [row[:] + [1 if i == j else 0 for j in range(n)]
                     for i, row in enumerate(matrix)]

        # Gauss-Jordan elimination
        for i in range(n):
            # Find pivot
            pivot = augmented[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular")

            # Divide row by pivot
            augmented[i] = [x / pivot for x in augmented[i]]

            # Eliminate column
            for j in range(n):
                if i != j:
                    factor = augmented[j][i]
                    augmented[j] = [augmented[j][k] - factor * augmented[i][k]
                                    for k in range(2 * n)]

        # Extract inverse
        return [[augmented[i][j + n] for j in range(n)]
                for i in range(n)]

    @staticmethod
    def condition_number(matrix: Matrix, p: Union[int, float] = 2) -> float:
        """Compute the condition number with respect to a given norm."""
        try:
            inv_data = Norm._matrix_inverse(matrix.data)
            inv_matrix = Matrix(inv_data)
            return Norm.induced_norm(matrix, p) * Norm.induced_norm(inv_matrix, p)
        except ValueError:
            return float('inf')

    @staticmethod
    def schatten_norm(matrix: Matrix, p: Union[int, float] = 2) -> float:
        """
        Compute the Schatten p-norm (p-norm of singular values).
        For p=1 this is the nuclear norm
        For p=2 this is the Frobenius norm
        For p=inf this is the spectral norm
        """
        if p < 1:
            raise ValueError("p must be greater than or equal to 1")

        singular_values = Norm._get_singular_values(matrix)
        return sum(sv ** (p / 2) for sv in singular_values) ** (1 / p)

    @staticmethod
    def relative_norm(matrix1: Matrix, matrix2: Matrix, norm_type: str = 'frobenius') -> float:
        """
        Compute the relative norm between two matrices: ||A-B|| / ||B||
        """
        if matrix1.data[0].shape != matrix2.data[0].shape:
            raise ValueError("Matrices must have the same dimensions")

        diff_matrix = Matrix([[a - b for a, b in zip(row1, row2)]
                              for row1, row2 in zip(matrix1.data, matrix2.data)])

        norm_method = getattr(Norm, norm_type.lower(), None)
        if norm_method is None:
            raise ValueError(f"Unsupported norm type: {norm_type}")

        return norm_method(diff_matrix) / norm_method(matrix2)
