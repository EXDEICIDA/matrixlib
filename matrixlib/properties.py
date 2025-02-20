from matrixlib.core import Matrix
from matrixlib.operations import Operation
from math import comb


# A class that consists of a class for classifying special matrices.
class MatrixProperties:
    def __init__(self, matrix: Matrix):
        """Initialize with a Matrix object or a list"""
        if isinstance(matrix, list):
            matrix = Matrix(matrix)
        elif not isinstance(matrix, Matrix):
            raise TypeError("Expected an Matrix object or a list")
        self.matrix = matrix

    # Upper Triangular Matrix Method
    def is_upper_triangular(self) -> bool:
        """
        Check if the matrix is upper triangular.
        Returns True if all elements below the main diagonal are zero.
        """
        if not self.matrix.is_square_matrix():
            return False

        # Get the matrix data
        data = self.matrix.data
        n = len(data)

        # elements below main diagonal
        for i in range(n):
            for j in range(i):
                if data[i][j] != 0:
                    return False
        return True

    def is_lower_triangular(self) -> bool:
        """
        Check if the matrix is lower triangular.
        Returns True if all elements above the main diagonal are zero.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):
            for j in range(i + 1, n):
                if data[i][j] != 0:
                    return False
        return True

    # function to check both upper and lower Hessenberg
    def _is_hessenberg(self, upper: bool) -> bool:
        """
        Generic method to check if the matrix is Hessenberg.
        If upper is True, checks for upper Hessenberg.
        If upper is False, checks for lower Hessenberg.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        # Checking upper Hessenberg (zeros below the first sub-diagonal)
        if upper:
            for i in range(2, n):  # Start from second row
                for j in range(i - 1):
                    if data[i][j] != 0:
                        return False
        # Checking lower Hessenberg (zeros above the first super-diagonal)
        else:
            for j in range(2, n):  # Start from second column
                for i in range(j - 1):
                    if data[i][j] != 0:
                        return False

        return True

    def is_upper_hessenberg(self) -> bool:
        """Check if the matrix is upper Hessenberg."""
        return self._is_hessenberg(upper=True)

    def is_lower_hessenberg(self) -> bool:
        """Check if the matrix is lower Hessenberg."""
        return self._is_hessenberg(upper=False)

    # Orthogonal Matrix Method
    def is_orthogonal(self) -> bool:
        """
        Check if the matrix is orthogonal.
        A matrix is orthogonal if A^T × A = I (identity matrix)
        Returns True if the matrix is orthogonal, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        # Get matrix dimensions
        n = len(self.matrix.data)

        # Calculate A^T × A
        transpose = Operation.transpose(self.matrix)
        product = Operation.multiply(transpose, self.matrix)

        # Check if product equals identity matrix
        for i in range(n):
            for j in range(n):
                # Should be 1 on diagonal
                if i == j and abs(product.data[i][j] - 1.0) > 1e-10:
                    return False
                # Should be 0 off diagonal
                elif i != j and abs(product.data[i][j]) > 1e-10:
                    return False

        return True

    def is_hollow(self) -> bool:
        """Check if the matrix is hollow.Return either true or false"""
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):
            if data[i][i] == 0:
                return False

        for i in range(n):
            for j in range(n):
                if i != j and data[i][j] != 0:
                    return False

        return True

    # Diagonal matrix is a square matrix in which all the
    # elements outside the main diagonal are zero, while the diagonal
    # elements can be any values (including zero).

    def is_diagonal(self) -> bool:
        """Checks if the matrix is diagonal.Returns either true or false"""
        if not self.matrix.is_square_matrix():
            return False
        data = self.matrix.data
        n = len(data)

        # Iterate over the dimensions
        for i in range(n):
            for j in range(n):
                if i != j and data[i][j] != 0:
                    return False

        return True

    # Definition: A diagonal matrix where all diagonal elements are 1.
    def is_identity(self) -> bool:
        """Check if the matrix is an identity matrix.
           An identity matrix has 1s on the main diagonal and 0s elsewhere.
        """

        if len(self.matrix.data) != len(self.matrix.data[0]):
            return False

        if not self.is_diagonal():
            return False

        data = self.matrix.data
        n = len(data)
        for i in range(n):
            if data[i][i] != 1:
                return False

        return True

    def is_anti_identity(self) -> bool:
        """Check if the matrix is an anti-identity matrix.
        An anti-identity matrix has 1s on the anti-diagonal (top-right to bottom-left)
        and 0s elsewhere.

        Returns:
            bool: True if the matrix is an anti-identity matrix, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):
            for j in range(n):
                # Check anti-diagonal elements (i + j = n - 1)
                if i + j == n - 1:
                    if data[i][j] != 1:
                        return False
                else:
                    if data[i][j] != 0:
                        return False

        return True

    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric.
           A matrix is symmetric if it equals its transpose (A[i][j] = A[j][i]).
          Returns True if the matrix is symmetric, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):
            for j in range(n):
                if data[i][j] != data[j][i]:
                    return False

        return True

    def is_skew_symmetric(self) -> bool:
        """
        Check if the matrix is skew-symmetric.
        A matrix is skew-symmetric if A[i][j] = -A[j][i] for all elements,
        which also implies that diagonal elements must be 0.

        Returns:
            bool: True if the matrix is skew-symmetric, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):

            if data[i][i] != 0:
                return False

            for j in range(i + 1, n):  # Only need to check upper triangle
                if data[i][j] != -data[j][i]:
                    return False

        return True

    def is_zero_matrix(self) -> bool:
        """
        Check if all elements in the matrix are 0.
        Works for both square and rectangular matrices.
        """
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0])  # Get number of columns from first row

        for i in range(rows):
            for j in range(cols):
                if data[i][j] != 0:
                    return False

        return True

    def is_ones_matrix(self) -> bool:

        """Definition: A matrix where all elements are 1."""
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0])  # Get number of columns from first row
        for i in range(rows):
            for j in range(cols):
                if data[i][j] != 1:
                    return False

        return True

    def is_sparse(self) -> bool:
        """Check if the matrix is sparse.
           A matrix is considered sparse if more than half of its elements are 0.
        """
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0]) if data else 0
        zero_count = sum(1 for row in data for value in row if value == 0)

        return zero_count > (rows * cols) / 2

    def is_pascal(self) -> bool:
        """Check if the matrix is a Pascal matrix.

        A Pascal matrix is symmetric and follows Pascal's triangle:
        Element A[i][j] = C(i, j) (combination formula: C(n, k) = n! / (k! * (n-k)!))

        Returns:
            True if the matrix is Pascal, False otherwise.
        """
        if not self.is_symmetric():
            return False

        data = self.matrix.data
        rows = len(data)

        for i in range(rows):
            for j in range(i + 1):  # Only check upper/lower triangle (symmetry)
                if data[i][j] != comb(i, j):  # Check if A[i][j] follows Pascal's rule
                    return False

        return True

    def is_hermitian(self) -> bool:
        """Check if the matrix is Hermitian.

        A Hermitian matrix satisfies: A[i][j] = conjugate(A[j][i]).

        Returns:
            True if the matrix is Hermitian, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False  # Must be square

        data = self.matrix.data
        rows = len(data)

        for i in range(rows):
            for j in range(rows):  # Since it's square, rows == cols
                if data[i][j] != Operation.conjugate(data[j][i]):
                    return False

        return True

    def is_singular(self) -> bool:
        """
        Check if the matrix is singular.
        A singular matrix is a square matrix with a determinant of 0.

        Returns:
            bool: True if the matrix is singular, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        return Operation.determinant(self.matrix) == 0

    def is_toeplitz(self) -> bool:
        """Check if the matrix is a Toeplitz matrix.

    A Toeplitz matrix is one where each descending diagonal from left to right
    contains the same element, i.e., data[i][j] = data[i-1][j-1] for all valid i,j.

    Returns:
        bool: True if the matrix is Toeplitz, False otherwise.
        """
        data = self.matrix.data
        if not data or not data[0]:  # Check for empty matrix
            return False

        rows = len(data)
        cols = len(data[0])

        for i in range(1, rows):
            for j in range(1, cols):
                if data[i][j] != data[i - 1][j - 1]:
                    return False

        return True

    def is_invertible(self) -> bool:
        """
        Check if the matrix is invertible.
        A matrix is invertible if and only if it is square and has a non-zero determinant.

        Returns:
            bool: True if the matrix is invertible, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        determinant = Operation.determinant(self.matrix)
        return determinant != 0

    def is_banded(self) -> bool:
        """
        Check if the matrix is banded.
        A banded matrix has non-zero elements confined to a diagonal band,
        including the main diagonal and adjacent diagonals.

        Returns:
            bool: True if the matrix is banded, False otherwise.
        """
        if not self.matrix.data or not self.matrix.data[0]:
            return False

        data = self.matrix.data
        rows = len(data)
        cols = len(data[0])

        upper_bandwidth = 0
        lower_bandwidth = 0

        # Find the upper bandwidth (distance from the main diagonal where elements are non-zero)
        for i in range(rows):
            for j in range(i + 1, cols):
                if data[i][j] != 0:
                    upper_bandwidth = max(upper_bandwidth, j - i)

        # Find the lower bandwidth (distance from the main diagonal where elements are non-zero)
        for i in range(rows):
            for j in range(i):
                if data[i][j] != 0:
                    lower_bandwidth = max(lower_bandwidth, i - j)

        # Check that all elements outside the band are zero
        for i in range(rows):
            for j in range(cols):
                distance = abs(i - j)
                if distance > upper_bandwidth and distance > lower_bandwidth:
                    if data[i][j] != 0:
                        return False

        return True

    def is_permutation(self) -> bool:
        """
        Check if the matrix is a permutation matrix.
        A permutation matrix is a square matrix where:
        1. Each row and column contains exactly one '1'
        2. All other elements are '0'
        3. The matrix is orthogonal (A^T = A^-1)

        Returns:
            bool: True if the matrix is a permutation matrix, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        # Use sets to track positions of 1s in rows and columns
        row_ones = set()
        col_ones = set()

        for i in range(n):
            for j in range(n):
                value = data[i][j]
                # Check if value is not 0 or 1
                if value not in (0, 1):
                    return False

                if value == 1:
                    # If we already found a 1 in this row or column, not a permutation
                    if i in row_ones or j in col_ones:
                        return False
                    row_ones.add(i)
                    col_ones.add(j)

        # Check if we found exactly one 1 in each row and column
        return len(row_ones) == n and len(col_ones) == n

    def is_hadamard(self) -> bool:
        """Check if matrix is Hadamard.
        A Hadamard matrix is a square matrix whose entries are 1 or -1,
        and whose rows are mutually orthogonal."""
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        # Check if all entries are 1 or -1
        for row in data:
            for val in row:
                if val not in (1, -1):
                    return False

        # Check orthogonality of rows
        for i in range(n):
            for j in range(i + 1, n):
                if sum(data[i][k] * data[j][k] for k in range(n)) != 0:
                    return False

        return True

    def is_circulant(self) -> bool:
        """Check if matrix is circulant.
        A circulant matrix is where each row is rotated one element
        to the right relative to the previous row."""
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)
        first_row = data[0]

        for i in range(1, n):
            # Check if current row is a rotation of first row
            if data[i] != first_row[-i:] + first_row[:-i]:
                return False

        return True

    def is_hilbert(self) -> bool:
        """Check if matrix is a Hilbert matrix.
        A Hilbert matrix has entries H[i,j] = 1/(i + j - 1)."""
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0])

        for i in range(rows):
            for j in range(cols):
                if abs(data[i][j] - 1 / (i + j + 1)) > 1e-10:
                    return False

        return True

    def is_vandermonde(self) -> bool:
        """Check if matrix is a Vandermonde matrix.
        A Vandermonde matrix has terms of geometric progression in each row."""
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0])

        for i in range(rows):
            if data[i][0] == 0:
                return False
            for j in range(1, cols):
                if abs(data[i][j] - data[i][0] ** j) > 1e-10:
                    return False

        return True

    def is_unitary(self) -> bool:
        """
          Checks if the matrix is unitary.
          A matrix is unitary if U^dagger * U = I, where U^dagger is the conjugate transpose
          of U and I is the identity matrix.

         Returns:
         bool: True if the matrix is unitary, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        # Calculate the conjugate transpose (U^dagger)
        conjugate_matrix = Operation.conjugate(self.matrix)
        conjugate_transpose = Operation.transpose(conjugate_matrix)

        # Multiply U^dagger * U
        product = Operation.multiply(conjugate_transpose, self.matrix)

        # Check if the result is an identity matrix
        n = len(self.matrix.data)
        for i in range(n):
            for j in range(n):
                # Should be 1 on diagonal
                if i == j and abs(product.data[i][j] - 1.0) > 1e-10:
                    return False
                # Should be 0 off diagonal
                elif i != j and abs(product.data[i][j]) > 1e-10:
                    return False

        return True

    def is_block(self) -> bool:
        """
        Check if the matrix is a block matrix.
        A matrix is considered a block matrix if it can be logically
        divided into smaller sub-matrices with consistent zero patterns.

        Returns:
            bool: True if the matrix is a block matrix, False otherwise.
        """
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0

        if rows < 2 or cols < 2:
            return False

        zero_rows = set()
        zero_cols = set()

        # Identify zero rows and zero columns
        for i in range(rows):
            if all(data[i][j] == 0 for j in range(cols)):
                zero_rows.add(i)

        for j in range(cols):
            if all(data[i][j] == 0 for i in range(rows)):
                zero_cols.add(j)

        if not zero_rows and not zero_cols:
            return False  # No divisions detected, so not a block matrix

        h_blocks = [0] + sorted(zero_rows) + [rows]
        v_blocks = [0] + sorted(zero_cols) + [cols]

        # Check if blocks contain non-zero elements
        for i in range(len(h_blocks) - 1):
            for j in range(len(v_blocks) - 1):
                start_row, end_row = h_blocks[i], h_blocks[i + 1]
                start_col, end_col = v_blocks[j], v_blocks[j + 1]

                # Skip empty blocks (entirely zero)
                if all(data[r][c] == 0 for r in range(start_row, end_row) for c in range(start_col, end_col)):
                    continue

                return True  # At least one valid block with non-zero elements

        return False  # No valid block was found

    def is_stochastic(self) -> bool:
        """Definition: A square matrix where all entries
         are non-negative, and each row sums to 1."""
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        for i in range(n):
            # Check if any element is negative
            for j in range(n):
                if data[i][j] < 0:
                    return False

            # Check if row sum equals 1
            row_sum = sum(data[i][j] for j in range(n))
            if abs(row_sum - 1.0) > 1e-10:  # Use small epsilon for floating-point comparison
                return False

        return True

    def is_hankel(self) -> bool:
        """
        Check if the matrix is a Hankel matrix.
        A Hankel matrix has constant values along anti-diagonals, where H[i, j] = H[i+1, j-1]
        for all valid indices.

        Returns:
            bool: True if the matrix is a Hankel matrix, False otherwise.
        """
        data = self.matrix.data
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0

        # Checks anti-diagonal property
        for i in range(rows - 1):
            for j in range(cols - 1):
                if data[i][j + 1] != data[i + 1][j]:
                    return False

        return True

    def is_companion(self) -> bool:
        """
        Check if the matrix is a companion matrix.
        A companion matrix is a square matrix of the form:
        [0  1  0  0 ...  0]
        [0  0  1  0 ...  0]
        [0  0  0  1 ...  0]
        [...             0]
        [0  0  0  0 ...  1]
        [an an-1 ... a2 a1]
        where the last row contains coefficients and there's a super-diagonal of 1's.

        Returns:
            bool: True if the matrix is a companion matrix, False otherwise.
        """
        if not self.matrix.is_square_matrix():
            return False

        data = self.matrix.data
        n = len(data)

        # Matrix must be at least 2x2
        if n < 2:
            return False

        # Check super-diagonal of 1's
        for i in range(n - 1):
            for j in range(n):
                if j == i + 1:
                    # Check super-diagonal elements are 1
                    if data[i][j] != 1:
                        return False
                elif i != n - 1:  # Skip last row in this check
                    # Check all other elements (except last row) are 0
                    if data[i][j] != 0:
                        return False

        # Check last row contains any values (coefficients)
        # We don't check specific values as they depend on the polynomial
        last_row_all_zero = all(data[n - 1][j] == 0 for j in range(n))
        if last_row_all_zero:
            return False

        return True
