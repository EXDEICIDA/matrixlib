from matrixlib.core import Matrix


class Operation:

    # Matrix addition => A+B = [aij] + [bij] = [aij + bij]
    @staticmethod
    def add(a: Matrix, b: Matrix) -> Matrix:
        """Element-wise addition of two matrices."""
        if a.shape != b.shape:
            raise ValueError("Matrices must have the same shape for addition.")

        def recursive_add(ai, bi):
            if isinstance(ai, list):
                return [recursive_add(x, y) for x, y in zip(ai, bi)]
            return ai + bi

        return Matrix(recursive_add(a.data, b.data))

    # Matrix Subtraction => A - B = [aij] - [bij] = [aij - bij]

    @staticmethod
    def subtract(a: Matrix, b: Matrix) -> Matrix:
        """Element-wise subtraction of two matrices."""
        if a.shape != b.shape:
            raise ValueError("Matrices must have the same shape for subtraction.")

        def recursive_subtract(ai, bi):
            if isinstance(ai, list):
                return [recursive_subtract(x, y) for x, y in zip(ai, bi)]
            return ai - bi

        return Matrix(recursive_subtract(a.data, b.data))

    # Matrix Transposition => A^T = [a_ji]
    @staticmethod
    def transpose(a: Matrix) -> Matrix:
        """Transpose method that works for any N-dimensional matrix."""
        return Matrix(Operation._transpose_recursive(a.data))

    @staticmethod
    def _transpose_recursive(data):
        """Recursively transpose the matrix."""
        if isinstance(data[0], list):
            # Recurse through rows and transform into columns
            return [Operation._transpose_recursive([data[j][i] for j in range(len(data))]) for i in range(len(data[0]))]
        else:
            # If we reach a scalar or 1D list, return it as-is
            return data

    # Matrix multiplication => A×B = [aij]×[bij] = [Σ(aik × bkj)]
    @staticmethod
    def multiply(a: Matrix, b: Matrix) -> Matrix:
        """Generalized matrix multiplication for any dimensional matrix."""
        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError("Multiplication requires at least 2D matrices.")

        if a.shape[-1] != b.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication.")

        def recursive_multiply(mat1, mat2):
            if isinstance(mat1[0], list) and isinstance(mat2[0], list):
                return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2))) for j in range(len(mat2[0]))] for i in
                        range(len(mat1))]
            else:
                return [x * y for x, y in zip(mat1, mat2)]  # Element-wise multiplication for 1D

        return Matrix(recursive_multiply(a.data, b.data))

    # Scalar matrix multiplication => c×A = c×[aij] = [c×aij]
    @staticmethod
    def scalar_multiply(matrix: Matrix, scalar: float) -> Matrix:
        """Multiply every element of the matrix by a scalar value."""

        def recursive_scalar_multiply(data, factor):
            if isinstance(data, list):
                return [recursive_scalar_multiply(sub, factor) for sub in data]
            return data * factor  # Multiply element by scalar

        return Matrix(recursive_scalar_multiply(matrix.data, scalar))

    # Determinant of  a matrix => det(A) = |A| = Σ(±a1j × M1j)
    @staticmethod
    def determinant(matrix: Matrix) -> float:
        """Calculate determinant of a square matrix."""
        if not matrix.is_square_matrix():
            raise ValueError("Determinant can only be calculated for square matrices.")

        # Base case: 1x1 matrix
        if len(matrix.data) == 1:
            return matrix.data[0][0]

        # Base case: 2x2 matrix
        if len(matrix.data) == 2:
            return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]

        # Recursive case: n x n matrix (Laplace expansion)
        det = 0
        for col in range(len(matrix.data)):
            sub_matrix = [row[:col] + row[col + 1:] for row in matrix.data[1:]]  # Get minor matrix
            det += ((-1) ** col) * matrix.data[0][col] * Operation.determinant(Matrix(sub_matrix))

        return det

    # Matrix adjugate or  transpose of its cofactor matrix.
    @staticmethod
    def adjugate(matrix: Matrix) -> Matrix:
        """Returns the  adjugate of the matrix."""
        cofactors = []
        for i in range(len(matrix.data)):
            cofactor_row = []
            for j in range(len(matrix.data[i])):
                # Calculate the cofactor of element (i, j)
                minor_matrix = [row[:j] + row[j + 1:] for row in matrix.data[:i] + matrix.data[i + 1:]]
                cofactor_row.append(((-1) ** (i + j)) * Operation.determinant(Matrix(minor_matrix)))
            cofactors.append(cofactor_row)

        # Step 3: Transpose the cofactor matrix to get  adjoin
        adjugate = Operation.transpose(Matrix(cofactors))
        return adjugate

    # Matrix inversion => A^-1 = 1/det(A) * adj(A)

    @staticmethod
    def inverse(matrix: Matrix) -> Matrix:
        """A method that calculates and returns inverse of a matrix"""
        if not matrix.is_square_matrix():
            raise ValueError("Inverse can only be calculated for square matrices.")

        determinant = Operation.determinant(matrix)  # Storing determinant
        if determinant == 0:
            raise ValueError("Matrix is singular cannot be inverted.")

        adjugate = Operation.adjugate(matrix)
        return Operation.scalar_multiply(adjugate, 1 / determinant)

    @staticmethod
    def conjugate(matrix: Matrix) -> Matrix:
        """Returns the element-wise complex conjugate of the matrix."""

        def recursive_conjugate(data):
            if isinstance(data, list):
                return [recursive_conjugate(sub) for sub in data]
            return data.conjugate()  # Apply complex conjugation to each element

        return Matrix(recursive_conjugate(matrix.data))

    # Dot Product Method
    @staticmethod
    def dot_product(a: Matrix, b: Matrix) -> Matrix:
        """
        Compute the dot product of two matrices. Supports N-dimensional matrices.

        Args:
            a (Matrix): First matrix.
            b (Matrix): Second matrix.

        Returns:
            Matrix: Resulting dot product matrix.

        Raises:
            ValueError: If the inner dimensions do not match.
        """
        if a.shape[-1] != b.shape[0]:
            raise ValueError("Inner dimensions must match for dot product.")

        def recursive_dot(mat1, mat2):
            """Recursively calculates the dot product for N-dimensional matrices."""
            if isinstance(mat1[0], list) and isinstance(mat2[0], list):
                return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2))) for j in range(len(mat2[0]))] for i in
                        range(len(mat1))]
            else:
                return sum(x * y for x, y in zip(mat1, mat2))  # Base case: 1D dot product

        return Matrix(recursive_dot(a.data, b.data))

    @staticmethod
    def trace(matrix: Matrix) -> float:
        """Compute the trace of a square matrix."""
        if not matrix.is_square_matrix():
            raise ValueError("Trace can only be calculated for square matrices.")
        return sum(matrix.data[i][i] for i in range(len(matrix.data)))

    @staticmethod
    def hadamard_product(a: Matrix, b: Matrix) -> Matrix:
        """Element-wise multiplication (Hadamard product) of two matrices."""
        if a.shape != b.shape:
            raise ValueError("Matrices must have the same shape for element-wise multiplication.")

        def recursive_hadamard(a_data, b_data):
            if isinstance(a_data, list):
                return [recursive_hadamard(x, y) for x, y in zip(a_data, b_data)]
            return a_data * b_data  # Element-wise multiplication

        return Matrix(recursive_hadamard(a.data, b.data))

    @staticmethod
    def swap_rows(matrix: Matrix, row1: int, row2: int) -> Matrix:
        """Swap two rows of the matrix."""
        matrix.data[row1], matrix.data[row2] = matrix.data[row2], matrix.data[row1]
        return matrix

    @staticmethod
    def scale_row(matrix: Matrix, row: int, scalar: float) -> Matrix:
        """Scale a row by a given scalar."""
        matrix.data[row] = [x * scalar for x in matrix.data[row]]
        return matrix

    @staticmethod
    def matrix_exponentiation(matrix: Matrix, n: int) -> Matrix:
        """Exponentiate a matrix (A^n)."""
        if not matrix.is_square_matrix():
            raise ValueError("Matrix exponentiation can only be done on square matrices.")
        result = matrix  # Start with matrix as the base
        for _ in range(n - 1):
            result = Operation.multiply(result, matrix)
        return result

    @staticmethod
    def correlation(matrix: Matrix) -> Matrix:
        """
        Compute the correlation matrix (Pearson correlation coefficient) for a given matrix.

        Args:
            matrix (Matrix): The input matrix (data with variables as columns)

        Returns:
            Matrix: The correlation matrix
        """
        data = matrix.data
        n = len(data)  # Number of rows
        m = len(data[0])  # Number of columns

        # Step 1: Compute the means of each column
        means = [sum(col) / n for col in zip(*data)]

        # Step 2: Compute the standard deviations of each column
        std_devs = []
        for i in range(m):
            # Use n-1 for sample standard deviation (Bessel's correction)
            variance = sum((data[j][i] - means[i]) ** 2 for j in range(n)) / (n - 1)
            std_devs.append(variance ** 0.5)

        # Step 3: Compute the correlation matrix
        correlation_matrix = []
        for i in range(m):
            correlation_row = []
            for j in range(m):
                # Skip division if either standard deviation is zero
                if std_devs[i] == 0 or std_devs[j] == 0:
                    correlation_row.append(1.0 if i == j else 0.0)
                    continue

                # Compute covariance between columns i and j using n-1
                covariance = sum((data[k][i] - means[i]) * (data[k][j] - means[j])
                                 for k in range(n)) / (n - 1)

                # Normalize the covariance by the standard deviations
                correlation = covariance / (std_devs[i] * std_devs[j])
                correlation_row.append(correlation)
            correlation_matrix.append(correlation_row)

        return Matrix(correlation_matrix)
