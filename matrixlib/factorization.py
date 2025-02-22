import math
from matrixlib.operations import Operation
from random import random

from matrixlib.core import Matrix


# This is a clas that helps to factorize matrices

class Factorization:

    @staticmethod
    def lu_decomposition(matrix: Matrix):
        """LU Decomposition of the matrix."""
        if not matrix.is_square_matrix():
            raise ValueError("LU Decomposition can only be done on square matrices.")

        data = [row[:] for row in matrix.data]  # Making a copy of the matrix
        n = len(data)
        l = [[0] * n for _ in range(n)]
        u = [[0] * n for _ in range(n)]

        # Perform LU Decomposition
        for i in range(n):
            # Upper Triangular
            for j in range(i, n):
                u[i][j] = data[i][j]
                for k in range(i):
                    u[i][j] -= l[i][k] * u[k][j]

            # Lower Triangular
            for j in range(i, n):
                if i == j:
                    l[i][i] = 1
                else:
                    l[j][i] = data[j][i]
                    for k in range(i):
                        l[j][i] -= l[j][k] * u[k][i]
                    l[j][i] /= u[i][i]

        return l, u

    @staticmethod
    def eigenvalue_decomposition(matrix: Matrix, max_iterations=100, tolerance=1e-10):
        """
        Computes the eigenvalues and eigenvectors of a square matrix.
        Returns a tuple of (eigenvalues, eigenvectors).

        Eigenvalues are returned as a list.
        Eigenvectors are returned as a list of lists, where each inner list represents an eigenvector.
        """
        if not matrix.is_square_matrix():
            raise ValueError("Eigenvalue decomposition can only be done on square matrices.")

        # Import needed for random vector generation

        n = matrix.shape[0]
        eigenvalues = []
        eigenvectors = []
        working_matrix = Matrix([row[:] for row in matrix.data])

        for _ in range(n):
            # Start with a random vector
            vector = [random() for _ in range(n)]
            vector_norm = math.sqrt(sum(v * v for v in vector))
            vector = [v / vector_norm for v in vector]

            # Power iteration
            for _ in range(max_iterations):
                # Multiply matrix by vector using Operation.multiply
                vector_matrix = Matrix([[v] for v in vector])
                new_vector_matrix = Operation.multiply(working_matrix, vector_matrix)
                new_vector = [new_vector_matrix.data[i][0] for i in range(n)]

                # Normalize
                new_vector_norm = math.sqrt(sum(v * v for v in new_vector))
                if new_vector_norm < tolerance:
                    break

                new_vector = [v / new_vector_norm for v in new_vector]

                # Check convergence
                if sum((a - b) ** 2 for a, b in zip(vector, new_vector)) < tolerance:
                    break

                vector = new_vector

            # Compute eigenvalue using Rayleigh quotient with Operation.multiply
            vector_matrix = Matrix([[v] for v in vector])
            matrix_times_vector = Operation.multiply(working_matrix, vector_matrix)
            matrix_times_vector = [matrix_times_vector.data[i][0] for i in range(n)]
            eigenvalue = sum(a * b for a, b in zip(matrix_times_vector, vector))

            eigenvalues.append(eigenvalue)
            eigenvectors.append(vector)

            # Deflate the matrix to find next eigenvalue
            outer_product = [[vector[i] * vector[j] for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    working_matrix.data[i][j] -= eigenvalue * outer_product[i][j]

        return eigenvalues, eigenvectors

    @staticmethod
    def svd(matrix: Matrix):
        """
        Singular Value Decomposition of the matrix.
        Returns U, S, V^T such that matrix = U * S * V^T,
        where U and V are orthogonal matrices and S is a diagonal matrix.
        """
        # Get the transpose of the matrix
        matrix_t = Operation.transpose(matrix)

        # Compute A^T * A for right singular vectors
        ata = Operation.multiply(matrix_t, matrix)

        # Compute A * A^T for left singular vectors
        aat = Operation.multiply(matrix, matrix_t)

        # Get eigenvalues and eigenvectors for A^T * A
        eigenvalues_v, eigenvectors_v = Factorization.eigenvalue_decomposition(ata)

        # Get eigenvalues and eigenvectors for A * A^T
        eigenvalues_u, eigenvectors_u = Factorization.eigenvalue_decomposition(aat)

        # Compute singular values (square root of eigenvalues)
        singular_values = [math.sqrt(ev) for ev in eigenvalues_v]

        # Sort singular values and corresponding eigenvectors in descending order
        sorted_triplets = sorted(zip(singular_values, eigenvectors_u, eigenvectors_v),
                                 key=lambda x: x[0], reverse=True)

        # Unpack the sorted values
        singular_values = [t[0] for t in sorted_triplets]
        eigenvectors_u = [t[1] for t in sorted_triplets]
        eigenvectors_v = [t[2] for t in sorted_triplets]

        # Construct the S diagonal matrix
        m, n = matrix.shape
        s_matrix = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(min(m, n)):
            if i < len(singular_values):
                s_matrix[i][i] = singular_values[i]

        # Return U, S, V^T
        return Matrix(eigenvectors_u), Matrix(s_matrix), Operation.transpose(Matrix(eigenvectors_v))

    @staticmethod
    def qr_decomposition(matrix: Matrix):
        """QR Decomposition using Gram-Schmidt Process."""
        if not matrix.is_square_matrix():
            raise ValueError("QR Decomposition requires a square matrix.")

        n = matrix.shape[0]
        q = [[0] * n for _ in range(n)]
        r = [[0] * n for _ in range(n)]
        a = [row[:] for row in matrix.data]

        for i in range(n):
            q[i] = a[i][:]  # Copy column
            for j in range(i):
                r[j][i] = sum(q[k][j] * a[k][i] for k in range(n))
                for k in range(n):
                    q[k][i] -= r[j][i] * q[k][j]

            r[i][i] = int(math.sqrt(sum(q[k][i] ** 2 for k in range(n))))
            for k in range(n):
                q[k][i] /= r[i][i]

        return Matrix(q), Matrix(r)

    @staticmethod
    def jordan_decomposition(matrix: Matrix):
        """
        Jordan Decomposition of a matrix.
        Returns P and J such that matrix = P * J * P^(-1),
        where J is the Jordan normal form and P is the transformation matrix.

        This is a simplified implementation that works well for diagonalizable matrices
        and approximates the Jordan form for non-diagonalizable matrices.
        """
        if not matrix.is_square_matrix():
            raise ValueError("Jordan Decomposition can only be done on square matrices.")

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = Factorization.eigenvalue_decomposition(matrix)
        n = matrix.shape[0]

        # Construct the transformation matrix P
        p_matrix = Matrix([eigenvector for eigenvector in eigenvectors])

        # Construct the Jordan form matrix
        j_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            j_matrix[i][i] = eigenvalues[i]

            # For non-diagonalizable matrices, we would add 1's above the diagonal
            # in the Jordan blocks. This is a simplified implementation.
            if i < n - 1 and abs(eigenvalues[i] - eigenvalues[i + 1]) < 1e-10:
                # Check if eigenvectors are linearly dependent (simplified check)
                vec1 = eigenvectors[i]
                vec2 = eigenvectors[i + 1]
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = math.sqrt(sum(a * a for a in vec1))
                norm2 = math.sqrt(sum(b * b for b in vec2))
                if abs(dot_product / (norm1 * norm2) - 1.0) < 1e-6:
                    j_matrix[i][i + 1] = 1  # Add superdiagonal entry for Jordan block

        # Compute P^(-1)
        p_inverse = Operation.inverse(p_matrix)

        return p_matrix, Matrix(j_matrix), p_inverse

    @staticmethod
    def polar_decomposition(matrix: Matrix):
        """
        Polar Decomposition of a matrix.
        Returns U and P such that matrix = U * P,
        where U is a unitary matrix and P is a positive semi-definite Hermitian matrix.

        For real matrices, U is orthogonal and P is symmetric positive semi-definite.
        """
        # Step 1: Compute A^T * A
        matrix_t = Operation.transpose(matrix)
        ata = Operation.multiply(matrix_t, matrix)

        # Step 2: Compute the square root of A^T * A, which is P
        # We'll use SVD to help us compute this
        eigenvalues, eigenvectors = Factorization.eigenvalue_decomposition(ata)

        # Construct P using the eigendecomposition
        n = matrix.shape[0]
        p_data = [[0 for _ in range(n)] for _ in range(n)]

        # Convert eigenvectors to column vectors for matrix multiplication
        eigenvector_matrix = Matrix(eigenvectors)
        eigenvector_matrix_t = Operation.transpose(eigenvector_matrix)

        # Construct diagonal matrix of square roots of eigenvalues
        sqrt_eig_diag = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            sqrt_eig_diag[i][i] = math.sqrt(abs(eigenvalues[i]))  # Use abs to handle small numerical errors

        # P = V * sqrt(D) * V^T
        temp = Operation.multiply(eigenvector_matrix, Matrix(sqrt_eig_diag))
        p_matrix = Operation.multiply(temp, eigenvector_matrix_t)

        # Step 3: Compute U = A * P^(-1)
        # If P is singular, use pseudoinverse or regularize
        try:
            p_inverse = Operation.inverse(p_matrix)
            u_matrix = Operation.multiply(matrix, p_inverse)
        except ValueError:
            # For singular P, use a regularization approach
            epsilon = 1e-10
            regularized_p = [[p_matrix.data[i][j] + (epsilon if i == j else 0)
                              for j in range(n)] for i in range(n)]
            p_inverse = Operation.inverse(Matrix(regularized_p))
            u_matrix = Operation.multiply(matrix, p_inverse)

        return u_matrix, p_matrix

