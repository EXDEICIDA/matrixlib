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
