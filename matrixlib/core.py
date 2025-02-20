class Matrix:
    def __init__(self, data):
        """Initialize a matrix with an N-D list"""
        if not all(data) or len(data[0]) == 0:
            raise ValueError("Matrix cannot have empty rows.")

        # Check that all dimensions are consistent
        self.data = data
        self.shape = self._get_shape(data)

        # Ensure all sub-lists (dimensions) are consistent in size
        if not all(self._get_shape(item) == self.shape[1:] for item in data):
            raise ValueError("All sub-lists must have the same dimensions.")

    def __eq__(self, other):
        """Override equality comparison to compare matrix data."""
        if isinstance(other, Matrix):
            return self.data == other.data
        return False

    @staticmethod
    def _get_shape(data):
        """Return the shape (dimensions) of the matrix."""
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            if len(data) > 0:
                data = data[0]
            else:
                break
        return tuple(shape)

    def __repr__(self):
        """Return a string representation of the matrix."""
        if len(self.shape) == 1:
            return str(self.data)
        return self._repr_recursive(self.data, 0)

    def _repr_recursive(self, data, depth):
        """Recursively represent multi-dimensional lists."""
        # Check if the current data is a scalar (e.g., an integer)
        if depth == len(self.shape) - 1 or not isinstance(data, list):
            return " ".join(map(str, data))  # Convert each item to string
        return "\n".join([self._repr_recursive(subdata, depth + 1) for subdata in data])

    def is_square_matrix(self):
        """Check if the matrix is square (same number of rows and columns)."""
        return len(self.shape) == 2 and self.shape[0] == self.shape[1]

    def get_rank(self):
        """Get the rank of the matrix using Gaussian elimination."""
        matrix = [row[:] for row in self.data]  # Make a copy of the matrix

        # Perform Gaussian elimination (Row Echelon Form)
        row_count = len(matrix)
        col_count = len(matrix[0])

        rank = 0
        for r in range(row_count):
            # Find the pivot column for row r
            if matrix[r][r] != 0:
                rank += 1
                for i in range(r + 1, row_count):
                    # Eliminate entries below the pivot
                    if matrix[i][r] != 0:
                        factor = matrix[i][r] / matrix[r][r]
                        for j in range(r, col_count):
                            matrix[i][j] -= factor * matrix[r][j]
            else:
                # If the current row's pivot is zero, find a row with a non-zero entry and swap
                for i in range(r + 1, row_count):
                    if matrix[i][r] != 0:
                        matrix[r], matrix[i] = matrix[i], matrix[r]
                        rank += 1
                        break

        return rank
