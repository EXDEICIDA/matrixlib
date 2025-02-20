from matrixlib.properties import MatrixProperties
from matrixlib.core import Matrix


def demonstrate_matrix_properties():
    """Comprehensive demonstration of various matrix properties"""

    # 1. Companion Matrix Example
    print("\n=== Companion Matrix Examples ===")
    companion = Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 2, 3, 4]  # Coefficients row
    ])
    properties = MatrixProperties(companion)
    print("Is companion matrix?", properties.is_companion())

    # 2. Upper and Lower Triangular Matrices
    print("\n=== Triangular Matrix Examples ===")
    upper_triangular = Matrix([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ])
    lower_triangular = Matrix([
        [1, 0, 0],
        [4, 5, 0],
        [7, 8, 9]
    ])
    print("Is upper triangular?", MatrixProperties(upper_triangular).is_upper_triangular())
    print("Is lower triangular?", MatrixProperties(lower_triangular).is_lower_triangular())

    # 3. Hessenberg Matrix Examples
    print("\n=== Hessenberg Matrix Examples ===")
    upper_hessenberg = Matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 9, 10, 11],
        [0, 0, 12, 13]
    ])
    print("Is upper Hessenberg?", MatrixProperties(upper_hessenberg).is_upper_hessenberg())

    # 4. Orthogonal Matrix Example
    print("\n=== Orthogonal Matrix Example ===")
    orthogonal = Matrix([
        [0, 1],
        [-1, 0]
    ])  # This is a 90-degree rotation matrix
    print("Is orthogonal?", MatrixProperties(orthogonal).is_orthogonal())

    # 5. Special Matrices
    print("\n=== Special Matrix Examples ===")

    # Identity Matrix
    identity = Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    print("Is identity?", MatrixProperties(identity).is_identity())

    # Anti-Identity Matrix
    anti_identity = Matrix([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    print("Is anti-identity?", MatrixProperties(anti_identity).is_anti_identity())

    # 6. Symmetric and Skew-Symmetric
    print("\n=== Symmetric Matrix Examples ===")
    symmetric = Matrix([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ])
    skew_symmetric = Matrix([
        [0, 2, -3],
        [-2, 0, 1],
        [3, -1, 0]
    ])
    print("Is symmetric?", MatrixProperties(symmetric).is_symmetric())
    print("Is skew-symmetric?", MatrixProperties(skew_symmetric).is_skew_symmetric())

    # 7. Sparse Matrix Example
    print("\n=== Sparse Matrix Example ===")
    sparse = Matrix([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
    ])
    print("Is sparse?", MatrixProperties(sparse).is_sparse())

    # 8. Toeplitz Matrix Example
    print("\n=== Toeplitz Matrix Example ===")
    toeplitz = Matrix([
        [1, 2, 3],
        [4, 1, 2],
        [5, 4, 1]
    ])
    print("Is Toeplitz?", MatrixProperties(toeplitz).is_toeplitz())

    # 9. Circulant Matrix Example
    print("\n=== Circulant Matrix Example ===")
    circulant = Matrix([
        [1, 2, 3],
        [3, 1, 2],
        [2, 3, 1]
    ])
    print("Is circulant?", MatrixProperties(circulant).is_circulant())

    # 10. Hilbert Matrix Example
    print("\n=== Hilbert Matrix Example ===")
    hilbert = Matrix([
        [1 / 1, 1 / 2, 1 / 3],
        [1 / 2, 1 / 3, 1 / 4],
        [1 / 3, 1 / 4, 1 / 5]
    ])
    print("Is Hilbert?", MatrixProperties(hilbert).is_hilbert())

    # 11. Vandermonde Matrix Example
    print("\n=== Vandermonde Matrix Example ===")
    vandermonde = Matrix([
        [1, 1, 1],
        [2, 4, 8],
        [3, 9, 27]
    ])
    print("Is Vandermonde?", MatrixProperties(vandermonde).is_vandermonde())

    # 12. Unitary Matrix Example
    print("\n=== Unitary Matrix Example ===")
    unitary = Matrix([
        [1 / 2, 1 / 2],
        [-1 / 2, 1 / 2]
    ])
    print("Is unitary?", MatrixProperties(unitary).is_unitary())

    # 13. Block Matrix Example
    print("\n=== Block Matrix Example ===")
    block = Matrix([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 5, 6],
        [0, 0, 7, 8]
    ])
    print("Is block matrix?", MatrixProperties(block).is_block())

    # 14. Stochastic Matrix Example
    print("\n=== Stochastic Matrix Example ===")
    stochastic = Matrix([
        [0.3, 0.7, 0.0],
        [0.5, 0.2, 0.3],
        [0.0, 0.6, 0.4]
    ])
    print("Is stochastic?", MatrixProperties(stochastic).is_stochastic())

    # 15. Hankel Matrix Example
    print("\n=== Hankel Matrix Example ===")
    hankel = Matrix([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])
    print("Is Hankel?", MatrixProperties(hankel).is_hankel())

    # 16. Additional Properties Tests
    print("\n=== Additional Properties ===")

    # Test singular matrix
    singular = Matrix([
        [1, 2],
        [2, 4]  # linearly dependent rows
    ])
    print("Is singular?", MatrixProperties(singular).is_singular())

    # Test invertible matrix
    invertible = Matrix([
        [1, 2],
        [3, 4]
    ])
    print("Is invertible?", MatrixProperties(invertible).is_invertible())

    # Test permutation matrix
    permutation = Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    print("Is permutation?", MatrixProperties(permutation).is_permutation())

    # Test Hadamard matrix
    hadamard = Matrix([
        [1, 1],
        [1, -1]
    ])
    print("Is Hadamard?", MatrixProperties(hadamard).is_hadamard())


if __name__ == "__main__":
    demonstrate_matrix_properties()