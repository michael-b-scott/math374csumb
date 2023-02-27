# -*- coding: utf-8 -*-
"""
Introduction to Matrix Algebra in Python
MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay

@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import numpy as np
from numpy.linalg import eig

# -----------------------------------------------------------------------------
# Objects in Linear Algebra defined as numpy arrays
# -----------------------------------------------------------------------------

# Scalar
# Typically a real number; could be rationals or complex numbers
# For MATH 374, scalars default to real numbers
scalar = 5.679 

# Vector n-tuples of real numbers can be rows or columns
vector_row = np.array([[1, 2, 3]])
vector_col = np.array([[1],
                       [2],
                       [3]])
print('\n vector_row dimension (rows x columns)\n', vector_row.shape)
print('\n vector_col dimension \n', vector_col.shape)


# Matrices
# An m×n matrix is a rectangular table (array) of numbers consisting of 
#   m rows and n columns
matrix = np.array([[1, 2], [3, 4]])
matrix2 = np.random.randint(10, size=(4, 6))

print('\n matrix dimension \n', matrix.shape)
print('\n matrix2 dimension \n', matrix2.shape)

# Tensors
# An nth-rank tensor in m-dimensional space is a mathematical object that 
#   has n indices and m^n components and obeys certain transformation rules.
#   You can think of a matrix of matrices. We won't be using tensors very often
#   in MATH 374.
tensor = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
])

# -----------------------------------------------------------------------------
# Operations on vector/matrix objects and special types of matrices.
# -----------------------------------------------------------------------------

# Scalar Muliplication
A = matrix
print('\n -5A = \n', -5*A)

# Matrix Addition (Vector addition is the same.)
B = np.array([[5, 6], [7, 8]])
sumAB = A + B
print('\nMatrix addiiton and subtraction.')
print('\nA+B = \n', A+B)
print('\nA-B = \n', A-B)

# Matrix Muliplication
ABProduct = np.dot(A,B)
print('\nAB = \n', ABProduct)
print('\nBA = \n', np.dot(B,A))

# Not all matrices can be multiplied. The number of columns for the left 
#   matrix must = the number of row of the right matrix.
C = np.array([[1,1,1], [1,-1,1]])
D = np.array([[1,1,1], [-1,1,1]])
# You will get an error with product below.
#CDProduct = np.dot(C,D)

# Identity Matrix
print('\n I_3 = \n', np.eye(3)) # Gives 3x3 Identity Matrix
print('\nAI = IA = \n', np.dot(A,np.eye(2)))

# Transpose of a Matrix (Switch rows and columns)
C_t = C.T

print('\n C_t*D = \n', np.dot(C_t, D)) # Can multiply C_t and D since number columns and rows match

# Other operations on matrices
M = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])

# Inverse of Matrix M
print("\nInverse of M:\n", np.linalg.inv(M))

# Rank of a matrix (Number of independent columns of M)
print("\nRank of M:", np.linalg.matrix_rank(M))
 
# Trace of matrix A
print("\nTrace of M:", np.trace(M))
 
# Determinant of a matrix
print("\nDeterminant of M:", np.linalg.det(M))

print("\nMatrix M raised to power 3:\n",
           np.linalg.matrix_power(M, 3))

# -----------------------------------------------------------------------------
# Inner product including dot product
# -----------------------------------------------------------------------------
# Define two row vectors
u = np.array([7, 2, 2])
v = np.array([1, 4, 9])
print('\nInner product <u,v> = ', np.dot(u, v))

# -----------------------------------------------------------------------------
# Eigenvalues and eigenvectors of a matrix
# -----------------------------------------------------------------------------
#Define new matrix A
A = np.array([[2,3],
              [3,-6]])

w,v = eig(A)
# Eigenvectors
# This gives normalized (unit “length”) eigenvectors, such that the column 
#   v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
print('E-vectors:\n', v)
print('\nE-values:', w)
l0 = w[0]
l1 = w[1]
v0 = v[:,0]
v1 = v[:,1]

print('\nAv0 = ', np.dot(A,v0))
print('\n',l0,'v0 = ', l0*v0)





