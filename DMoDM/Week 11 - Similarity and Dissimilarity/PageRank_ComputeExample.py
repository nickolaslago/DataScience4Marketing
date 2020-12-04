## Compute ranks example
# Adapted from https://www.rose-hulman.edu/~bryan/googleFinalVersionFixed.pdf and
#              https://glowingpython.blogspot.com/2011/05/four-ways-to-compute-google-pagerank.html

# Packages
import numpy as np
from numpy import *

# Function
def powerMethodBase(A,x0,iter):
 """ basic power method """
 for i in range(iter):
  x0 = dot(A,x0)
  x0 = x0/linalg.norm(x0,1)
 return x0


def powerMethod(A,x0,m,iter):
 """ power method modified to compute
     the maximal real eigenvector 
     of the matrix M built on top of the input matrix A """
 n = A.shape[1]
 delta = m*(array([1]*n,dtype='float64')/n) # array([1]*n is [1 1 ... 1] n times
 for i in range(iter):
  x0 = dot((1-m),dot(A,x0)) + delta
 return x0

def maximalEigenvector(A):
 """ using the eig function to compute eigenvectors """
 n = A.shape[1]
 w,v = linalg.eig(A)
 return abs(real(v[:n,0])/linalg.norm(v[:n,0],1))

def linearEquations(A,m):
 """ solving linear equations 
     of the system (I-(1-m)*A)*x = m*s """
 n = A.shape[1]
 C = eye(n,n)-dot((1-m),A)
 b = m*(array([1]*n,dtype='float64')/n)
 return linalg.solve(C,b)

def getTeleMatrix(A,m):
 """ return the matrix M
     of the web described by A """
 n = A.shape[1]
 S = ones((n,n))/n
 return (1-m)*A+m*S


# Example
matrix = [[0.0, 0.167, 0.231, 0.333, 0.286, 0.0, 0.167],
          [0.111, 0.0, 0.154, 0.0, 0.143, 0.250, 0.167],
          [0.333, 0.333, 0.0, 0.333, 0.286, 0.500, 0.333],
          [0.222, 0.000, 0.154, 0.167, 0.143, 0.0, 0.0],
          [0.222, 0.167, 0.154, 0.167, 0.0, 0.0, 0.167],
          [0.0, 0.167, 0.154, 0.0, 0.0, 0.0, 0.167],
          [0.111, 0.167, 0.154, 0.0, 0.143, 0.250, 0.0]
         ]

A = np.array(matrix)
n = A.shape[1]
m = 1/n
M = getTeleMatrix(A,m)

x0 = [1]*n
x1 = powerMethod(A,x0,m,200)
x2 = powerMethodBase(M,x0,200)
x3 = maximalEigenvector(M)
x4 = linearEquations(A,m)

# Print all methods results
print([x1, x2, x3, x4])