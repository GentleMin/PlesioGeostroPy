"""Functions for linear algebra
"""


import numpy as np
from scipy import linalg
import flamp, gmpy2

from typing import Tuple


class LinSysSolver:
    """Linear system solver, abstract class for all linear solvers
    
    :ivar str _name: name of the solver
    :ivar list _dependencies: dependent libraries / backend libraries
    """
    
    def __init__(self, name: str = "lin_solver", dependencies: list = []) -> None:
        self._name = name
        self._dependencies = dependencies
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a matrix;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a Hermitian matrix;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a generalized eigenvalue problem;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        """Invert a matrix;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a diagonal linear system;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a linear system via explicit inversion;
        abstract, to be overriden
        """
        raise NotImplementedError
    
    def solve(self, A: np.ndarray, B: np.ndarray, diag: bool = False, explicit: bool = True, **kwargs) -> np.ndarray:
        """Solve a linear system; this is the final interface for solving linear systems.
        """
        if diag:
            return self.solve_diag(A, B, **kwargs)
        if explicit:
            return self.solve_explicit(A, B, **kwargs)
        raise NotImplementedError


class StdLinSolver(LinSysSolver):
    """Standard linear solver, built on numpy/scipy, to double precision
    or other default precision depending on the platform
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="standard_solver", 
            dependencies=["numpy", "scipy"])
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a matrix
        """
        return np.linalg.eig(A)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a Hermitian matrix
        """
        return np.linalg.eigh(A, **kwargs)
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a generalized eigenproblem
        """
        return linalg.eig(A, b=B, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        """Invert a matrix
        """
        return np.linalg.inv(A)
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a diagonal linear system
        """
        return (B.T / np.diag(A)).T
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a linear system via explicit inversion
        """
        return self.inv(A) @ B


class MultiPrecLinSolver(LinSysSolver):
    """Multi-precision linear solver, built on gmpy2 + flamp, 
    up to arbitrary precision
    
    :ivar int prec: internal calculation precision 
    """
    
    def __init__(self, prec: int = 113) -> None:
        super().__init__(
            name="multiprec_solver", 
            dependencies=["gmpy2", "flamp"])
        self._prec = prec
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a matrix
        """
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eig(A, **kwargs)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Calculate eigenvalues and eigenvectors of a Hermitian matrix
        """
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eigh(A, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        """Invert a matrix
        """
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.inverse(A)
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a diagonal linear system
        """
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return (B.T / np.diag(A)).T
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        """Solve a linear system via explicit inversion
        """
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return self.inv(A) @ B


def eig_generalized(M: np.ndarray, K: np.ndarray, diag: bool = False, 
    solver: LinSysSolver = StdLinSolver(), **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem.
    This is the interface for solving generalized eigenproblem:
    
    .. math:: 

        \\mathbf{K} \\mathbf{x} = \\lambda \\mathbf{M} \\mathbf{x}
    
    :param np.ndarray M: M (mass) matrix
    :param np.ndarray K: K (stiffness) matrix
    :param bool diag: whether to invert M as a diagonal matrix
    :param LinSysSolver solver: solver to be used
    
    :returns: eigenvalues, eigenvectors
    """
    
    A = solver.solve(M, K, diag=diag, explicit=True)
    return solver.eig(A, **kwargs)

