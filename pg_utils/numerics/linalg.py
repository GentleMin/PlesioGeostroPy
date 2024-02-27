"""Functions for linear algebra
"""


import numpy as np
from scipy import linalg
import flamp, gmpy2

from typing import Tuple


class LinSysSolver:
    
    def __init__(self, name: str = "lin_solver", dependencies: list = []) -> None:
        self._name = name
        self._dependencies = dependencies
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def solve(self, A: np.ndarray, B: np.ndarray, diag: bool = False, explicit: bool = True, **kwargs) -> np.ndarray:
        if diag:
            return self.solve_diag(A, B, **kwargs)
        if explicit:
            return self.solve_explicit(A, B, **kwargs)
        raise NotImplementedError


class StdLinSolver(LinSysSolver):
    
    def __init__(self) -> None:
        super().__init__(
            name="standard_solver", 
            dependencies=["numpy", "scipy"])
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eig(A)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(A, **kwargs)
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        return linalg.eig(A, b=B, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        return np.linalg.inv(A)
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        return (B.T / np.diag(A)).T
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        return self.inv(A) @ B


class MultiPrecLinSolver(LinSysSolver):
    
    def __init__(self, prec: int = 113) -> None:
        super().__init__(
            name="multiprec_solver", 
            dependencies=["gmpy2",])
        self._prec = prec
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eig(A, **kwargs)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eigh(A, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.inverse(A)
    
    def solve_diag(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return (B.T / np.diag(A)).T
    
    def solve_explicit(self, A: np.ndarray, B: np.ndarray, **kwargs) -> np.ndarray:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return self.inv(A) @ B


def eig_generalized(M: np.ndarray, K: np.ndarray, diag: bool = False, 
    solver: LinSysSolver = StdLinSolver(), **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem.
    This should serve as the final interface for generalized eigenproblems.
    """
    
    A = solver.solve(M, K, diag=diag, explicit=True)
    return solver.eig(A, **kwargs)

