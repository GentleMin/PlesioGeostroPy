import numpy as np
import h5py

matrix_file_base = "./out/cases/Malkus/Matrix.h5"
matrix_file_comp = "./out/cases/Malkus/Matrix_2.h5"


def test_matrix_equivalence():
    
    with h5py.File(matrix_file_base, 'r') as fread:
        M_old = fread["M"][()]
        K_old = fread["K"][()]
    
    with h5py.File(matrix_file_comp, 'r') as fread:
        M_new = fread["M"][()]
        K_new = fread["K"][()]
    
    assert np.allclose(M_old, M_new)
    assert np.allclose(K_old, K_new)


