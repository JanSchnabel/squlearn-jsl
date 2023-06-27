"""
Collection of several regularization techniques of kernel
matrices to ensure positive semi-definiteness and a kernel wrapper
"""

# kernel util
import numpy as np
import scipy
from sklearn.gaussian_process.kernels import Kernel
from ..matrix.kernel_matrix_base import KernelMatrixBase

def kernel_wrapper(kernel_matrix: KernelMatrixBase):
    class CustomKernel(Kernel):
        def __init__(self, kernel_matrix: KernelMatrixBase):
            self.kernel_matrix = kernel_matrix
            super().__init__()

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            kernel_matrix = self.kernel_matrix.evaluate(X, Y)
            if eval_gradient:
                raise NotImplementedError("Gradient not yet implemented for this kernel.")
            else:
                return kernel_matrix

        def diag(self, X):
            return np.diag(self.kernel_matrix.evaluate(X))

        @property
        def requires_vector_input(self):
            return True

        def is_stationary(self):
            return self.kernel_matrix.is_stationary()

    return CustomKernel(kernel_matrix)


def regularize_kernel(gram_matrix):
    """
    Regularizes a given Gram matrix by setting its negative eigenvalues
    to zero. This is done via full eigenvalue decomposition, adjustment
    of the negative eigenvalues and composition of the adjusted spectrum
    and original eigenvectors. This is equivalent of finding the positive
    semi-definite matrix closest to the original one.
    """
    evals, evecs = scipy.linalg.eig(gram_matrix)
    reconstruction = evecs @ np.diag(evals.clip(min=0)) @ evecs.T
    return np.real(reconstruction)


# deprecated regularization technique
def tikhonov_regularization(gram_matrix):
    """
    Tikhonov regularization method of a given Gram matrix. A positive
    semi-definite matrix is obtained by displacing the spectrum of
    the original matrix by its smallest eigenvalue if its negative,
    by subtracting it from the diagonal.
    """
    evals = scipy.linalg.eigvals(gram_matrix)
    if np.min(np.real(evals)) < 0:
        gram_matrix -= np.min(np.real(evals)) * np.identity(gram_matrix.shape[0])
    return gram_matrix


def regularize_full_kernel(K_train, K_testtrain, K_test):
    """
    To regularize ernel matrices calculated inherently within a GPR procedure.
    First, the total Gram matrix is put together and subsequently this matrix
    is passed to the regularize_kernel routine.
    """
    gram_matrix_total = np.block([[K_train, K_testtrain.T], [K_testtrain, K_test]])
    reconstruction = regularize_kernel(gram_matrix_total)
    print(f"Reconstruction error {np.sum(reconstruction - gram_matrix_total)}")

    K_train = reconstruction[: K_train.shape[0], : K_train.shape[1]]
    K_testtrain = reconstruction[K_train.shape[0] :, : K_testtrain.shape[1]]
    K_test = reconstruction[-K_test.shape[0] :, -K_test.shape[1] :]

    return K_train, K_testtrain, K_test
