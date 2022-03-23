import numpy.typing as npt
import numpy as np
import numba as nb

def exterior(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Returns the wedge/exterior product of two vectors. This is the slower version of the function. For the faster version, use
    the function wedge.

    Args:
        a (ArrayLike): A vector.
        b (ArrayLike): A vector.
    
    Returns:
        np.ndarray: The wedge product of a and b, c, as a list c_ij, when j > i from i = 0 to j = n.
        e.g. for 3-dimensional cartesian vectors, you'd get [c_xy, c_xz, c_yz] which is equivalent to [z, -y, x].
    """
    #Turn both inputs into numpy arrays, if they aren't already.
    a,b = np.array(a), np.array(b)

    #Find the outer product, which will always be square.
    outer = np.outer(a, b)

    #Subtract the outer product from its transpose.
    diff = outer - outer.T

    #Return the result.
    return diff[np.triu_indices_from(diff, k=1)]

def inverse_triangular(n:int, rounding_places:int = 3) -> float:
    """Returns the inverse of the triangular number n.

    Args:
        n (int): An integer.
        rounding_places (int): The number of decimal places to round to.

    Returns:
        float: The inverse of the triangular number n.
    """
    #Return the inverse of the triangular number.
    return np.round(np.sqrt(2 * n + (1 / 4)) - (1 / 2), rounding_places)

@nb.jit(nb.float64[::1](nb.float64[::1],nb.float64[::1]), nopython=True)
def wedge(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """Returns the wedge/exterior product of two vectors.

    Args:
        a (ArrayLike): A vector.
        b (ArrayLike): A vector.
    
    Returns:
        np.ndarray: The wedge product of a and b, c, as a list c_ij, when j > i from i = 0 to j = n.
        e.g. for 3-dimensional cartesian vectors, you'd get [c_xy, c_xz, c_yz] which is equivalent to [z, -y, x].
    """
    
    #
    n               = len(a)
    idx             = np.triu_indices(n, k=1)
    return_length   = n * (n - 1) // 2
    return_array    = np.empty(return_length)

    for k in range(return_length):
        i, j            = idx[0][k], idx[1][k]
        return_array[k] = a[i] * b[j] - a[j] * b[i]
    
    return return_array

@nb.jit(nb.float64(nb.float64[::1],nb.float64[::1]), nopython=True)
def wedge_norm(a:np.ndarray, b:np.ndarray) -> np.float64:
    """Returns the norm of the wedge/exterior product of two vectors.

    Args:
        a (float64 array): A vector.
        b (float64 array): A vector.
    
    Returns:
        float64: The norm of the wedge product of a and b.
    """
    n               = len(a)
    idx             = np.triu_indices(n, k=1)
    return_length   = n * (n - 1) // 2
    return_value    = 0.0

    for k in range(return_length):
        i, j            = idx[0][k], idx[1][k]
        return_value   += np.square(a[i] * b[j] - a[j] * b[i])
    
    return np.sqrt(return_value)

@nb.jit(nb.float64[::1](nb.float64[::1], nb.float64[::1]), nopython=True)
def geometric_product_norm(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """Returns the geometric product of two vectors.

    Args:
        a (float64 array): A vector.
        b (float64 array): A vector.
    
    Returns:
        np.ndarray: The geometric product of a and b. The first element is the norm of the wedge product, and the second
        element is the dot product of the two vectors.
    """
    return_array    = np.zeros(2)
    return_array[0] = wedge_norm(a, b)
    for i in range(len(a)):
        return_array[1] += a[i] * b[i]
    return return_array