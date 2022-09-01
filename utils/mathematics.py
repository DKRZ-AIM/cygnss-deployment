import numpy as np

import utils.constants as const

def standard_scale(x, mean=None, scale=None, return_params=False):
    """
    Apply standard scaler to data

    If mean and scale are not provided, calculate on x

    If return_params, return mean and scale along transformed x
    Else return transformed x
    """

    if mean is None and scale is None:
        mean = np.mean(x)
        scale = np.std(x)

    x_tilde = (x - mean) / scale

    if return_params:
        return x_tilde, mean, scale
    return x_tilde

def standard_scale_inverse(x_tilde, mean, scale):
    """
    Apply inverse standard scaler
    """

    return scale * x_tilde + mean


def great_circle_distance(phi1, lambda1, phi2, lambda2, degree=True):
    """
    Calculate the great circle distance between two points

    phi1, lambda1 - lat / lon coordinates of point 1
    phi2, lambda2 - lat / lon coordinates of point 2
    degree - if True, coordinates are converted to rad first
    """

    if degree:
        phi1 = phi1 * np.pi / 180
        phi2 = phi2 * np.pi / 180
        lambda1 = lambda1 * np.pi / 180
        lambda2 = lambda2 * np.pi / 180

        x = np.sin( (phi2-phi1)/2 )**2 + np.cos(phi1)*np.cos(phi2) * np.sin( (lambda2-lambda1)/2)**2
        alpha = 2 * np.arcsin( np.sqrt(x) )

        return const.MEAN_EARTH_RADIUS_KM * alpha


def losses_by_quantiles(arr, y_true, y_pred, n_quantiles=10):
    """Returns the MSE loss for y_true and y_pred for n_quantiles quantiles computed on arr.
    Useful to answer questions like 'How does my loss change by the amount of rain observed?'."""
    arr, y_true, y_pred = arr.flatten(), y_true.flatten(), y_pred.flatten()
    assert arr.size == y_true.size == y_pred.size

    quantiles = np.quantile(arr, np.arange(0, 1.0, 1.0 / n_quantiles))
    q_idx = np.digitize(arr, quantiles)
    losses = [mse(y_true[q_idx == i], y_pred[q_idx == i]) for i in range(1, len(quantiles) + 1)]
    return losses, quantiles

def bias_by_quantiles(arr, y_true, y_pred, n_quantiles=10):
    """Returns the bias for y_true and y_pred for n_quantiles quantiles computed on arr."""
    arr, y_true, y_pred = arr.flatten(), y_true.flatten(), y_pred.flatten()
    assert arr.size == y_true.size == y_pred.size

    quantiles = np.quantile(arr, np.arange(0, 1.0, 1.0 / n_quantiles))
    q_idx = np.digitize(arr, quantiles)
    losses = [np.mean(y_pred[q_idx == i] - y_true[q_idx == i]) for i in range(1, len(quantiles) + 1)]
    return losses, quantiles


def losses_by_quantiles_2d(arr1, arr2, y_true, y_pred, n_quantiles=10):
    """Returns the MSE loss for y_true and y_pred for the 100 quantiles computed, based on all combinations of the
    10 quantiles of arr1 and arr2."""
    arr1, arr2, y_true, y_pred = arr1.flatten(), arr2.flatten(), y_true.flatten(), y_pred.flatten()
    assert arr1.size == arr2.size == y_true.size == y_pred.size

    quantiles1 = np.quantile(arr1, np.arange(0, 1.0, 1.0 / n_quantiles))
    quantiles2 = np.quantile(arr2, np.arange(0, 1.0, 1.0 / n_quantiles))
    q_idx1 = np.digitize(arr1, quantiles1)
    q_idx2 = np.digitize(arr2, quantiles2)
    losses = [mse(y_true[(q_idx1 == i) & (q_idx2 == j)], y_pred[(q_idx1 == i) & (q_idx2 == j)])
              for i in range(1, n_quantiles + 1) for j in range(1, n_quantiles + 1)]
    losses = np.array(losses).reshape(n_quantiles, n_quantiles)
    sizes = [y_true[(q_idx1 == i) & (q_idx2 == j)].size for i in range(1, n_quantiles + 1) for j in range(1, n_quantiles + 1)]
    sizes = np.array(sizes).reshape(n_quantiles, n_quantiles)
    return losses, quantiles1, quantiles2, sizes


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def cartesian_to_latlon(x, y, z):
    '''
    Convert geocentric cartesian coordinates on Earth to lat/lon

    Returns:
    latitude (deg) / longitude (deg)
    '''

    # TODO get proper formula for the ellipsoid

    lat = np.arcsin( z / const.MEAN_EARTH_RADIUS_KM )
    lon = np.arctan2( y, x )

    lat *= 180 / np.pi
    lon *= 180 / np.pi

    return lat, lon


def cartesian_to_spherical(x, y, z, degree=False):
    '''
    Convert cartesian coordinates to spherical coordinates

    Returns:
    radius (m) / polar angle (deg if True) / azimuthal angle (deg if True)
    '''

    r = np.sqrt( x**2 + y**2 + z**2 )
    theta = np.arccos( z / r )
    phi = np.arctan2( y, x )

    if degree:
        theta *= 180 / np.pi
        phi *= 180 / np.pi

    return r, theta, phi

def kullback_leibler_divergence(p, q):
    '''
    Calculate the Kullback Leibler divergence between two probability distributions p, q
    '''

    kld = np.sum( np.where( (p>0) & (q>0), p*np.log(p/q), 0))

    return kld

def ks_test(y1, y2, alpha=0.95):
    '''
    The KS statistic is the largest difference in frequencies in the ECDF 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp

    If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis
    that the distributions of the two samples are the same.

    Parameters:
    y1, y2 - arrays with samples to compare
    alpha - confidence level (default: 0.95)

    Returns:
    True if we cannot reject the hypothesis that the two distributions are the same
    '''

    from scipy.stats import ks_2samp

    stat = ks_2samp(y1, y2)

def f_exp(x, a, b, c):
    '''
    Returns exponential function
    '''
    return a*np.exp(b*x) + c

def f_poly2(x, a, b, c):
    '''
    Returns 2nd degree polynom
    '''

    return a*x**2 + b*x + c

def quantile25(y):
    '''
    Returns 25% quantile of array y
    '''
    return np.quantile(y, 0.25)

def quantile75(y):
    '''
    Returns 75% quantile of array y
    '''
    return np.quantile(y, 0.75)

def weibull(x, k, l):
    '''
    Returns Weibull distribution

    x - 1D array
    k - shape parameter
    l - scale parameter (lambda)
    '''

    if k < 0:
        raise ValueError('Shape parameter > 0 required')
    if l < 0:
        raise ValueError('Scale parameter > 0 required')

    y = k / l * ( x / l )**(k-1) * np.exp(-(x/l)**k)
    return y

def log_transform(y, eps=1e-3):
    '''
    Perform the transformation

    y_tilde = log( eps + y ) - log( eps )

    Keeping zeroes in y unchanged in y_tilde

    Parameters:
    y - 1D array
    eps - small offset (default: 1e-3)

    Returns:
    Transformed array
    '''

    y_tilde = np.log( eps + y ) - np.log( eps )
    return y_tilde

def average_to_grid(lon, lat, var, resolution=1):
    '''
    Grid a time-dependent variable in lon/lat and average over all counts
    
    lon - time series of lon coordinate (1D) (0...360)
    lat - time series of lat coordinate (1D)
    var - time series of variable (1D)
    resolution - target grid resolution (default: 1 deg)
    
    Returns:
    2D gridded arrays for lat, lon, count-averaged var, count
    '''

    assert len(lon) == len(lat)
    assert len(lon) == len(var)

    grid_lon = np.arange(0, 360+resolution, resolution)
    grid_lat = np.arange(-90, 90+resolution, resolution)[::-1] # top left is +lat

    ix_lon = np.digitize(lon, grid_lon)
    ix_lat = np.digitize(lat, grid_lat)

    xx, yy = np.meshgrid(grid_lon, grid_lat, indexing='ij')
    zz = np.zeros([len(grid_lon), len(grid_lat)])
    nn = np.zeros([len(grid_lon), len(grid_lat)]) # counts

    for v,i,j in zip(var, ix_lon, ix_lat):
        zz[i,j] += v
        nn[i,j] += 1


    zz[nn>0] /= nn[nn>0]
    zz[nn==0] = None

    return xx[1:,1:], yy[1:,1:], zz[1:,1:], nn[1:,1:]

def power_transform(y, scale):
    '''
    Apply the power transform to input vector y

    y_tilde = (y**scale - 1) / scale

    Parameters:
    y - np array to be transformed
    scale - scale parameter
    
    Returns:
    y_tilde - transformed np array  
    '''

    if scale < 0:
        raise ValueError("Scale must not be negative: ", scale)
    if scale == 0:
        raise ValueError("Not implemented: scale = 0")
    if np.any(y<0):
        raise ValueError("y must be semi-positive.")

    return (y**scale - 1) / scale

def inverse_power_transform(y, scale):
    '''
    Perform the inverse transformation to power_transform.
    '''

    return ( scale * y + 1 )**(1/scale)

def scale_zero_mean_unit_variance(y):
    '''
    Scale numbers in y to zero mean and unit variance

    Returns:
    y_scaled, mu, sigma
    '''

    mu = np.mean(y)
    sigma = np.std(y)

    y -= mu
    y /= sigma

    return y, mu, sigma
    
def inverse_scale_zero_mean_unit_variance(y, mu, sigma):
    '''
    Perform the inverse operation to scale_zero_mean_unit_variance
    '''
    
    return y * sigma + mu




