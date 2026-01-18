import numpy as np
from gglasso.problem import glasso_problem 
from nilearn.connectome import ConnectivityMeasure
from scipy import stats


def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    """
    Compute a connectivity matrix from a given timeseries.

    Parameters
    ----------
    timeseries : np.ndarray
        The input timeseries to compute the connectivity matrix from.
        Input shape: (n_subjects, n_timepoints, n_nodes)
    conn_type : str
        The type of connectivity to compute. 
        Options: 'corr', 'partial_corr', 'tang', 'glasso'.

    Returns
    -------
    conn : np.ndarray
        The computed connectivity matrix.
    """
    if isinstance(timeseries, list):
        timeseries = np.array(timeseries)

    if timeseries.ndim != 3:
        raise ValueError(f"Input timeseries should have shape (n_subjects, n_timepoints, n_nodes), "
                         f"but got {timeseries.shape}")
    
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='correlation', 
                                   standardize=False).fit_transform(timeseries)
        
        conn[conn == 1] = 0.999999
        
        for i in conn:
            np.fill_diagonal(i, 0)
        conn = np.arctanh(conn)

    elif conn_type == 'partial_corr':
        conn = ConnectivityMeasure(kind='partial correlation', 
                                   standardize=False).fit_transform(timeseries)
        
    elif conn_type == 'tang':
        conn = ConnectivityMeasure(kind='tangent', 
                                   standardize=False).fit_transform(timeseries)
        
    elif conn_type == 'glasso':
        nsub, _, nodes = timeseries.shape
        conn = np.zeros((nsub, nodes, nodes))
        # glasso estimates one sub at a time
        for sub in range(nsub):
            conn[sub] = graphicalLasso(timeseries[sub])
    
    else:
        raise NotImplementedError
    
    return conn


def graphicalLasso(data, L1=0.03):
    '''
    !code by Peterson, K. L., Sanchez-Romero, R., Mill, R. D., & Cole, M. W. (2023). 
    Regularized partial correlation provides reliable functional connectivity estimates 
    while correcting for widespread confounding. 
    In bioRxiv (p. 2023.09.16.558065). https://doi.org/10.1101/2023.09.16.558065 

    https://github.com/ColeLab/ActflowToolbox/blob/master/connectivity_estimation/graphicalLassoCV.py
    
    Calculates the L1-regularized partial correlation matrix of a dataset. 
    Runs GGLasso's graphical lasso function (glasso_problem.solve()) and several other necessary steps.
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value
    OUTPUT:
        glassoParCorr : regularized partial correlation coefficients (i.e., FC matrix)
        prec : precision matrix, where entries are not yet transformed into partial correlations 
        (used to compute loglikelihood)
    '''
    data = data.T
    nNodes = data.shape[0] # for one person 
    # Number of timepoints in data
    nTRs = data.shape[1]

    # Z-score the data
    data_scaled = stats.zscore(data, axis=1)

    # Estimate the empirical covariance
    empCov = np.cov(data_scaled, rowvar=True)

    # Run glasso
    glasso = glasso_problem(empCov, nTRs, 
                            reg_params={'lambda1': L1},
                            latent=False, do_scaling=False)
    
    glasso.solve(verbose=False)
    prec = np.squeeze(glasso.solution.precision_)

    # Transform precision matrix into regularized partial correlation matrix
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    glassoParCorr = -prec * denom * denom.T
    np.fill_diagonal(glassoParCorr, 0)

    return glassoParCorr