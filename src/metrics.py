import numpy as np
from fc import get_connectome
import pandas as pd


def qc_fc(fc, mean_rms_vec) -> np.ndarray:
    """
    Compute the quality control functional connectivity (QC-FC) matrix.

    This function calculates the correlation between each element of the 
    functional connectivity matrix and the mean root mean squared 
    (RMS) vector. The resulting matrix represents the QC-FC values.

    Parameters
    ----------
    fc : np.ndarray
        A 3D numpy array representing the functional connectivity data 
        with shape (subjects, regions, regions).
    mean_rms_vec : np.ndarray
        A 1D numpy array containing the mean RMS values for each subject.
        It should have the same length as the number of subjects in the 
        functional connectivity data. 
        The order of subjects in fc matrix should be the same as mean_rms_vec.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the QC-FC matrix with shape 
        (regions, regions), where each element is the correlation 
        coefficient between the FC data and the mean RMS vector.
    """

    qc_mat = np.zeros((fc.shape[1], fc.shape[2]))

    for i in range(fc.shape[1]):
        for t in range(fc.shape[2]):
            # Calculate the correlation between the FC measure and motion estimates
            qc_mat[i, t] = np.corrcoef(fc[:, i, t], mean_rms_vec)[0, 1]

    return qc_mat


def icc_matrix(data) -> np.ndarray:
    """
    Computes the ICC coefficient for all edges in the correlation matrices.
    
    Parameters:
    - data: np.ndarray of shape (K, N_subjects, N_rois, N_rois)
            where K is the number of conditions (e.g., open, close),
            N_subjects is the number of subjects,
            and N_rois is the number of regions of interest (ROIs).
    
    Returns:
    - ICC matrix of shape (N_rois, N_rois) containing ICC values for all edges.
    """
    K, N_subjects, N_rois, _ = data.shape

    # Compute the grand mean across all subjects and conditions for each edge
    grand_mean = data.mean(axis=(0, 1))  # Shape: (N_rois, N_rois)

    # Compute the subject means for each edge
    subject_means = data.mean(axis=0)  # Shape: (N_subjects, N_rois, N_rois)

    # Compute Between Mean Squares (BMS)
    BMS = K * np.sum((subject_means - grand_mean) ** 2, axis=0) / (N_subjects - 1)  # Shape: (N_rois, N_rois)

    # Compute Within Mean Squares (WMS)
    deviations = data - subject_means[np.newaxis, :, :, :]  # Shape: (K, N_subjects, N_rois, N_rois)
    WMS = np.sum(deviations ** 2, axis=(0, 1)) / (N_subjects * (K - 1))  # Shape: (N_rois, N_rois)

    # Compute ICC for all edges
    ICC = (BMS - WMS) / (BMS + (K - 1) * WMS + 0.0000001)
    np.fill_diagonal(ICC, 1.0)

    return ICC



def strategies_comparison(closed, opened) -> pd.DataFrame:
    """
    Compute the mean correlation coefficient between all pairs of subjects 
    for the 6 strategies in China dataset.

    Parameters
    ----------
    closed : np.ndarray
        A 3D numpy array representing the functional connectivity data 
        from the closed eyes condition with shape (strategies, subjects, regions, regions).
    opened : np.ndarray
        A 3D numpy array representing the functional connectivity data 
        from the open eyes condition with shape (strategies, subjects, regions, regions).

    Notes
    -----
    Strategies shoulb ordered: '24P', 'aCompCor+12P', 'aCompCor50+12P', 
    'aCompCor+24P', 'aCompCor50+24P', 'a/tCompCor50+24P' + same with GSR

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mean correlation coefficients between all pairs of subjects 
        for the 6 strategies in China dataset.
    """
    data = {}
    strategies = ['24P', 'aCompCor+12P', 'aCompCor50+12P', 
                  'aCompCor+24P', 'aCompCor50+24P', 'a/tCompCor50+24P']
    
    for i in range(len(strategies)):
        data[f'close_{strategies[i]}'] = closed[i]
        data[f'open_{strategies[i]}'] = opened[i]
        
        data[f'close_{strategies[i]}_GSR'] = closed[i+6]
        data[f'open_{strategies[i]}_GSR'] = opened[i+6]
    
    k = sorted(data)
    print(k.keys())
    out = np.zeros((24, 24))
    for en, i in enumerate(k):
        for en2, t in enumerate(k):
            out[en, en2] = np.mean(
                [np.corrcoef(
                    data[i].reshape(len(data[i]), -1)[sub], 
                    data[t].reshape(len(data[i]), -1)[sub])[0, 1] 
                for sub in range(len(data[i]))]) # все люди
            
    # Create DataFrame for visualization        
    cols_closed, cols_open = [], []
    for i in strategies:
        cols_closed.extend([f'close_{i}', f'close_{i}_GSR'])
        cols_open.extend([f'open_{i}', f'open_{i}_GSR'])    

    cols_closed.extend(cols_open)    
    df = pd.DataFrame(data=out, columns=cols_closed, index=cols_closed)
    
    return df


    
    