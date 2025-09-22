import re
import numpy as np

def load_data(closed, opened, fc, atlas, strategy, gsr):
    """
    Loads data from closed and opened eyes conditions given the functional connectivity measure,
    atlas, strategy, and global signal regression (gsr) parameters.

    Parameters
    ----------
    closed : dict
        A dictionary containing the functional connectivity matrices for the closed eyes condition.
    opened : dict
        A dictionary containing the functional connectivity matrices for the open eyes condition.
    fc : str
        The functional connectivity measure to use ('corr', 'pc', 'tang', 'glasso').
    atlas : str
        The atlas to use (e.g., 'AAL', 'Schaefer200', 'Brainnetome', 'HCPex').
    strategy : int
        The strategy to use.
    gsr : str
        Whether to use global signal regression (GSR) or not (noGSR).

    Returns
    -------
    cl_data, op_data : tuple
        A tuple containing the loaded data for the closed and open eyes conditions, respectively.
    Each element of the tuple is a numpy array containing the functional connectivity matrices.
    """
    op, cl = opened[fc], closed[fc]
    pattern_op = f'china_open\\d+_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
    pattern_cl = f'china_close\\d+_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
    op_matches = [f for f in op.keys() if re.fullmatch(pattern_op, f)]
    cl_matches = [f for f in cl.keys() if re.fullmatch(pattern_cl, f)]
    
    op_data = [op[m] for m in op_matches]
    cl_data = [cl[m] for m in cl_matches]
    
    return np.concatenate(cl_data), np.concatenate(op_data)
    
    