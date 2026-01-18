import re
import os
from pathlib import Path
import numpy as np
import pandas as pd

def load_data(closed_path, open_path, fc, atlas, strategy, gsr, icc_session=None):
    """
    Loads data from closed and opened eyes conditions given the functional connectivity measure,
    atlas, strategy, and global signal regression (gsr) parameters.

    Parameters
    ----------
    closed_path : str
        The path to the directory containing the functional connectivity matrices for the closed eyes condition.
    opened_path : str
        The path to the directory containing the functional connectivity matrices for the opened eyes condition.
    fc : str
        The functional connectivity measure to use ('corr', 'pc', 'tang', 'glasso').
    atlas : str
        The atlas to use (e.g., 'AAL', 'Schaefer200', 'Brainnetome', 'HCPex').
    strategy : int
        The strategy to use.
    gsr : str
        Whether to use global signal regression (GSR) or not (noGSR).
    icc_session : int, optional
        The session number to use for ICC metric (default is None).
        Only for EC.

    Returns
    -------
    cl_data, op_data or cl_data_1session, cl_data_2session : tuple
        A tuple containing the loaded data for the closed and open eyes conditions, respectively.
        Each element of the tuple is a numpy array containing the functional connectivity matrices.
    """
    # for closed eyes 
    # invered indx is for open
    sub_idx = {
            2: [1,  3,  5,  7,  9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 42, 43, 44, 46], 
            3: [0,  2,  4,  6,  8, 10, 12, 14, 16, 19, 21, 25, 27, 29, 31, 32, 34, 36, 38, 40, 45]
            }
    
    op, cl = os.listdir(open_path), os.listdir(closed_path)

    if icc_session is None:
        
        pattern_op2 = f'china_open2_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        pattern_op3 = f'china_open3_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        pattern_cl = f'china_close1_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'

        op_matches2 = [f for f in op if re.fullmatch(pattern_op2, f)]
        op_matches3 = [f for f in op if re.fullmatch(pattern_op3, f)]
        cl_matches = [f for f in cl if re.fullmatch(pattern_cl, f)]

        op_data2 = [np.load(f'{open_path}/{m}') for m in op_matches2]
        op_data3 = [np.load(f'{open_path}/{m}') for m in op_matches3]
        cl_data = [np.load(f'{closed_path}/{m}') for m in cl_matches]

        op_data = np.zeros((47, op_data2[0].shape[1], op_data2[0].shape[2]))
        op_data[sub_idx[3]] = op_data2[0]
        op_data[sub_idx[2]] = op_data3[0]

        op_data = np.delete(op_data, 23, axis=0)

        return np.concatenate(cl_data), op_data

    else:

        pattern_1 = f'china_close1_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        pattern_2 = f'china_close{icc_session}_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'

        cl_matches_1 = [f for f in cl if re.fullmatch(pattern_1, f)]
        cl_matches_2 = [f for f in cl if re.fullmatch(pattern_2, f)]

        cl_data_1 = np.load(f'{closed_path}/{cl_matches_1[0]}')
        cl_data_2 = np.load(f'{closed_path}/{cl_matches_2[0]}')

        cl_data_1 = np.insert(cl_data_1, 23, np.zeros_like(cl_data_2[0]), axis=0) 

        return cl_data_1[sub_idx[icc_session]], cl_data_2


def load_timeseries_stack(
    root_dir: str | Path,
    atlas: str = "Brainnetome",
    task: str = "rest",
    run: str = "1",
    strategy: str = "AROMA_aggrDenoised",
    gsr: str = "noGSR",
    dtype=np.float32,
):
    root_dir = Path(root_dir)

    # Под твой пример имени файла:
    # sub-001_task-rest_run-1_time-series_Brainnetome_strategy-AROMA_aggrDenoised-noGSR.csv
    fname = (
        f"sub-*_task-{task}_run-{run}_time-series_{atlas}_"
        f"strategy-{strategy}_{gsr}.csv"
    )

    pattern = f"sub-*/time-series/{atlas}/{fname}"
    files = sorted(root_dir.glob(pattern))  # glob по шаблону в подкаталогах [web:1][web:8]

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    subj_ids = []
    arrays = []
    shape0 = None

    for f in files:
        subj_ids.append(f.parts[f.parts.index("time-series") - 2] if "time-series" in f.parts else f.parent.parent.parent.name)
        arr = pd.read_csv(f).to_numpy(dtype=dtype)  # DataFrame -> numpy [web:6][web:9]

        if shape0 is None:
            shape0 = arr.shape
        elif arr.shape != shape0:
            raise ValueError(f"Shape mismatch: {f} has {arr.shape}, expected {shape0}")

        arrays.append(arr)

    X = np.stack(arrays, axis=0)  # (subjects, T, parcels) если CSV = T x parcels
    return X, subj_ids, files



def __load_data(closed, opened, fc, atlas, strategy, gsr, icc_session=None):
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
    icc_session : int, optional
        The session number to use for ICC metric (default is None).
        Only for EC.

    Returns
    -------
    cl_data, op_data : tuple
        A tuple containing the loaded data for the closed and open eyes conditions, respectively.
        Each element of the tuple is a numpy array containing the functional connectivity matrices.
    """
    
    if icc_session is None:
        op, cl = opened[fc], closed[fc]

        pattern_op = f'china_open\\d+_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        pattern_cl = f'china_close1_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        op_matches = [f for f in op.keys() if re.fullmatch(pattern_op, f)]
        cl_matches = [f for f in cl.keys() if re.fullmatch(pattern_cl, f)]
        
        op_data = [op[m] for m in op_matches]
        cl_data = [cl[m] for m in cl_matches]

        return np.concatenate(cl_data), np.concatenate(op_data)

    else:
        cl = closed[fc]

        pattern_1 = f'china_close1_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'
        pattern_2 = f'china_close{icc_session}_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy'

        cl_matches_1 = [f for f in cl.keys() if re.fullmatch(pattern_1, f)]
        cl_matches_2 = [f for f in cl.keys() if re.fullmatch(pattern_2, f)]

        cl_data_1 = [cl[m] for m in cl_matches_1]
        cl_data_2 = [cl[m] for m in cl_matches_2]

        sub_idx = {
            2: [1,  3,  5,  7,  9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 42, 43, 44, 46], 
            3: [0,  2,  4,  6,  8, 10, 12, 14, 16, 19, 21, 25, 27, 29, 31, 32, 34, 36, 38, 40, 45]
            }
        
        cl_data_1.insert(23, np.zeros_like(cl_data_2[0]))

        return np.concatenate(cl_data_1)[sub_idx[icc_session]], np.concatenate(cl_data_2)
