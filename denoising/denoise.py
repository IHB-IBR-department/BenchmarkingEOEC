import os
import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from tqdm.notebook import tqdm


class Denoising:
    r""" 
    Confound regression using fmriprep output

    Attributes
    ----------
    strategy: denoising strategy to be used

    Methods
    -------
    denoise(sub)
        Performs denoising 

    fetch outputs(sub)
        Returns preprocessed time-series
    """

    def __init__(self, dataset, atlas,
                 strategy, n_compcor=None,
                 use_GSR=False,
                 smoothing=None):
        r"""
        Parameters
        ----------
        dataset: Dataset
            Dataset instance to denoise
        atlas: Atlas
            Atlas instance to be used
        strategy: int
            Strategy number:

            1. 24 parameters

            2. aCompCor + 12P

            3. aCompCor50 + 12P

            4. aCompCor + 24P

            5. aCompCor50 + 24P
            
            6. a/tCompCor50 + 24P


        n_compcor: int, optional
            Number of principal components to be used in compcor in strategy 2 and 4 (10 by default)
        use_GSR: bool, optional
            Whether to use Global signal regressors. If True, 4 parameters are used (False by default)
    
        smoothing: float, optional
            If not None, smoothing with passed kernel is applied (None by default)
        """

        self.dataset = dataset
        self.atlas = atlas
        self.masker = atlas.masker

        self.int_strategy = strategy
        #self.ica_aroma = dataset.ica_aroma

        assert strategy in [1, 2, 3, 4, 5, 6], 'Wrong strategy number (valid: 1-6)'
        #assert self.ica_aroma and strategy in [7, 8], 'ICA-AROMA can only be used with strategy 7 or 8'

        self.use_GSR = use_GSR
        self.smoothing = smoothing
        self.n_compcor = 10 if n_compcor is None else n_compcor

    @property
    def strategy(self):
        # 24P
        strategy_1 = {'strategy': ['motion'],
                      'motion': 'full',
                      }
        # compcor, 12p
        strategy_2 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'derivatives',
                      'compcor': 'anat_combined',
                      'n_compcor': self.n_compcor}
        # compcor50, 12p
        strategy_3 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'derivatives',
                      'compcor': 'anat_combined',
                      'n_compcor': 'all'}
        # compcor, 24p
        strategy_4 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'full',
                      'compcor': 'anat_combined',
                      'n_compcor': self.n_compcor}
        # compcor50, 24p
        strategy_5 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'full',
                      'compcor': 'anat_combined',
                      'n_compcor': 'all'}

        strategy_6 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'full',
                      'compcor': 'temporal_anat_combined',
                      'n_compcor': 'all'}

        strategy = [strategy_1, strategy_2, strategy_3,
                    strategy_4, strategy_5, strategy_6][self.int_strategy-1]

        if self.use_GSR:
            strategy['strategy'].append('global_signal')
            strategy['global_signal'] = 'full'  # 4p


        if self.smoothing is not None:
            self.masker.set_params(smoothing_fwhm=self.smoothing)

        return strategy

    def denoise(self, sub=None, save_outputs=True, folder=None):
        """ 
        Denoising process

        Parameters
        ----------
        sub: list of str, optional
            List of subject labels to process if provided. Otherwise all subjects in the datased are processed (None by default)
        
        save_outputs: bool, optional
            whether to save outputs as csv files (True by default)
        
        folder: str or None, optional
            path to folder (otherwise outputs will be saved in derivatives path)

        Returns
        -------
        list of lists of np.arrays
            For every subject, for every run denoised time series (np.array of shape (n_timepoints, n_roi))
        
        """

        # use all the subjects defined in dataset
        if sub is None:
            sub = self.dataset.sub_labels

        denoised = []
        failed_subs = []

        for s in tqdm(sub):
            try:
                denoised.append(
                    self._denoise_one_sub(sub=s, save_outputs=save_outputs, folder=folder))
            except ValueError:
                failed_subs.append(s)
                continue
            except IndexError:
                failed_subs.append(s)
                continue

        if failed_subs:
            print(f'failed to process: {failed_subs}')

        return denoised
    

    def _denoise_one_sub(self, sub, save_outputs, folder):
        """
        Process and save one subject

        Parameters
        ----------
        sub: str
            Subject label
        
        Returns
        -------
        list
            List of len runs with time series (np.array) 
        """

        imgs = self.dataset.get_func_files(sub=sub)
        #masks = self.dataset.get_mask_files(sub=sub)
        #mask = 
        #print(len(masks), 'mask')
        #assert len(imgs) == self.dataset.runs, "All runs should be in one folder"

        confounds, _ = load_confounds(imgs, **self.strategy)
        denoised_ts  = []

        #runs = self.dataset.runs
        #sessions = 1 if self.dataset.sessions is None else self.dataset.sessions
        
        for i in range(len(imgs)):
            
            #self.masker.set_params(mask_img=masks[i])
            d = self.masker.fit_transform(imgs[i], confounds=confounds[i])

            # for one subject d.shape is (1, :, :)
            d = np.squeeze(d)

            denoised_ts.append(d)

            if save_outputs: # fix run and session in name
                _ = self._save_outputs(d, sub, run=i, folder=folder)

        return denoised_ts#, d

    def _save_outputs(self, outputs, sub, run, folder=None):
        """
        Saves processed time-series as csv files for every run

        Parameters
        ----------
        outputs: np.array
            Array with time-series
        sub: str
            Subject label without 'sub'
        run: int
            Run int

        Returns
        -------
        pd.DataFrame
            DataFrame where column names are roi labels
        """
        if folder is None:
            folder = self.dataset.derivatives

        path_to_save = os.path.join(folder, f'sub-{sub}',
                                    'time-series', self.atlas.atlas_name)
        
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        # TODO add GSR and smoothing to filename

        name = f'sub-{sub}_task-{self.dataset.task}_run-{run+1}_time-series_{self.atlas.atlas_name}_strategy-{self.int_strategy}{"_GSR" if self.use_GSR else ""}.csv'

        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(path_to_save, name), index=False)

        return df

    def fetch_timeseries(self, run, sub=None):
        # TODO
        """
        Fetch processed time-series one run

        Parameters
        ----------
        run: int
            Run number
        sub: list of str, optional
            Subject labels to fetch data from

        Returns
        -------
        list of np.arrays

        """

        if sub is None:
            sub = self.dataset.sub_labels

        ts = []
        for i in sub:
            name = f'sub-{i}_task-{self.dataset.task}_run-{run}_time-series_{self.atlas.atlas_name}_strategy-{self.int_strategy}.csv'
            path = os.path.join(
                self.dataset.derivatives, f'sub-{i}', 'time-series', self.atlas.atlas_name, name)
            ts.append(pd.read_csv(path).values)

        return ts



