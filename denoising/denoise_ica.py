from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np

from bids import BIDSLayout
from nilearn.maskers import NiftiLabelsMasker

from collections.abc import Iterable
import os


@dataclass
class DenoiseAROMA:
    aroma_deriv_dir: str | Path          # деривативы fmripost-aroma (BIDS-derivatives dataset)
    fmriprep_deriv_dir: str | Path       # деривативы fmriprep (BIDS-derivatives dataset)
    atlas_labels_img: str | Path         # NIfTI atlas labels (например Schaefer)
    aroma_desc: str = "nonaggrDenoised"  # nonaggrDenoised/aggrDenoised/orthaggrDenoised
    space: str | None = "MNI152NLin6Asym"
    #res: str | None = None              # например "2" или "02" (зависит от именования)
    compcor_kind: str = "a"             # "a" (aCompCor) или "t" (tCompCor)
    n_compcor: int | None = None        # если None — берем все найденные компоненты
    standardize: str | None = "zscore_sample"

    def __post_init__(self):
        self.aroma_deriv_dir = Path(self.aroma_deriv_dir)
        self.fmriprep_deriv_dir = Path(self.fmriprep_deriv_dir)
        self.atlas_labels_img = str(self.atlas_labels_img)

        # PyBIDS: индексируем каждый derivatives dataset отдельно
        self.layout_aroma = BIDSLayout(str(self.aroma_deriv_dir), 
                                       config=['bids', 'derivatives'],
                                       validate=False)
        self.layout_fmriprep = BIDSLayout(str(self.fmriprep_deriv_dir), 
                                          config=['bids', 'derivatives'],
                                          validate=False)
        
        self.masker = NiftiLabelsMasker(
                            labels_img=self.atlas_labels_img,
                            memory="/data/workdir/.nilearn_cache",
                            verbose=1,
                            standardize=False, #'zscore_sample',
                            detrend=True,
                            resampling_target='data', #'labels'
                            n_jobs=1)

    @staticmethod
    def _select_confounds(df: pd.DataFrame,
                          compcor_kind: str = "a",
                          n_compcor: int | None = None) -> pd.DataFrame:
        """
        Выбирает WM/CSF CompCor + cosine drifts из fMRIPrep confounds TSV.
        Колонки fMRIPrep: a_comp_cor_00..., t_comp_cor_00..., cosine_00... [web:54]
        """
        # cosine drifts
        cosine_cols = [c for c in df.columns if c.startswith("cosine")]

        # CompCor компоненты (обычно a_comp_cor_XX или t_comp_cor_XX)
        if compcor_kind.lower().startswith("a"):
            compcor_re = re.compile(r"^a_comp_cor_\d+")
        else:
            compcor_re = re.compile(r"^t_comp_cor_\d+")

        compcor_cols = [c for c in df.columns if compcor_re.match(c)]
        compcor_cols = sorted(compcor_cols, key=lambda x: int(x.split("_")[-1]))

        if n_compcor is not None:
            compcor_cols = compcor_cols[:n_compcor]

        keep = compcor_cols + cosine_cols
        if len(keep) == 0:
            raise ValueError("Не найдены ни CompCor (a_comp_cor_/t_comp_cor_), ни cosine_* колонки в confounds TSV.")

        # Заполняем NaN, чтобы регрессия не падала
        out = df[keep].copy()
        out = out.fillna(0.0)
        return out

    def _find_aroma_bold(self, subject: str, 
                         task: str | None = None, 
                         run: str | None = None):
        """
        Ищет AROMA denoised BOLD (NIfTI) по BIDS-entities.
        """
        filters = dict(
            subject=subject,
            datatype="func",
            suffix="bold",
            extension=[".nii", ".nii.gz"],
            desc=self.aroma_desc,
        )
        if task is not None:
            filters["task"] = task
        if run is not None:
            filters["run"] = run
        if self.space is not None:
            filters["space"] = self.space
        #if self.res is not None:
            #filters["res"] = self.res

        files = self.layout_aroma.get(return_type="file", **filters)
        if len(files) == 0:
            raise FileNotFoundError(f"Не найден AROMA BOLD по фильтрам: {filters}")
        if len(files) > 1:
            raise RuntimeError(f"Найдено несколько AROMA BOLD файлов, уточните фильтры (task/run/space/res): {files}")
        return files[0]

    def _find_fmriprep_confounds(self, subject: str, task: str | None = None, run: str | None = None):
        """
        Ищет fMRIPrep confounds TSV, соответствующий этому run/task.
        """
        filters = dict(
            subject=subject,
            datatype="func",
            suffix="timeseries",
            desc="confounds",
            extension=".tsv",
        )
        if task is not None:
            filters["task"] = task
        if run is not None:
            filters["run"] = run
        if self.space is not None:
            # confounds TSV у fMRIPrep обычно без space, но иногда встречается; поэтому не фильтруем по space жестко
            pass

        files = self.layout_fmriprep.get(return_type="file", **filters)
        if len(files) == 0:
            raise FileNotFoundError(f"Не найден fMRIPrep confounds TSV по фильтрам: {filters}")
        if len(files) > 1:
            # Если несколько — попробуем выбрать тот, где совпадает session/acq и т.п. при необходимости
            # Здесь — просто требуем уточнить.
            raise RuntimeError(f"Найдено несколько confounds TSV, уточните фильтры (task/run/session/acq): {files}")
        return files[0]

    def denoise_one_subject(self,
                            subject: str,
                            task: str | None = None,
                            run: str | None = None,
                            save_outputs: bool = False,
                            folder: str | None = None) -> tuple[np.ndarray, dict]:
        """
        Возвращает (roi_ts, info), где roi_ts: (T, Nroi)
        """
        bold = self._find_aroma_bold(subject=subject, task=task, run=run)
        conf_tsv = self._find_fmriprep_confounds(subject=subject, task=task, run=run)

        conf_df = pd.read_csv(conf_tsv, sep="\t")
        conf_sel = self._select_confounds(conf_df, compcor_kind=self.compcor_kind, n_compcor=self.n_compcor)

    
        # NiftiLabelsMasker регрессирует confounds, переданные в fit_transform() [web:47]
        roi_ts = self.masker.fit_transform(bold, confounds=conf_sel.to_numpy())

        info = {
            "bold_file": bold,
            "confounds_file": conf_tsv,
            "confounds_columns": list(conf_sel.columns),
            "n_timepoints": roi_ts.shape[0],
            "n_rois": roi_ts.shape[1],
        }

        if save_outputs:
            _ = self._save_outputs(roi_ts, sub=subject, 
                                   run=run, task=task, 
                                   folder=folder)
        return roi_ts#, info
    

    def list_subjects(self) -> list[str]:
        return self.layout_aroma.get_subjects()  # or layout_fmriprep.get_subjects() [web:71]

    def list_runs(self, subject: str, task: str | None = None) -> list[str | None]:
        # return unique run IDs present; if dataset has no run entity, returns [None]
        q = dict(subject=subject, datatype="func", suffix="bold", desc=self.aroma_desc,
                 extension=[".nii", ".nii.gz"])
        if task is not None:
            q["task"] = task
        if self.space is not None:
            q["space"] = self.space
        #if self.res is not None:
            #q["res"] = self.res

        runs = self.layout_aroma.get(return_type="id", target="run", **q)  # PyBIDS entity listing [web:42]
        return runs if len(runs) > 0 else [None]

    def denoise_many(self,
                     subjects: Iterable[str] | None = None,
                     task: str | None = None,
                     save_outputs: bool = False,
                     folder: str | None = None) -> list[dict]:
        """
        Возвращает список info-словарей (по subject/run), а не один.
        """
        if subjects is None:
            subjects = self.list_subjects()

        results, failed_subs = [], []
        for sub in subjects:
            try:
                onesub = []
                for run in self.list_runs(subject=sub, task=task):
                    roi_ts = self.denoise_one_subject(subject=sub, 
                                                      task=task, 
                                                      run=run, 
                                                      save_outputs=save_outputs, 
                                                      folder=folder)
                    onesub.append(roi_ts)
                results.append(onesub)

            except ValueError:
                failed_subs.append(sub)
                continue

            except IndexError:
                failed_subs.append(sub)

        if failed_subs:
            print(f'failed to process: {failed_subs}')

        return results


    def _save_outputs(self, outputs, sub, run, task, folder=None):
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

        atlases =    {116: "AAL",
                      200: "Schaefer200",
                      246: "Brainnetome",
                      426: "HCPex"}
        
        atlas_name = atlases[outputs.shape[1]]
        
        path_to_save = os.path.join(folder, f'sub-{sub}',
                                    'time-series', atlas_name)
        
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        # TODO добавить в название файла GSR и smoothing

        name = f'sub-{sub}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-AROMA_{self.aroma_desc}-noGSR.csv'

        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(path_to_save, name), index=False)

        print(f"-----------Saved {sub}----------------")

        return df
