from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
import nibabel as nib
import numpy as np


def new_mask_gsr(data, path_to_save, thr=0.6):

    for sub in data.sub_labels:
        try:
            # read func and mask files 
            func_files = data.get_func_files(sub)[0]
            mask_files = data.get_mask_files(sub)[0]
            # get global signal from confounds file
            gs = data.get_confounds_one_subject(sub)[0].global_signal.values

            mskr = NiftiMasker(mask_img=mask_files)
            mskr.fit(func_files)
            tr = mskr.transform(func_files)
            
            # Check voxel signal at each timepoint
            # If it exceeds the global signal threshold, keep it
            a = np.array([tr[i] > (gs[i] * thr) for i in range(120)])
            t = tr * a

            out_mask = mskr.inverse_transform(np.sum(t, axis=0) != 0)
            nib.save(out_mask, f'{path_to_save}/sub-{sub}_run-1_new_mask.nii.gz')
        
        except IndexError:
            print(sub)
            continue

def coverage(atlas, mask):
    atlas_img = nib.load(atlas.atlas_path)
    masker_labels = atlas.atlas_labels
    

    # Convert all non-zero values to one
    # Create binary atlas for voxel counting
    atlas_img_bin = nib.Nifti1Image(
        (atlas_img.get_fdata() > 0).astype(np.uint8), 
        atlas_img.affine, 
        atlas_img.header,)


    sum_masker_masked = NiftiLabelsMasker(
                            labels_img=atlas_img,
                            labels=masker_labels,
                            background_label=0,
                            mask_img=mask,
                            smoothing_fwhm=None,
                            standardize=False,
                            strategy="sum",
                            resampling_target='data',  # Note: verify resampling behavior
                            )

    # no mask image here !!
    sum_masker_unmasked = NiftiLabelsMasker(
                            labels_img=atlas_img,
                            labels=masker_labels,
                            background_label=0,
                            smoothing_fwhm=None,
                            standardize=False,
                            strategy="sum", # sum to see number of 
                            resampling_target='data',
                            )


    # Pass the binarized atlas instead of brain image
    # Count how many voxels fall within the mask (sum voxels per ROI)
    n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
    # Count total voxels per ROI in the atlas
    n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
    # Calculate proportion of voxels within the mask
    parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)

    return parcel_coverage
