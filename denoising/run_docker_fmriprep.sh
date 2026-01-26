docker run -i --rm \
    -v /arch/OpenCloseProject:/data:ro \
    -v /data/Projects/OpenCloseChina/derivatives_MNI152NLin6Asym:/out \
    -v /home/tm/new_space_work:/work \
    -v /home/tm/freesurfer:/license \
    nipreps/fmriprep:23.2.0 \
    /data /out participant \
    --nthreads 32 --fs-license-file /license/license.txt \
    --skull-strip-t1w auto --fs-no-reconall --skip-bids-validation \
    --work-dir /work --output-spaces MNI152NLin6Asym:res-02
