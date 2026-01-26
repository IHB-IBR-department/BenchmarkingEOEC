docker run -i --rm \
    -v /data/Projects/OpenCloseChina/derivatives_MNI152NLin6Asym:/data:ro \
    -v /data/Projects/OpenCloseChina/derivatives_MNI152NLin6Asym/aroma:/out \
    -v /data/workdir:/h \
    nipreps/fmripost-aroma:main \
    /data /out participant \
    -t rest --skip-bids-validation \
    --nthreads 32 --denoising-method nonaggr aggr \
    --work-dir /h
