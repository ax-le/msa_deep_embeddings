# msa_deep_embeddings

Drop of the code for the SMC submission "Unsupervised Evaluation of Deep Audio Embeddings for Music Structure Analysis". 

### About code quality
The code is not properly formatted for open-source use, and may be unstable, sorry about that. In near future, I will make it better, but, as of now, experiments may be reproduced using the "run_experiments.py" file.

## Embedding computation
Note that embeddings and segmentation are handled in two different folders. We design experiments in that way mainly due to virtual environment problems. Indeed, all evaluated models do not use the same packages and versions; adding a model could then break the environment required for another one.

Hence, we first compute embeddings using a different venv for each model. We use `uv` for that matter, and advise potential users to do so, because it largely facilitated the creation/development/maintain of several different venv with different python versions.

For each model, we installed the small `embed_utils` package (using pip), contained in "Compute Embeddings" folder, than contained important functions for pre- and post-processing the embeddings.

### Download embeddings and bars
Embeddings are available at this zenodo link: TODO. (will be done after ICASSP 2026, so mid-May)

## Music Structure Analysis
The main code for segmentation is contained in the `as_seg` package (https://gitlab.imt-atlantique.fr/a23marmo/autosimilarity_segmentation).

### as_seg package version
The code must use a version higher than 1.12 of as_seg, because older versions will fail!

The good news is that the package is up-to-date on PyPi, so pip install should work flawlessly!

## Troubleshooting
Please open issues if you encounter problems, or send me mail at axel.marmoret at imt-atlantique.fr. Many thanks!
