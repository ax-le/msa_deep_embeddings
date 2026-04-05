# msa_deep_embeddings

Drop of the code for the SMC submission "Unsupervised Evaluation of Deep Audio Embeddings for Music Structure Analysis". 

!warning
    The code is not properly formatted for open-source use, and may be unstable, sorry about that. In near future, I will make it better, but, as of now, experiments may be reproduced using the "run_experiments.py" file.

## Embedding computation
Note that embeddings and segmentation are handled in two different folders. We design experiments in that way mainly due to virtual environment problems. Indeed, all evaluated models do not use the same packages and versions; adding a model could then break the environment required for another one.

Hence, we first compute embeddings using a different venv for each model. We use `uv` for that matter, and advise potential users to do so, because it largely facilitated the creation/development/maintain of several different venv with different python versions.

For each model, we installed the small `embed_utils` package (using pip), contained in "Compute Embeddings" folder, than contained important functions for pre- and post-processing the embeddings.

## Music Structure Analysis
The main code for segmentation is contained in the `as_seg` package (https://gitlab.imt-atlantique.fr/a23marmo/autosimilarity_segmentation).

!warning
    Please note that the package is not updated on PyPi yet, so installing the package using the pip automatic install will surely result in errors. Instead, you should either install it from the git repo (using pip), or by downloading the sources locally and then install this version. Sorry for the inconvenience.

## Troubleshooting
Please open issues if you encounter problems, or send me mail at axel.marmoret at imt-atlantique.fr. Many thanks!