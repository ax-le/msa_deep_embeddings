import os
import librosa
import numpy as np
import embed_utils.utils_for_codecs as utils

DATASET_DEFAULT_PATH = "/Brain/public/datasets/MIR"
CACHE_DEFAULT_PATH = "/Brain/private/a23marmo/projects/cbm_embeddings/cache"


def compute_barwise_embeddings(
    embed_fn,
    model_name,
    dataset_name,
    model,
    processor=None,
    sr=44100,
    dataset_root=DATASET_DEFAULT_PATH,
    cache_root=CACHE_DEFAULT_PATH,
    time_reduction_method="mean",
    time_axis=None,
    save_suffix="",
    verbose=True,
    **embed_kwargs,
):
    """
    Compute barwise embeddings for every audio file in a dataset.

    Parameters
    ----------
    embed_fn : callable
        ``embed_fn(signal, model, processor=..., **kwargs) -> tensor/array``
    model_name : str
        Name of the embedding model (used for the save path).
    dataset_name : str
        Name of the dataset.
    model : object
        The loaded model object, passed directly to *embed_fn*.
    processor : object, optional
        An optional processor/tokenizer, passed directly to *embed_fn*.
    sr : int, optional
        Sampling rate for ``librosa.load`` (default 44100).
    embeddings_root : str, optional
        Root directory where embeddings are saved.
    cache_root : str, optional
        Root directory where cache files (bars, etc.) are stored.
    save_suffix : str, optional
        Suffix appended to saved filenames (default ``""``).
        Useful to distinguish runs with different ``**embed_kwargs``.
    time_reduction_method : str, optional
        Method used to reduce time axes of embeddings (default ``"mean"``).
    verbose : bool, optional
        Whether to print progress.
    **embed_kwargs
        Extra keyword arguments forwarded to *embed_fn* (e.g. ``discrete=True``).
    """
    dataset_path = f"{dataset_root}/{dataset_name}/audio"
    bars_dir = f"{cache_root}/{dataset_name}/bars"
    save_embeddings_path = f"{cache_root}/{dataset_name}/embeddings/{model_name}"

    os.makedirs(save_embeddings_path, exist_ok=True)

    if verbose:
        print("Loading dataset...")
    dataset_file_paths = utils.load_dataset(dataset_path, dataset=dataset_name)

    previous_embeddings_shape = None

    if embed_kwargs:
        save_suffix += "_".join([f"{k}_{v}" for k, v in embed_kwargs.items()])

    if verbose:
        print("Parsing audio files...")
    for audio_idx, audio_path in enumerate(dataset_file_paths):
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file {audio_path} does not exist.")

        # Get song id
        song_id = audio_path.split("/")[-1].split(".")[0]

        # Skip if already computed
        save_path = f"{save_embeddings_path}/barwise_{song_id}_{save_suffix}_time_{time_reduction_method}.npy"
        if os.path.exists(save_path):
            if verbose:
                print(f"File {save_path} already exists. Skipping.")
            continue

        wv, loaded_sr = librosa.load(audio_path, sr=sr)

        # Print some info
        if verbose:
            print(
                f"Audio {audio_idx}/{len(dataset_file_paths)}: {audio_path}, "
                f"shape: {wv.shape}, SR: {loaded_sr}, length: {len(wv)/loaded_sr} seconds"
            )

        # Load bars
        bars = np.load(f"{bars_dir}/{song_id}.npy", allow_pickle=True)

        if bars.shape[0] == 0:
            if verbose:
                print(f"########################## No bars found for song {song_id}. Skipping.")
            continue

        # Cut signal on bars (barwise parts of the signal.)
        wv_cut_in_bars = utils.cut_signal_on_bars(wv, bars, loaded_sr)

        # Print some info
        if verbose:
            print(f"Bars shape: {bars.shape}, len wv_cut_in_bars: {len(wv_cut_in_bars)}")

        barwise_embeddings = _embed_bars_time(
            wv_cut_in_bars, model, processor, embed_fn, time_reduction_method, time_axis, verbose, **embed_kwargs
        )

        if verbose:
            print(f"Shape of barwise embeddings: {barwise_embeddings.shape}")

        # Check shape consistency across songs
        if time_reduction_method in ["mean", "max", "min"]:
            if previous_embeddings_shape is not None:
                assert barwise_embeddings.shape[1] == previous_embeddings_shape[1], f"Embeddings shape changed from {previous_embeddings_shape[1]} to {barwise_embeddings.shape[1]}, while the reduction method is {time_reduction_method}, thus expecting a constant shape among songs."
            previous_embeddings_shape = barwise_embeddings.shape

        np.save(save_path, barwise_embeddings)

    print("Done!")


def _embed_bars_time(wv_cut_in_bars, model, processor, embed_fn, time_reduction_method="mean", time_axis=None, verbose=True, **embed_kwargs):
    """Embed each bar with *embed_fn* and reduce varying time axes by averaging."""
    barwise_embeddings_gpu = []
    for i, bar in enumerate(wv_cut_in_bars):
        if verbose:
            print(f"Bar {i}/{len(wv_cut_in_bars)}")
        audio_codes = embed_fn(bar, model, processor=processor, **embed_kwargs)
        if verbose:
            print(f"Audio codes shape: {audio_codes.shape}, prod shapes: {np.prod(audio_codes.shape)}")
        barwise_embeddings_gpu.append(audio_codes)

    barwise_embeddings = [arr.cpu().numpy() for arr in barwise_embeddings_gpu]
    barwise_embeddings = utils.make_2D_array_from_different_length_arrays(
        barwise_embeddings, selection_mode=time_reduction_method, default_time_axis=time_axis
    )
    return barwise_embeddings
