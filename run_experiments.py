"""
Experiment runner for CBM segmentation on codec embeddings.

Evaluates all combinations of:
  - Embedding models (codicodec, music2latent) with their config variants
  - Similarity functions (cosine, autocorrelation, rbf, centered_rbf)
  - CBM penalty weights (0, 1)

Produces per-condition scores (precision, recall, F1) at two tolerances
(0.5s and 3s), averaged across songs with standard deviation.
Results are saved as a CSV and printed to the console.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from itertools import product
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from as_seg.CBM_algorithm import CBMEstimator
from as_seg.baseline_segmenter.baseline_estimators import FooteEstimator, LSDEstimator
from sklearn.base import BaseEstimator
from typing import Protocol
from msa_dataloader import RWCPopDataloader, SALAMIDataloader, HarmonixDataloader


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Which embedding file to load.

    Parameters
    ----------
    model : str
        Model name, e.g. "codicodec", "music2latent".
    configs : tuple of (str, value) pairs, optional
        Each pair is (config_key, config_value).  The tuple can be:
          - empty  → no config suffix  (filename: barwise_{track_id}.npy)
          - length 1 → one config       (filename: barwise_{track_id}_{key}_{val}.npy)
          - length N → N configs        (filename: barwise_{track_id}_{k1}_{v1}_{k2}_{v2}.npy)
    """
    model: str
    configs: tuple = ()   # tuple of (key, value) pairs – hashable & frozen-safe

    @property
    def display_name(self) -> str:
        """Human-readable model name: last path component (e.g. 'MERT-v1-95M')."""
        return self.model.split("/")[-1]

    @property
    def label(self) -> str:
        if not self.configs:
            return self.display_name
        cfg_str = ", ".join(f"{k}={v}" for k, v in self.configs)
        return f"{self.display_name}/{cfg_str}"

    @property
    def config_suffix(self) -> str:
        """The part of the filename that encodes the config, e.g. '_discrete_True'."""
        if not self.configs:
            return ""
        return "_" + "_".join(f"{k}_{v}" for k, v in self.configs)

    def embedding_filename(self, track_id: str) -> str:
        if self.config_suffix:
            return f"barwise_{track_id}{self.config_suffix}_time_mean.npy"
        return f"barwise_{track_id}__time_mean.npy"

    def embedding_path(self, track_id: str, base_path: str) -> str:
        return os.path.join(base_path, self.model, self.embedding_filename(track_id))


class EstimatorConfig(Protocol):
    @property
    def label(self) -> str: ...
    def build_estimator(self) -> BaseEstimator: ...

@dataclass(frozen=True)
class CBMEstimatorConfig:
    """How the CBM estimator is parameterised."""
    similarity: str
    penalty_weight: float
    bands_number: int | None  # None = full kernel, int = reduced to that many bands
    max_size: int = 32
    penalty_func: str = "modulo8"

    @property
    def label(self) -> str:
        return f"CBM_{self.similarity}_pw{self.penalty_weight}_bn{self.bands_number}"

    def build_estimator(self) -> BaseEstimator:
        return CBMEstimator(
            similarity_function=self.similarity,
            max_size=self.max_size,
            penalty_weight=self.penalty_weight,
            penalty_func=self.penalty_func,
            bands_number=self.bands_number,
        )

@dataclass(frozen=True)
class FooteEstimatorConfig:
    """How the Foote baseline estimator is parameterised."""
    similarity: str
    M_gaussian: int = 16  # placeholder – override via FOOTE_M_GAUSSIAN_VALUES
    L_peaks: int = 16     # placeholder – override via FOOTE_L_PEAKS_VALUES
    pre_filter: int = 0
    post_filter: int = 0

    @property
    def label(self) -> str:
        return f"Foote_{self.similarity}_M{self.M_gaussian}_L{self.L_peaks}"

    def build_estimator(self) -> BaseEstimator:
        return FooteEstimator(
            similarity_function=self.similarity,
            M_gaussian=self.M_gaussian,
            L_peaks=self.L_peaks,
            pre_filter=self.pre_filter,
            post_filter=self.post_filter,
        )

@dataclass(frozen=True)
class LSDEstimatorConfig:
    """How the LSD baseline estimator is parameterised."""
    scluster_k: int = 9  # placeholder – override via LSD_SCLUSTER_K_VALUES
    evec_smooth: int = 9
    rec_smooth: int = 1
    rec_width: int = 1

    @property
    def label(self) -> str:
        return f"LSD_k{self.scluster_k}_es{self.evec_smooth}_rs{self.rec_smooth}_rw{self.rec_width}"

    def build_estimator(self) -> BaseEstimator:
        return LSDEstimator(scluster_k=self.scluster_k, evec_smooth=self.evec_smooth, rec_smooth=self.rec_smooth, rec_width=self.rec_width)


@dataclass(frozen=True)
class ExperimentCondition:
    """A single, fully-specified experiment = embedding + estimator."""
    embedding: EmbeddingConfig
    estimator: EstimatorConfig

    @property
    def label(self) -> str:
        return f"{self.embedding.label}|{self.estimator.label}"


# ---------------------------------------------------------------------------
# Condition generation
# ---------------------------------------------------------------------------

# Each entry: (model_name, {config_key: [possible_values], ...})
# An empty dict means no config variants.
MODEL_CONFIGS: dict[str, dict[str, list]] = {
    "dac": {},
    "codicodec":     {"discrete": [True, False]},
    # "music2latent":  {"extract_features": [True, False]},
    "AudioMAE": {},
    "m2d": {"_space": ["audio", "audio_text"]}, 
    "matpac_plus": {},
    "MERT/m-a-p/MERT-v1-95M": {},
    "OpenMuQ": {"_space": ["audio", "audio_text"]},
    "musicfm": {},
    "CLAP/laion/clap-htsat-unfused": {},  # CLAP with the "htsat-unfused" config, which is the one we used in the paper

    # CLAP!
    # Example with no config:        "wavlm": {},
    # Example with several configs:  "dac": {"bitrate": ["8kbps", "24kbps"], "layer": [4, 8]},
}


def _expand_model_configs(model: str, config_axes: dict[str, list]) -> list[EmbeddingConfig]:
    """Expand a model's config axes into all EmbeddingConfig combinations."""
    if not config_axes:
        return [EmbeddingConfig(model)]
    keys = list(config_axes.keys())
    value_lists = [config_axes[k] for k in keys]
    return [
        EmbeddingConfig(model, tuple(zip(keys, combo)))
        for combo in product(*value_lists)
    ]


EMBEDDING_CONFIGS = [
    cfg
    for model, axes in MODEL_CONFIGS.items()
    for cfg in _expand_model_configs(model, axes)
]

SIMILARITIES = ["cosine", "rbf"]
PENALTY_WEIGHTS = [0]
BANDS_NUMBERS = [None, 7]  # None = full kernel, 7 = reduced to 7 bands

# Hyperparameter grids for baseline estimators
LSD_SCLUSTER_K_VALUES = [4, 6, 8, 9, 10, 11, 12, 13, 14, 16]
FOOTE_M_GAUSSIAN_VALUES = [8, 12, 16]
FOOTE_L_PEAKS_VALUES = [8, 12, 16]

# Fixed Foote params (placeholders – set your defaults here)
FOOTE_PRE_FILTER = 0
FOOTE_POST_FILTER = 0

TRIM_CONDITIONS = {
    "no_trim":              (False, False),   # (my_trim, mir_eval_trim)
    "mir_eval_trim":        (False, True),
    "my_trim":              (True, False),
    "my_trim+mir_eval_trim": (True,  True),
}


def get_trim_conditions(dataset_name: str) -> dict[str, tuple[bool | None, bool]]:
    """Return trim conditions for a dataset.

    Harmonix annotations are pre-trimmed at load time, so my_trim must always
    be None for that dataset.
    """
    if dataset_name == "harmonix":
        return {
            "no_trim": (None, False),
            "mir_eval_trim": (None, True),
        }
    return TRIM_CONDITIONS

# Map dataset name → (DataloaderClass, download, extra_kwargs)
DATASET_REGISTRY: dict[str, tuple] = {
    "rwcpop":  (RWCPopDataloader,   True, {}),
    "salami":  (SALAMIDataloader,   True,  {"subset": "test"}),
    "harmonix": (HarmonixDataloader, True, {}),
}


def build_all_conditions(include_baselines: bool = True) -> list[ExperimentCondition]:
    """Generate every combination of embedding config × estimator config."""
    conditions = []
    for emb in EMBEDDING_CONFIGS:
        for sim in SIMILARITIES:
            # CBM Conditions
            for pw in PENALTY_WEIGHTS:
                for bn in BANDS_NUMBERS:
                    conditions.append(
                        ExperimentCondition(emb, CBMEstimatorConfig(similarity=sim, penalty_weight=pw, bands_number=bn))
                    )

            # Foote Conditions
            if include_baselines:
                for M in FOOTE_M_GAUSSIAN_VALUES:
                    for L in FOOTE_L_PEAKS_VALUES:
                        conditions.append(
                            ExperimentCondition(emb, FooteEstimatorConfig(sim, M_gaussian=M, L_peaks=L))
                        )

        # LSD Conditions
        if include_baselines:
            for k in LSD_SCLUSTER_K_VALUES:
                conditions.append(
                    ExperimentCondition(emb, LSDEstimatorConfig(scluster_k=k, evec_smooth=k, rec_smooth=1, rec_width=1))
                )

    return conditions


# ---------------------------------------------------------------------------
# Spectrogram saving (optional)
# ---------------------------------------------------------------------------

def save_spectrogram(spec, save_path: str, title: str = "Spectrogram",
                     cmap=cm.Greys, invert_y_axis: bool = True):
    """Save a spectrogram as a PNG image."""
    figsize = (7, 7) if spec.shape[0] == spec.shape[1] else None
    if figsize:
        plt.figure(figsize=figsize)
    plt.pcolormesh(
        np.arange(spec.shape[1]), np.arange(spec.shape[0]), spec,
        cmap=cmap, shading="auto",
    )
    plt.title(title)
    plt.xlabel("Bars")
    plt.ylabel("Bars")
    if invert_y_axis:
        plt.gca().invert_yaxis()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Score collection helpers
# ---------------------------------------------------------------------------

SCORE_COLUMNS = [
    "prec_05", "rec_05", "f1_05",
    "prec_3", "rec_3", "f1_3",
]


def flatten_scores(close_tolerance: tuple, large_tolerance: tuple) -> list[float]:
    """Flatten the two (prec, rec, f1) tuples into a single 6-element list."""
    return list(close_tolerance) + list(large_tolerance)


def aggregate_scores(per_song_scores: list[list[float]]) -> dict:
    """Compute mean and std across songs for each metric."""
    arr = np.array(per_song_scores)  # shape (n_songs, 6)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    result = {"n_songs": len(per_song_scores)}
    for i, col in enumerate(SCORE_COLUMNS):
        result[f"{col}_mean"] = means[i]
        result[f"{col}_std"] = stds[i]
    return result


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    dataset_name: str,
    datasets_base_path: str,
    cache_path: str,
    plot_dir: str | None = None,
    results_dir: str = "results",
):
    """Run all experiment conditions for a single dataset and save results."""
    DataloaderClass, download, extra_kwargs = DATASET_REGISTRY[dataset_name]
    dataset_path = os.path.join(datasets_base_path, dataset_name)
    ds_cache_path = os.path.join(cache_path, dataset_name)
    embeddings_base_path = os.path.join(ds_cache_path, "embeddings")

    print(f"Loading dataset: {dataset_name}  ({dataset_path})")
    dataset = DataloaderClass(dataset_path, download=download,
                              cache_path=ds_cache_path, sr=44100, verbose=True, **extra_kwargs)

    print(f"Dataset loaded. Number of tracks: {len(dataset)}")

    conditions = build_all_conditions()
    print(f"Total conditions to evaluate: {len(conditions)}")

    # Pre-build estimators (one per unique EstimatorConfig)
    estimator_cache: dict[EstimatorConfig, BaseEstimator] = {}
    for cond in conditions:
        if cond.estimator not in estimator_cache:
            estimator_cache[cond.estimator] = cond.estimator.build_estimator()

    # Collect per-song scores: (condition_label, trim_key) -> list of 6-element score vectors
    trim_conditions = get_trim_conditions(dataset_name)
    trim_keys = list(trim_conditions.keys())
    all_scores: dict[tuple[str, str], list[list[float]]] = {
        (c.label, tk): [] for c in conditions for tk in trim_keys
    }

    for element in dataset:
        _, track, annotations_intervals, labels, len_signal = element
        if track is None:
            print("################ WARNING AXEL: Track is None, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Track: {track.track_id}  ({track.audio_path})")
        print(f"{'='*60}")

        bars = dataset.get_bars(track.audio_path, index=track.track_id)

        for cond in conditions:
            emb_cfg = cond.embedding

            emb_path = emb_cfg.embedding_path(track.track_id, embeddings_base_path)
            print(f"  Loading embeddings: {emb_path}")
            embeddings = np.load(emb_path, allow_pickle=True)
            estimator = estimator_cache[cond.estimator]

            # Predict once, then score under every trim condition
            try:
                segments = estimator.predict_in_seconds(embeddings, bars)
            except Exception as e:
                print(f"################ WARNING AXEL: ERROR during prediction for {cond.label} on track {track.track_id}: {e}")
                continue

            for trim_key, (my_trim_val, mir_trim_val) in trim_conditions.items():
                close_tol, large_tol = estimator.score(
                    segments, annotations_intervals,
                    trim=mir_trim_val,
                    my_trim_flag=my_trim_val,
                    len_signal=len_signal,
                    labels=labels,
                )
                scores = flatten_scores(close_tol, large_tol)
                all_scores[(cond.label, trim_key)].append(scores)

            # Print the mir_eval_trim variant as the representative
            rep_scores = all_scores[(cond.label, "mir_eval_trim")][-1]
            print(f"  {cond.label}  F1@0.5={rep_scores[2]:.3f}  F1@3={rep_scores[5]:.3f}")

    # ------------------------------------------------------------------
    # Aggregate and save results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean ± std across songs)")
    print(f"{'='*60}\n")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"scores_embeddings_{dataset_name}_{timestamp}.csv")

    # Determine the superset of config keys across all models
    all_config_keys = sorted({k for axes in MODEL_CONFIGS.values() for k in axes})

    csv_columns = [
        "model", *all_config_keys,
        "estimator", "similarity", "penalty_weight", "bands_number",
        "scluster_k", "M_gaussian", "L_peaks",
        "trim_condition",
        "n_songs",
    ]
    for col in SCORE_COLUMNS:
        csv_columns.extend([f"{col}_mean", f"{col}_std"])
    csv_columns.extend(["f05_mean", "f05_std", "f3_mean", "f3_std"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for cond in conditions:
            for trim_key in trim_keys:
                per_song = all_scores[(cond.label, trim_key)]
                if not per_song:
                    print(f" WARNING AXEL:  {cond.label} [{trim_key}]: NO DATA")
                    continue

                agg = aggregate_scores(per_song)
                cfg_dict = dict(cond.embedding.configs)
                row = {
                    "model": cond.embedding.display_name,
                    **{k: cfg_dict.get(k, "") for k in all_config_keys},
                    "estimator": cond.estimator.label,
                    "similarity": getattr(cond.estimator, "similarity", ""),
                    "penalty_weight": getattr(cond.estimator, "penalty_weight", ""),
                    "bands_number": getattr(cond.estimator, "bands_number", ""),
                    "scluster_k": getattr(cond.estimator, "scluster_k", ""),
                    "M_gaussian": getattr(cond.estimator, "M_gaussian", ""),
                    "L_peaks": getattr(cond.estimator, "L_peaks", ""),
                    "trim_condition": trim_key,
                    **agg,
                    "f05_mean": agg["f1_05_mean"],
                    "f05_std": agg["f1_05_std"],
                    "f3_mean": agg["f1_3_mean"],
                    "f3_std": agg["f1_3_std"],
                }
                writer.writerow(row)

                print(
                    f"  {cond.label:50s}  [{trim_key}]  "
                    f"n={agg['n_songs']:3d}  "
                    f"F1@0.5={agg['f1_05_mean']:.3f}±{agg['f1_05_std']:.3f}  "
                    f"F1@3={agg['f1_3_mean']:.3f}±{agg['f1_3_std']:.3f}"
                )

    print(f"\nResults saved to: {csv_path}")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run CBM segmentation experiments on codec embeddings.",
    )
    parser.add_argument(
        "--dataset", type=str,
        default="rwcpop",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to evaluate (default: rwcpop).",
    )
    parser.add_argument(
        "--datasets-base-path", type=str,
        default="/Brain/public/datasets/MIR",
        help="Root folder containing one sub-folder per dataset.",
    )
    parser.add_argument(
        "--cache-path", type=str,
        default="/Brain/private/a23marmo/projects/cbm_embeddings/cache",
        help="Cache path for the dataloader.",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save autosimilarity spectrograms to --plot-dir.",
    )
    parser.add_argument(
        "--plot-dir", type=str,
        default="/Brain/private/a23marmo/projects/cbm_embeddings/test_plots",
        help="Directory for spectrogram PNG files.",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/Brain/private/a23marmo/projects/cbm_embeddings/csv_results",
        help="Directory for CSV result files.",
    )
    args = parser.parse_args()

    run_experiments(
        dataset_name=args.dataset,
        datasets_base_path=args.datasets_base_path,
        cache_path=args.cache_path,
        # plot_dir=args.plot_dir if args.save_plots else None,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
