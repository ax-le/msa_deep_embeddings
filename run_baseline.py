"""
Baseline experiment runner for structural segmentation.

Evaluates baseline methods (LSD, Foote, CBM on handcrafted barwise features)
across similarity functions, CBM penalties, and trim conditions.

Produces per-track scores and grouped averages, then writes both to CSV.
"""

import argparse
import os
from datetime import datetime

import pandas as pd

import as_seg.autosimilarity_computation as as_comp
import as_seg.barwise_input as bi
from as_seg.CBM_algorithm import CBMEstimator
from as_seg.baseline_segmenter.baseline_estimators import FooteEstimator, LSDEstimator
from msa_dataloader import HarmonixDataloader, RWCPopDataloader, SALAMIDataloader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIMILARITIES = ["cosine", "rbf"]
PENALTY_WEIGHTS = [0]
BANDS_NUMBERS = [None, 7]  # None = full kernel, int = reduced to that many bands

# Hyperparameter grids for baseline estimators
LSD_SCLUSTER_K_VALUES = [4, 6, 8, 9, 10, 11, 12, 13, 14, 16]
FOOTE_M_GAUSSIAN_VALUES = [8, 12, 16]
FOOTE_L_PEAKS_VALUES = [8, 12, 16]

# Fixed Foote params (placeholders – set your defaults here)
FOOTE_PRE_FILTER = 0
FOOTE_POST_FILTER = 0

TRIM_CONDITIONS = {
    "no_trim": (False, False),
    "mir_eval_trim": (False, True),
    "my_trim": (True, False),
    "my_trim+mir_eval_trim": (True, True),
}

# Map dataset name -> (DataloaderClass, download, extra_kwargs)
DATASET_REGISTRY: dict[str, tuple] = {
    "rwcpop": (RWCPopDataloader, True, {}),
    "salami": (SALAMIDataloader, True, {"subset": "test"}),
    "harmonix": (HarmonixDataloader, True, {}),
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_all_trims(
    estimator,
    segments_time,
    annotations_intervals,
    labels,
    len_signal,
    track_id: str,
    trim_conditions: dict[str, tuple[bool | None, bool]],
    method: str,
    similarity: str | None = None,
    penalty: float | None = None,
    bands: int | None = None,
    scluster_k: int | None = None,
    M_gaussian: int | None = None,
    L_peaks: int | None = None,
) -> list[dict]:
    """Score one prediction under every trim condition."""
    rows = []
    for trim_key, (my_trim, mir_trim) in trim_conditions.items():
        sc, sl = estimator.score(
            segments_time,
            annotations_intervals,
            trim=mir_trim,
            my_trim_flag=my_trim,
            len_signal=len_signal,
            labels=labels,
        )
        rows.append(
            {
                "track_id": track_id,
                "method": method,
                "similarity": similarity,
                "penalty": penalty,
                "bands": bands,
                "scluster_k": scluster_k,
                "M_gaussian": M_gaussian,
                "L_peaks": L_peaks,
                "trim": trim_key,
                "precision_close": sc[0],
                "recall_close": sc[1],
                "f05": sc[2],
                "precision_large": sl[0],
                "recall_large": sl[1],
                "f3": sl[2],
            }
        )
        print(f"  [{trim_key}]  F1@0.5={sc[2]:.4f}  F1@3={sl[2]:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_baseline(
    dataset_name: str,
    datasets_base_path: str,
    cache_path: str,
    results_dir: str,
    subdivision: int,
    hop_length: int,
    sr: int,
):
    """Run handcrafted-feature baselines for one dataset and save results."""
    hop_length_secs = hop_length / sr
    dataset_path = os.path.join(datasets_base_path, dataset_name)
    ds_cache_path = os.path.join(cache_path, dataset_name)
    trim_conditions = get_trim_conditions(dataset_name)

    DataloaderClass, download, extra_kwargs = DATASET_REGISTRY[dataset_name]
    print(f"Loading dataset: {dataset_name}  ({dataset_path})")
    dataset = DataloaderClass(
        dataset_path,
        download=download,
        cache_path=ds_cache_path,
        sr=sr,
        verbose=True,
        **extra_kwargs,
    )
    total_tracks = len(dataset)
    print(f"Dataset loaded. Number of tracks: {total_tracks}")

    lsd_estimators = {k: LSDEstimator(scluster_k=k, evec_smooth=k, rec_smooth=1, rec_width=1) for k in LSD_SCLUSTER_K_VALUES}
    foote_estimators = {
        (sim, M, L): FooteEstimator(
            similarity_function=sim,
            M_gaussian=M,
            L_peaks=L,
            pre_filter=FOOTE_PRE_FILTER,
            post_filter=FOOTE_POST_FILTER,
        )
        for sim in SIMILARITIES
        for M in FOOTE_M_GAUSSIAN_VALUES
        for L in FOOTE_L_PEAKS_VALUES
    }
    cbm_estimators = {
        (sim, p_weight, bands): CBMEstimator(
            similarity_function=sim,
            penalty_weight=p_weight,
            bands_number=bands,
        )
        for sim in SIMILARITIES
        for p_weight in PENALTY_WEIGHTS
        for bands in BANDS_NUMBERS
    }

    all_results: list[dict] = []
    tracks_skipped_none = 0
    tracks_seen = 0
    tracks_with_output = 0
    tracks_without_output = 0

    for sig, track, annotations_intervals, labels, len_signal in dataset:
        if track is None:
            print("################ WARNING AXEL: Track is None, skipping")
            tracks_skipped_none += 1
            continue

        tracks_seen += 1
        rows_before_track = len(all_results)

        track_id = track.track_id
        print(f"\n{'=' * 60}")
        print(f"Track: {track_id}  ({track.audio_path})")
        print(f"{'=' * 60}")

        try:
            bars = dataset.get_bars(track.audio_path, index=track_id)
            spectrogram = dataset.get_spectrogram(sig, feature="log_mel", hop_length=hop_length)
            barwise_tf = bi.barwise_TF_matrix(
                spectrogram,
                bars,
                hop_length_seconds=hop_length_secs,
                subdivision=subdivision,
                subset_nb_bars=None,
            )
        except ValueError as e:
            print(f"################ WARNING AXEL: error while computing barwise TF for {track_id}: {e}")
            barwise_tf = None

        if barwise_tf is None:
            tracks_without_output += 1
            continue

        for k in LSD_SCLUSTER_K_VALUES:
            lsd_estimator = lsd_estimators[k]
            print(f"  Running LSD [k={k}]")
            try:
                lsd_segments = lsd_estimator.predict_in_seconds(barwise_tf, bars)
                all_results.extend(
                    evaluate_all_trims(
                        estimator=lsd_estimator,
                        segments_time=lsd_segments,
                        annotations_intervals=annotations_intervals,
                        labels=labels,
                        len_signal=len_signal,
                        track_id=track_id,
                        trim_conditions=trim_conditions,
                        method="LSD",
                        scluster_k=k,
                    )
                )
            except Exception as e:
                print(f"################ WARNING AXEL: ERROR during LSD prediction for {track_id} [k={k}]: {e}")


        for similarity in SIMILARITIES:
            ssm = as_comp.switch_autosimilarity(barwise_tf, similarity)
            for M in FOOTE_M_GAUSSIAN_VALUES:
                for L in FOOTE_L_PEAKS_VALUES:
                    print(f"  Running Foote [{similarity}, M={M}, L={L}]")
                    foote_estimator = foote_estimators[(similarity, M, L)]
                    try:
                        foote_segments = foote_estimator.predict_in_seconds_this_autosimilarity(ssm, bars)

                        all_results.extend(
                            evaluate_all_trims(
                                estimator=foote_estimator,
                                segments_time=foote_segments,
                                annotations_intervals=annotations_intervals,
                                labels=labels,
                                len_signal=len_signal,
                                track_id=track_id,
                                trim_conditions=trim_conditions,
                                method="Foote",
                                similarity=similarity,
                                M_gaussian=M,
                                L_peaks=L,
                            )
                        )

                    except Exception as e:
                        print(f"################ WARNING AXEL: ERROR during Foote prediction for {track_id} [sim={similarity}, M={M}, L={L}]: {e}")


            for penalty in PENALTY_WEIGHTS:
                for bands in BANDS_NUMBERS:
                    print(f"  Running CBM [{similarity}, p={penalty}, bands={bands}]")
                    cbm_estimator = cbm_estimators[(similarity, penalty, bands)]
                    try:
                        cbm_segments = cbm_estimator.predict_in_seconds_this_autosimilarity(
                            ssm_matrix=ssm,
                            bars=bars,
                        )

                        all_results.extend(
                            evaluate_all_trims(
                                estimator=cbm_estimator,
                                segments_time=cbm_segments,
                                annotations_intervals=annotations_intervals,
                                labels=labels,
                                len_signal=len_signal,
                                track_id=track_id,
                                trim_conditions=trim_conditions,
                                method="CBM",
                                similarity=similarity,
                                penalty=penalty,
                                bands=bands,
                            )
                        )

                    except Exception as e:
                        print(
                            f"################ WARNING AXEL: ERROR during CBM prediction for "
                            f"{track_id} [sim={similarity}, p={penalty}, bands={bands}]: {e}"
                        )

        if len(all_results) > rows_before_track:
            tracks_with_output += 1
        else:
            tracks_without_output += 1

    if not all_results:
        print("No results were produced. Exiting without writing CSV files.")
        print(
            f"Run counts: total={total_tracks}, seen={tracks_seen}, "
            f"with_output={tracks_with_output}, without_output={tracks_without_output}, "
            f"skipped_none={tracks_skipped_none}"
        )
        return

    df = pd.DataFrame(all_results)

    group_keys = ["method", "similarity", "penalty", "bands", "scluster_k", "M_gaussian", "L_peaks", "trim"]
    averages = (
        df.groupby(group_keys, dropna=False)
        .agg(
            n_songs=("track_id", "nunique"),
            f05=("f05", "mean"),
            f05_std=("f05", "std"),
            f3=("f3", "mean"),
            f3_std=("f3", "std"),
        )
        .reset_index()
    )
    averages["tracks_in_dataset"] = total_tracks
    averages["tracks_seen"] = tracks_seen
    averages["tracks_with_output"] = tracks_with_output
    averages["tracks_without_output"] = tracks_without_output
    averages["tracks_skipped_none"] = tracks_skipped_none

    print(f"\n{'=' * 60}")
    print("AGGREGATED RESULTS (mean across tracks)")
    print(f"{'=' * 60}")
    print(
        f"Run counts: total={total_tracks}, seen={tracks_seen}, "
        f"with_output={tracks_with_output}, without_output={tracks_without_output}, "
        f"skipped_none={tracks_skipped_none}"
    )
    print(averages.to_string(index=False))

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # per_track_path = os.path.join(results_dir, f"scores_baseline_tracks_{dataset_name}_{timestamp}.csv")
    averages_path = os.path.join(results_dir, f"scores_baseline_summary_{dataset_name}_{timestamp}.csv")

    # df.to_csv(per_track_path, index=False)
    averages.to_csv(averages_path, index=False)
    # print(f"\nPer-track results saved to: {per_track_path}")
    print(f"Summary results saved to:   {averages_path}")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline segmentation experiments on handcrafted barwise features.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rwcpop",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to evaluate (default: rwcpop).",
    )
    parser.add_argument(
        "--datasets-base-path",
        type=str,
        default="/Brain/public/datasets/MIR",
        help="Root folder containing one sub-folder per dataset.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/Brain/private/a23marmo/projects/cbm_embeddings/cache",
        help="Cache path root for the dataloaders.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Brain/private/a23marmo/projects/cbm_embeddings/csv_results",
        help="Directory for CSV result files.",
    )
    parser.add_argument(
        "--subdivision",
        type=int,
        default=96,
        help="Number of frames per bar for barwise features.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=32,
        help="Hop length (in samples) used for the log-mel spectrogram.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate used by the dataloaders and feature extraction.",
    )

    args = parser.parse_args()
    run_baseline(
        dataset_name=args.dataset,
        datasets_base_path=args.datasets_base_path,
        cache_path=args.cache_path,
        results_dir=args.results_dir,
        subdivision=args.subdivision,
        hop_length=args.hop_length,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()