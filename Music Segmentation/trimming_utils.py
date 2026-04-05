import numpy as np

# ---------------------------------------------------------------------------
# Silent-segment symbols (extend this list if needed)
# ---------------------------------------------------------------------------
SILENT_SYMBOLS = ["silent", "#", 'silence', 'end']


def escape_label(label):
    """Normalize a section label for robust silent-segment comparisons."""
    return str(label).replace("\n", "").strip().lower()


# ---------------------------------------------------------------------------
# Silent-segment trimming (annotations — label-based)
# ---------------------------------------------------------------------------

def trim_silent_segments(annotations, labels, len_signal=None):
    """Remove leading/trailing segments whose label is a silent symbol.

    Parameters
    ----------
    annotations : np.ndarray, shape (N, 2)
        Annotation intervals.
    labels : list of str
        Parallel list of labels for each annotation segment.
    len_signal : float, optional
        Duration of the signal. Can be used for optional edge-based trimming.

    Returns
    -------
    trimmed_annotations : np.ndarray
        Annotations with silent head/tail segments removed.
    trimmed_labels : list of str
        Corresponding labels after removal.
    """
    annotations = np.array(annotations, dtype=float)
    labels = list(labels)

    # Trim from the front
    while len(annotations) > 0 and escape_label(labels[0]) in SILENT_SYMBOLS:
        annotations = annotations[1:]
        labels = labels[1:]

    # Trim from the end
    while len(annotations) > 0 and escape_label(labels[-1]) in SILENT_SYMBOLS:
        annotations = annotations[:-1]
        labels = labels[:-1]

    return annotations, labels


# ---------------------------------------------------------------------------
# Segment trimming according to temporal information
# ---------------------------------------------------------------------------

def trim_according_to_length(segments, len_signal=None, tolerance=0.5):
    """
    Remove segments according to temporal information: 
        - If the segment starts at 0 (or very shotly after), remove it.
        - If the segments ends after the end of the song, or close ot it (with a tolerance), remove it.
    This function is used in order to trim the predicitons without needing to access annotations.
    """
    # # Remove first segment if it starts precisely at time 0
    while len(segments) > 0 and segments[0, 0] <= 1e-3:
        segments = segments[1:]
    
    # Remove last segment if it ends around the end of the bar/signal
    if len_signal is not None and len(segments) > 0:
        while len(segments) > 0 and segments[-1, 1] >= len_signal - tolerance:
            segments = segments[:-1]

    return segments

# ---------------------------------------------------------------------------
# Prediction trimming (to match trimmed annotations)
# It requires an access to annotations - not "fully unsupervised" in a way.
# ---------------------------------------------------------------------------

def trim_predictions_to_match(predictions, annotations):
    """Remove prediction segments outside the annotation time range.

    After annotations have been trimmed of silent segments, predictions may
    still contain segments in the now-removed silent zones.  This function
    removes any prediction segment whose *end* is <= the first annotation
    start, or whose *start* is >= the last annotation end.

    Parameters
    ----------
    predictions : np.ndarray, shape (M, 2)
    annotations : np.ndarray, shape (N, 2)
        Already-trimmed annotations.

    Returns
    -------
    np.ndarray
        Predictions with out-of-range segments removed.
    """
    predictions = np.array(predictions, dtype=float)
    annotations = np.array(annotations, dtype=float)

    if len(annotations) == 0 or len(predictions) == 0:
        return predictions

    annot_start = annotations[0, 0]
    annot_end = annotations[-1, 1]

    mask = (predictions[:, 1] > annot_start) & (predictions[:, 0] < annot_end)
    return predictions[mask]


# ---------------------------------------------------------------------------
# Synthetic silent-segment insertion (for my_trim=False)
# ---------------------------------------------------------------------------

def add_silent_segments_to_predictions(predictions, annotations, len_signal):
    """Ensure predictions span the same time range as (untrimmed) annotations.

    When ``my_trim=False``, the annotations include their silent head/tail
    segments.  Predictions typically do not cover those zones, so we add
    synthetic silent segments at the start and/or end of the predictions to
    make them consistent with the annotations.

    Parameters
    ----------
    predictions : np.ndarray, shape (M, 2)
    annotations : np.ndarray, shape (N, 2)
        Untrimmed annotations (may include silent segments).
    len_signal : float
        Duration of the audio signal in seconds.

    Returns
    -------
    np.ndarray
        Predictions with synthetic silent segments prepended/appended as
        needed.
    """
    predictions = np.array(predictions, dtype=float)
    annotations = np.array(annotations, dtype=float)

    if len(predictions) == 0:
        return predictions

    annot_start = annotations[0, 0] if len(annotations) > 0 else 0.0
    annot_end = annotations[-1, 1] if len(annotations) > 0 else len_signal

    pred_start = predictions[0, 0]
    pred_end = predictions[-1, 1]

    parts = []

    # Add a synthetic segment at the front if predictions don't cover it
    if pred_start > annot_start + 1e-3:
        parts.append(np.array([[annot_start, pred_start]]))

    parts.append(predictions)

    # Add a synthetic segment at the end if predictions don't cover it
    if pred_end < annot_end - 1e-3:
        parts.append(np.array([[pred_end, annot_end]]))

    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def apply_my_trim(annotations, predictions, labels, len_signal, my_trim):
    """Apply or reverse silent-segment trimming on both annotations and predictions.

    If annotations is a list of arrays (e.g. from multiple annotators), this
    will recursively process each one and return lists of annotations and predictions.
    """
    if isinstance(annotations, list) and (len(annotations) == 0 or isinstance(annotations[0], np.ndarray)):
        new_annotations, new_predictions = [], []
        for i in range(len(annotations)):
            na, np_preds = apply_my_trim(annotations[i], predictions, labels[i], len_signal, my_trim)
            new_annotations.append(na)
            new_predictions.append(np_preds)
        return new_annotations, new_predictions

    if my_trim: # Remove the silent segments.
        ## Remove them using the annotation labels in the annotations
        annotations, _ = trim_silent_segments(annotations, labels, len_signal=len_signal)

        ## Remove predicted segments that start near 0 and end near/after the length of the signal 
        predictions = trim_according_to_length(predictions, len_signal=len_signal)
        # predictions = trim_predictions_to_match(predictions, annotations) # This requires annotations, this is not desired.

    elif not my_trim: # my_trim may also be set to None, not just a boolean.
        # If my_trim is set to False, add silent segments to match annotations.
        predictions = add_silent_segments_to_predictions(predictions, annotations, len_signal)
    return annotations, predictions