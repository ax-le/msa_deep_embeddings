"""
Utils for computing embeddings, such as loading datasets, cutting the signal according to bars, etc.
"""

import os
import copy
import numpy as np

def load_dataset(dataset_path, dataset="rwcpop"): 
    # Load all data paths.
    # Use this instead of mirdata, to avoid installing the package. Ugly, but works.
    paths_lists = []
    if dataset in ["rwcpop", "salami", "harmonix"]:
        for file in os.listdir(dataset_path):
            if file.endswith(".mp3") or file.endswith(".wav"):
                paths_lists.append(f"{dataset_path}/{file}")
        if dataset == "rwcpop":
            print(f"Found {len(paths_lists)} path for rwcpop, expected 100.")
        elif dataset == "salami":
            print(f"Found {len(paths_lists)} path for salami, expected around 1420.")
        elif dataset == "Harmonix":
            print(f"Found {len(paths_lists)} path for harmonix, expected around 1000.")
    else:
        raise ValueError(f"Dataset {dataset} not found.")
        
    return paths_lists

def bars_in_time_to_samples(bars, sr):
    # Converts bar times in frame indices.
    return np.array([[int(bar[0]*sr), int(bar[1]*sr)] for bar in bars])

def cut_signal_on_bars(signal, bars, sr):
    # Cut the signal in barwise excerpts, that will be embeded individually.
    assert bars.shape[0] > 0, "Bars list is empty."
    barwise_signal = []
    if bars[0][0] == 0: # or (bars[0][1] - bars[0][0] < 0.5):
        bars = bars[1:] # Remove the first bar if it starts at 0, because this is probably a silenced first bar.
    
    bars_in_samples = bars_in_time_to_samples(bars, sr)
    
    for this_bar in bars_in_samples:
        t_0 = this_bar[0]
        t_1 = this_bar[1]
        barwise_signal.append(signal[t_0:t_1])
    return barwise_signal

def make_2D_array_from_different_length_arrays(list_of_arrays, selection_mode="subsample_uniform", default_time_axis=None):
    """Build a 2D matrix (n_bars, features) from a list of per-bar arrays.

    Each element of `list_of_arrays` is one bar's data (arbitrary shape).
    Axes whose size is *the same* across all bars are preserved.
    Axes whose size *varies* are reduced according to `selection_mode`.
    The result is then flattened per bar to produce a (n_bars, features) matrix.

    selection_mode : str
        "subsample_uniform" | "average_pool" — resample varying axes to their min size.
        "mean" | "max" | "min" | "sum"       — collapse varying axes entirely.
    """
    if not list_of_arrays:
        raise ValueError("List of arrays is empty.")

    all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(list_of_arrays)

    if not varying_axes: # This is problem for some subsample methods like mean max, min, where the time axis needs to be collapsed somehow to have the same shape accross all songs.
        if selection_mode in ["mean", "max", "min"]:
            if default_time_axis is None:
                raise ValueError(f"A reduction mode should be applied (mean, max, min), but no varying axes found. You have to provide a default time axis (param default_time_axis). Info: {selection_mode} was chosen, the array shape is {all_shapes[0]}")
            else:
                varying_axes = [default_time_axis]
        else: # Subsample_uniform or average_pool, we don't care about the time axis, we just want to have the same shape accross all bars.
            return np.array([arr.flatten() for arr in list_of_arrays])

    # Reduce each bar along varying axes
    target = {ax: min(s[ax] for s in all_shapes) for ax in varying_axes}
    processed = []
    for arr in list_of_arrays:
        for ax in sorted(varying_axes, reverse=True):
            arr = _reduce_axis(arr, ax, target[ax], selection_mode)
        processed.append(arr.flatten())

    return np.array(processed)


def _reduce_axis(arr, axis, target_size, mode):
    """Reduce `arr` along `axis` to `target_size` using `mode`.

    For "subsample_uniform" / "average_pool", resamples to `target_size`.
    For "mean" / "max" / "min" / "sum", collapses the axis entirely.
    """
    n = arr.shape[axis]

    if mode == "subsample_uniform":
        if n == target_size:
            return arr
        idx = np.linspace(0, n - 1, num=target_size, dtype=int)
        return np.take(arr, idx, axis=axis)

    if mode == "average_pool":
        if n == target_size:
            return arr
        arr_moved = np.moveaxis(arr, axis, -1)
        edges = np.linspace(0, n, target_size + 1, dtype=int)
        pooled = np.stack(
            [np.mean(arr_moved[..., edges[i]:edges[i+1]], axis=-1)
             for i in range(target_size)],
            axis=-1,
        )
        return np.moveaxis(pooled, -1, axis)

    # Aggregation modes: collapse the axis entirely
    func = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}
    if mode in func:
        return func[mode](arr, axis=axis)

    raise ValueError(f"Unknown reduction mode: {mode}")

def _get_all_shapes_and_varying_axes(list_of_arrays):
    """
    Returns the shapes of all arrays and the axes that vary across them.

    Seems easy, but has to handle the case when some arrays have fewer dimensions than others, typically when the temporal dimension is not present in some arrays.
    Because the order of dimensions is not the same accross embeddings, we need to first find what dimensions are varying, 
    and then normalize all arrays to have the same number of dimensions by replacing missing dim by 1 dim at the correct position.
    """
    def _get_varying_axes(shapes, max_ndim):
        return [
            ax for ax in range(max_ndim)
            if len(np.unique([s[ax] for s in shapes])) > 1 # This dim has several values across the bars, hence it is changing.
        ]

    def _update_list_of_arrays(arrays_list, axes_to_add):
        new_list_of_arrays = copy.deepcopy(arrays_list)
        for i, arr in enumerate(arrays_list):
            if arr.ndim == min_dim: # finding the arrays with fewer dimensions
                for ax in sorted(axes_to_add):
                    arr = np.expand_dims(arr, axis=ax)
                new_list_of_arrays[i] = arr
            else: # Useless probably, but hey
                assert arr.ndim == max_ndim, "Seems to have a list of arrays with (at least) 3 different number of dimensions. This case was not anticipated, hence you need to resolve it (sorry)."
        return new_list_of_arrays

    max_ndim = max(arr.ndim for arr in list_of_arrays) # Get the maximal number of dim accross all arrays.

    # First, easy case.
    if all(arr.ndim == max_ndim for arr in list_of_arrays): # All arrays have the same number of dimensions. We can just compute shapes.
        all_shapes = [arr.shape for arr in list_of_arrays]
        # Find axes whose size differs across bars
        varying_axes = _get_varying_axes(all_shapes, max_ndim)

        return all_shapes, varying_axes, list_of_arrays

    # More complicated case: some arrays have fewer dimensions than others.
    else: # Some arrays have fewer dimensions than others. We need to normalize them.

        arrays_of_max_ndim = [arr for arr in list_of_arrays if arr.ndim == max_ndim] # Get the arrays that have the maximal number of dimensions.
        shapes_of_max_ndim = [a.shape for a in arrays_of_max_ndim] # Get their shapes.

        varying_axes = _get_varying_axes(shapes_of_max_ndim, max_ndim) # Get varying axes in the arrays of max dims

        min_dim = min(arr.ndim for arr in list_of_arrays)
        
        # Assert than all arrays have either the maximal or the minimal number of dimensions.
        # Otherwise, it should be handled differently.
        assert all(a.ndim in [max_ndim, min_dim] for a in list_of_arrays), "Seems to have a list of arrays with (at least) 3 different number of dimensions. This case was not anticipated, hence you need to resolve it (sorry)."

        n_missing = max_ndim - min_dim # Number of missing dimensions in the arrays of minimal ndim. Expected to be one, but developed to handle the case where it's not.

        # First, the case where there are as many missing dimensions as the varying ones.
        # In this case, the missing dimensions should exactly be the ones that are varying accross the arrays, i.e. (assuming) the temporal one.
        if len(varying_axes) == n_missing:
            # Insert size-1 dims at exactly the varying axes
            new_list_of_arrays = _update_list_of_arrays(list_of_arrays, varying_axes)

            assert all(arr.ndim == max_ndim for arr in new_list_of_arrays), "TO DEBUG: So my code (_get_all_shapes_and_varying_axes()) did not work as anticipated. Has to be fixed."

            all_shapes = [arr.shape for arr in new_list_of_arrays] # Let's recompute all shapes.
            # Varying axes should not have changed.
            return all_shapes, varying_axes, new_list_of_arrays

        else: # Well, some axis is not present in the embedding, while not varying in the other ones. A bit of a problem.
            # For now, let's just assume this case will never happen and raise an error if it does.
            # Ok; so it happens. Fuck.
            # Here is an example: all_shapes: [(3, 8, 64), (3, 8, 64), (3, 8, 64), (3, 8, 64), (3, 8, 64), (8, 64)]
            
            # First, let's find unmatching shaes given both shapes.
            def _get_unmatching_axes(shape_max_dims, shape_min_dims):
                # find the dims in max_shape that are missing in min_shape
                # This function allows to avoid for duplicates in max_shape.
                unmatching_axes = []
                remaining_min_shape = list(copy.deepcopy(shape_min_dims)) # copy of small shape to consume
                for ax_in_max in range(len(shape_max_dims)):
                    # If current dimension matches the next expected dimension from min_shape
                    if remaining_min_shape and shape_max_dims[ax_in_max] == remaining_min_shape[0]:
                        remaining_min_shape.pop(0) # it matches, so we "keep" it
                        continue # move to next axis in max_shape
                    
                    # Otherwise, this axis is missing from min_shape
                    unmatching_axes.append(ax_in_max)
                
                return unmatching_axes

            arrays_of_max_ndim = [arr for arr in list_of_arrays if arr.ndim == max_ndim] # Get the arrays that have the maximal number of dimensions.
            arrays_of_min_ndim = [arr for arr in list_of_arrays if arr.ndim == min_dim] # Get the arrays that have the minimal number of dimensions.
            assert all(arr.shape == arrays_of_min_ndim[0].shape for arr in arrays_of_min_ndim), "Check that all arrays of min dim have the same shape. If not, it means that the embeddings are not of the same shape, which is not anticipated."

            # Common shapes for template matching
            min_shape = arrays_of_min_ndim[0].shape
            max_shape = arrays_of_max_ndim[0].shape 
            
            # There might be a loophole if max_shapes are not all equal, and min_shape is equal to only one particular max_shape. This case should is handled in the assertion below.
            assert all(a_min_shape in max_shape for a_min_shape in min_shape), "All shapes in min_shape should be present in max_shape (otherwise, there is an axis in min_shape that is not present in max_shape, which is weird)."

            unmatching_axes = _get_unmatching_axes(max_shape, min_shape)
            
            if len(unmatching_axes) != n_missing: # Test if the number of unmatching axes is the same as the number of missing axes. Hopefully, yes.
                raise NotImplementedError(f"Cannot reliably determine missing axes. Max: {max_shape}, Min: {min_shape}, unmatching axes: {unmatching_axes}")

            # Normalize by inserting size-1 dims at the identified missing_axes
            new_list_of_arrays = _update_list_of_arrays(list_of_arrays, unmatching_axes)

            # Recompute shapes and update varying_axes to include these new ones
            all_shapes = [arr.shape for arr in new_list_of_arrays]
            varying_axes = _get_varying_axes(all_shapes, max_ndim)

            return all_shapes, varying_axes, new_list_of_arrays