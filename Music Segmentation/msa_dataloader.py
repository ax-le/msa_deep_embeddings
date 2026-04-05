"""
This module contains the dataloaders for the main datasets in MSA: SALAMI [1], RWCPOP [2] and Beatles [3] datasets.

It loads the data, but also computes the barwise TF matrix and the bars from the audio files.
When the barwise TF matrix and bars are computed, they are saved in a cache folder (if provided) to avoid recomputing them.

References
----------
[1] Smith, J. B. L., Burgoyne, J. A., Fujinaga, I., De Roure, D., & Downie, J. S. (2011). Design and creation of a large-scale database of structural annotations. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 555-560). Miami, FL.

|2] Goto, M., Hashiguchi, H., Nishimura, T., & Oka, R. (2002). RWC Music Database: Popular, classical and jazz music databases. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 287-288).

[3] Harte, C. (2010). Towards automatic extraction of harmony information from music signals (Doctoral dissertation). Queen Mary University of London.
http://isophonics.net/content/reference-annotations-beatles
"""
import librosa
import mirdata
import mirdata.download_utils
import pathlib
import shutil
import numpy as np
import os
import warnings
from beat_this.inference import File2Beats

# import as_seg
import as_seg.data_manipulation as dm
import as_seg.model.errors as err

import base_audio.signal_to_spectrogram as signal_to_spectrogram
import trimming_utils

BASE_MODEL_PATH = "/Brain/public/models"
DEFAULT_BEAT_THIS_MODEL_CKPT = f"{BASE_MODEL_PATH}/beat_this/beat_this-final0.ckpt"

eps = 1e-10

class MSABaseDataloader():
    def __init__(self, cache_path = None, sr=44100, verbose = False):
        """
        Constructor of the MSABaseDataloader class.

        Parameters
        ----------
        feature : string
            The feature to compute the spectrogram. Must be a valid feature name.
        cache_path : string
            The path where to save the computed barwise TF matrices and bars. If None, the cache is not used.
            The default is None.
        sr : int
            The sampling rate of the audio files.
            The default is None, meaning that it will keep the original sampling rate of the audio file.
        n_fft : int
            The number of samples in each STFT window.
        hop_length : int
            The hop length of the spectrogram.
            The default is 512.
        subdivision : int
            The number of subdivisions of a bar.
            The default is 96.
        verbose : bool
            If True, print some information about the cache.
            The default is False
        """
        self.cache_path = cache_path
        self.verbose = verbose
        self.sr=sr
        self.indexes = NotImplementedError("Needs to be defined in child classes") # Needs to be redefined in child classes.

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        raise NotImplementedError("This method should be implemented in the child class") from None

    def get_item_of_id(self, audio_id):
        """
        Returns the item of the given id. Requires self.indexes to be set.
        
        Parameters
        ----------
        audio_id : str
            Id of the signal in the dataset.

        Returns
        -------
        Whatever is returned in the getter of the current class.
        """
        # index = self.indexes.index(audio_id)
        try:
            index = self.indexes.index(audio_id)
        except ValueError:
            try:
                index = self.indexes.index(str(audio_id))
            except ValueError:
                raise ValueError(f"Audio {audio_id} not found in the dataset") from None

        return self.__getitem__(index)

    def __len__(self):
        """
        Return the number of tracks in the dataset.

        By default, returns the number of elements in the indexes.
        """
        return len(self.indexes)

    def get_spectrogram(self, signal, feature="log_mel", hop_length = 512): # The spectrogram is not saved in the cache because it is too large in general
        """
        Returns the spectrogram, from the signal of a song.
        """
        feature_object = signal_to_spectrogram.FeatureObject(sr=self.sr, feature=feature, hop_length=hop_length, mel_grill = True)
        return feature_object.get_spectrogram(signal)

    def get_bars(self, audio_path, index = None, checkpoint_path = DEFAULT_BEAT_THIS_MODEL_CKPT):
        """
        Return the bars of the song.
        They are computed from the audio file.
        If the cache is used, the bars are saved in the cache.
        An identifier of the song should be provided to save the bars in the cache.
        """
        def _compute_bars(): # Define the function to compute the bars
            file2beats = File2Beats(checkpoint_path=checkpoint_path, device="cuda", dbn=False)
            beats, downbeats = file2beats(audio_path)
            assert len(beats) > 0, f"No beats found for {audio_path}"
            assert len(downbeats) > 0, f"No downbeats found for {audio_path}"
            try:
                last_beat = beats[-1]
                if last_beat not in downbeats:
                    downbeats = np.append(downbeats, last_beat) # Adding the as end of the song, if not already in the downbeats list. Not sure of this one, but let's see. Will probably miss the final moment of the song (like the end of the final beat), but better than nothing I guess. Should not be taken into account with trimmed evaluaton anyway.
            except IndexError:
                warnings.warn(f"No beats found for {audio_path}. This is weird actually. Check that downbeats is not empty either.")
                assert len(downbeats) > 1, f"No downbeats found for {audio_path}."
            return dm.frontiers_to_segments(downbeats)

        # If a cache is set
        if self.cache_path is not None:

            # No identifier is provided for this song, hence it cannot be saved in the cache
            if index is None:
                warnings.warn("No index provided for the cache, the cache will be ignored")
            
            # An identifier is provided
            else:
                dir_save_bars_path = f"{self.cache_path}/bars"
                
                # Tries to load the bars from the cache
                try:
                    bars = np.load(f"{dir_save_bars_path}/{index}.npy", allow_pickle=True)
                    if self.verbose:
                        print("Using cached bars.")
                
                # If the file is not found, the bars are computed and saved in the cache
                except FileNotFoundError:
                    bars = _compute_bars() # Compute the bars

                    # Save the bars in the cache
                    pathlib.Path(dir_save_bars_path).mkdir(parents=True, exist_ok=True)
                    np.save(f"{dir_save_bars_path}/{index}.npy", bars)
                
                # Return the bars
                return bars
        # No cache is set, the bars are computed and returned
        return _compute_bars()

    def save_segments(self, segments, name):
        """
        Save the segments of a song in the original folder.
        Important for reproducibility.
        """
        # mirdata_segments = mirdata.annotations.SectionData(intervals=segments, interval_unit="s")
        # jams_segments = mirdata.jams_utils.sections_to_jams(mirdata_segments)
        dir_save_path = f"{self.data_path}/estimations/segments/{self.name.lower()}"
        pathlib.Path(dir_save_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{dir_save_path}/{name}.npy", segments)

    def score_flat_segmentation(self, segments, annotations, trim=False):
        """
        Compute the score of a flat segmentation.
        """
        close_tolerance = dm.compute_score_of_segmentation(annotations, segments, window_length=0.5, trim=trim)
        large_tolerance = dm.compute_score_of_segmentation(annotations, segments, window_length=3, trim=trim)
        return close_tolerance, large_tolerance

class RWCPopDataloader(MSABaseDataloader):
    """
    Dataloader for the RWC Pop dataset.
    """

    name = "rwcpop"

    def __init__(self, datapath, cache_path = None, download=False, sr=44100, verbose = False):
        """
        Constructor of the RWCPopDataloader class.

        Parameters
        ----------
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False.
        """
        if self.name is None: # Should never happen
            raise ValueError("Name of the dataset is not set.")

        # If a cache path is provided, the cache path is set to the cache path of the dataset
        new_cache_path = cache_path
        if cache_path is not None and self.name not in cache_path:
            new_cache_path = f"{cache_path}/{self.name}" # Add the dataset name if not already in the cache path
        
        # Initialize the base dataloader
        super().__init__(cache_path=new_cache_path, sr=sr, verbose=verbose) # feature is not used here
        
        # Load the dataset
        self.datapath = datapath
        rwcpop = mirdata.initialize('rwc_popular', data_home = datapath)
        if download:
            # Adding the MIREX10_SECTIONS annotations
            mirdata.datasets.rwc_popular.REMOTES["MIREX10_SECTIONS"]=mirdata.download_utils.RemoteFileMetadata(
                    filename="MIREX10_SECTIONS.zip",
                    url="https://github.com/ax-le/mirex10_sections/archive/main.zip",
                    checksum="85f71a8cf3dda4438366b55364d29c59",
                    destination_dir="annotations")

            rwcpop.download()
            
        self.all_tracks = rwcpop.load_tracks()

        # Modifying the audio paths: now, they are named with the indexes of the songs
        # Instead of having several folders indexing from 1 to 16 containing the files.
        # This goes against the mirdata standards, but it is more convenient for me.
        warnings.warn("Paths to RWC-Pop audio files are modified, and this is hardcoded. This modification assumes that all audio files of RWC-Pop are stored in one unique folder instead of several.")
        for a_track in self.all_tracks.values():
            a_track.audio_path = f"{datapath}/audio/{a_track.track_id}.mp3"

        self.indexes = rwcpop.track_ids

        # self.dataset_name = "rwcpop"

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        # Compute audio
        sig, _ = librosa.load(track.audio_path, sr=self.sr, mono=True)

        len_signal = len(sig)/self.sr
        annotations_intervals, labels = self.get_annotations(track, len_signal=len_signal, annot_type="MIREX10")

        # Return the the bars, the barwise TF matrix and the annotations
        return sig, track, annotations_intervals, labels, len_signal

    def get_annotations(self, track, len_signal, annot_type = "MIREX10"):
        """
        Returns the raw annotations and labels of the track (no trimming applied).
        """
        assert len_signal is not None, "len_signal must be provided for MIREX10 annotations."

        match annot_type:
            case "AIST": #  Was never tested, to be careful.
                print("DEBUG: Using AIST annotations, but they were never tested. To be careful.")
                annotations_intervals = track.sections.intervals
                labels = track.sections.labels

            case "MIREX10":
                track_id = track.track_id
                annot_path_mirex = f"{self.datapath}/annotations/mirex10_sections-main/{track_id}.BLOCKS.lab"
                all_annot, labels = self.get_rwcpop_annotated_segments_from_txt(annot_path_mirex, "MIREX10", return_labels=True)
                all_annot = np.array(all_annot)
                annotations_intervals = all_annot[:,0:2] # Keep only the intervals

            case _:
                raise err.InvalidArgumentValueException("Invalid annotation type. Must be either 'MIREX10' or 'AIST'.")

        assert len(annotations_intervals) == len(labels), f"Number of segments ({len(annotations_intervals)}) is different from number of labels ({len(labels)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary."

        return np.asarray(annotations_intervals, dtype=float), list(labels)

    def get_rwcpop_annotated_segments_from_txt(self, path, annotations_type, return_labels=False):
        """
        Reads the segmentation annotations, and returns it in a list of tuples (start, end, index as a number)
        This function has been developped for AIST and MIREX10 annotations, adapted for these types of annotations.
        It will not work with another set of annotation.

        Parameters
        ----------
        path : String
            The path to the annotation.
        annotations_type : "AIST" [1] or "MIREX10" [2]
            The type of annotations to load (both have a specific behavior and formatting)
            
        Raises
        ------
        err.InvalidArgumentValueException
            If the type of annotations is neither AIST or MIREX10

        Returns
        -------
        segments : list of tuples (float, float, integer)
            The segmentation, formatted in a list of tuples, and with labels as numbers (easier to interpret computationnally).

        References
        ----------
        [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
        
        [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
        Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

        """
        file_seg = open(path)
        segments = []
        unique_labels = []
        all_labels = []
        
        for part in file_seg.readlines():
            tupl = part.split("\t")
            all_labels.append(tupl[2])
            if tupl[2] not in unique_labels: # If label wasn't already found in this annotation
                idx = len(unique_labels)
                unique_labels.append(tupl[2])
            else: # If this label was found for another segment
                idx = unique_labels.index(tupl[2])
            if annotations_type == "AIST":
                segments.append(((int(tupl[0]) / 100), (int(tupl[1]) / 100), idx))
            elif annotations_type == "MIREX10":
                segments.append((round(float(tupl[0]), 3), round(float(tupl[1]), 3), idx))
            else:
                raise err.InvalidArgumentValueException("Annotations type not understood")
        if return_labels:
            return segments, all_labels
        return segments
    
    def format_dataset_from_mirdata_standards(self, file_extension = "mp3"):
        """      
        I found very confusing the way mirdata handles the paths, because songs are not named with their indexes.
        So I changed them to a unique list of files, ranging from 1 to 100.
        You can follow mirdata standards if you want, but you will have to modify the dataloader accordingly (in particular the datapaths in the tracks).
        Also, the cache may not work, please be careful.
        If you want to follow my version but have the data as mirdata, you can use the following function to copy the audio files to the right location.
        CAREFUL: It is not extensively tested though, so don't delete the original files.

        Parameters
        ----------
        file_extension : string
            The extension of the audio files.
            Default is "mp3".
        """
        def _filename_as_RWC(val):
            if type(val) == int:
                val = str(val)
            return "RM-P" + val.zfill(3)
        
        offset_previous_folders = 0
        for folder in sorted(os.listdir(f"{self.datapath}/audio")):
            try:
                count_files = 0
                for file in sorted(os.listdir(f"{self.datapath}/audio/{folder}")):

                    if file_extension not in file:
                        continue
                    new_number_file = int(file[:2]) + offset_previous_folders
                    src = f"{self.datapath}/audio/{folder}/{file}"
                    dest = f"{self.datapath}/audio/{_filename_as_RWC(new_number_file)}.{file_extension}"

                    if self.verbose:
                        print(f"Copying {src} to {dest}")
                    shutil.copy(src, dest)

                    count_files += 1

                offset_previous_folders += count_files
            
            except NotADirectoryError:
                pass

class SALAMIDataloader(MSABaseDataloader):
    """
    Dataloader for the SALAMI dataset.
    """

    name = "salami"

    def __init__(self, datapath, cache_path = None, download=False, subset = None, sr=44100, verbose = False, annotation_level="upper", annotator=1):
        """
        Constructor of the SALAMIDataloader class.

        Parameters
        ----------  
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False.
        subset : string
            The subset of the dataset to use. Can be "train", "test" or "debug".
        """
        if self.name is None:
            raise ValueError("Name of the dataset is not set.")
        new_cache_path = cache_path
        if cache_path is not None and self.name not in cache_path:
            new_cache_path = f"{cache_path}/{self.name}" 
        super().__init__(cache_path=new_cache_path, sr=sr, verbose=verbose) # feature is not used here
        
        # self.dataset_name = "salami"

        self.annotation_level = annotation_level
        self.annotator = annotator

        self.datapath = datapath
        salami = mirdata.initialize('salami', data_home = datapath)
        if download:
            salami.download()            
        self.all_tracks = salami.load_tracks()
        self.indexes = salami.track_ids

        self.subset = subset
        if subset is not None:
            train_indexes, test_indexes = self.split_training_test()
            if subset == "train":
                self.indexes = train_indexes
            elif subset == "test":
                self.indexes = test_indexes
            elif subset == "debug":
                self.indexes = test_indexes[:4]
            else:
                raise ValueError("Subset should be either 'train' or 'test'")

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        # Parsing through files ordered with self.indexes
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        try:
            # Compute audio
            sig, _ = librosa.load(track.audio_path, sr=self.sr, mono=True)

            # Get the annotations
            len_signal = len(sig)/self.sr
            dict_annotations = self.get_annotations(track, len_signal=len_signal)
            annotations_intervals, labels = self.get_this_set_annotations(
                dict_annotations,
                annotation_level=self.annotation_level,
                annotator=self.annotator
            )

            # Return the signal, track, annotations, labels, and signal length
            return sig, track, annotations_intervals, labels, len_signal
    
        except FileNotFoundError: # Handling file not found errors without throwing errors. Must be catched later on.
            warnings.warn(f'{track_id} not found.')
            return None, None, None, None, None
            # raise FileNotFoundError(f"Song {track_id} not found, normal ?") from None

    def split_training_test(self):
        """
        Split the dataset in training and test set.
        The split is made as: 
        - All tracks with two sets of annotations are in the test set
        - All tracks with one set of annotations are in the training set
        """
        indexes_train = []
        indexes_test = []
        for track_id in self.indexes:
            track = self.all_tracks[track_id]
            try:
                track.sections_annotator_1_uppercase.intervals
                track.sections_annotator_2_uppercase.intervals
                indexes_test.append(track_id)
            except AttributeError:
                indexes_train.append(track_id)
        return indexes_train, indexes_test
            
    def get_annotations(self, track, len_signal):
        """
        Return the raw annotations of the track (no trimming), in the form of a dict.
        It returns the annotations of the first annotator, and if available, the annotations of the second annotator.
        It returns both levels of annotations (upper and lower) for each annotator.
        Labels are stored alongside intervals so trimming can be applied later.
        """
        dict_annotations = {}
        try: 
            # Trying to get the first annotator
            dict_annotations["upper_level_annotations"] = np.array(track.sections_annotator_1_uppercase.intervals)
            dict_annotations["upper_level_labels"] = list(track.sections_annotator_1_uppercase.labels)
            dict_annotations["lower_level_annotations"] = np.array(track.sections_annotator_1_lowercase.intervals)
            dict_annotations["lower_level_labels"] = list(track.sections_annotator_1_lowercase.labels)
            try: # Trying to load the second annotator
                dict_annotations["upper_level_annotations_2"] = np.array(track.sections_annotator_2_uppercase.intervals)
                dict_annotations["upper_level_labels_2"] = list(track.sections_annotator_2_uppercase.labels)
                dict_annotations["lower_level_annotations_2"] = np.array(track.sections_annotator_2_lowercase.intervals)
                dict_annotations["lower_level_labels_2"] = list(track.sections_annotator_2_lowercase.labels)
                dict_annotations["annot_number"]  = 2
            except AttributeError: # Only the first annotator was loaded
                dict_annotations["annot_number"]  = 1
        except AttributeError:
            try:
                # Trying to get the second annotator (no first one)
                dict_annotations["upper_level_annotations"] = np.array(track.sections_annotator_2_uppercase.intervals)
                dict_annotations["upper_level_labels"] = list(track.sections_annotator_2_uppercase.labels)
                dict_annotations["lower_level_annotations"] = np.array(track.sections_annotator_2_lowercase.intervals)
                dict_annotations["lower_level_labels"] = list(track.sections_annotator_2_lowercase.labels)
                dict_annotations["annot_number"]  = 1
            except AttributeError:
                raise AttributeError(f"No annotations found for {track.track_id}")
        
        return dict_annotations
    
    def get_this_set_annotations(self, dict_annotations, annotation_level = "upper", annotator = 1):
        """
        Return a particular set of annotations and labels from all the annotations.
        1 will always return the annotator if there is only one, or the first annotator if there are two.
        2 will always return the second annotator if there are two.
        'both' for `annotator` or `annotation_level` returns a list of arrays.
        """
        levels = ["upper", "lower"] if annotation_level == "both" else [annotation_level]
        annotators = [1, 2] if annotator == "both" else [annotator]

        list_annots = []
        list_labels = []

        for ann in annotators:
            if ann > dict_annotations.get("annot_number", 1):
                if annotator == "both":
                    raise ValueError("Invalid annotation level")
                    # continue  # Just ignore if second one doesn't exist when 'both' is requested
                else:
                    raise ValueError(f"Annotator {ann} not found.")

            for lvl in levels:
                if ann == 1:
                    a = dict_annotations[f"{lvl}_level_annotations"]
                    l = dict_annotations[f"{lvl}_level_labels"]
                    assert len(a) == len(l), f"Number of segments ({len(a)}) is different from number of labels ({len(l)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary."

                elif ann == 2:
                    a = dict_annotations[f"{lvl}_level_annotations_2"]
                    l = dict_annotations[f"{lvl}_level_labels_2"]
                    assert len(a) == len(l), f"Number of segments ({len(a)}) is different from number of labels ({len(l)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary."
                else:
                    raise ValueError("Invalid annotator number")
                
                list_annots.append(a)
                list_labels.append(l)

        if len(list_annots) == 1:
            return list_annots[0], list_labels[0]
        return list_annots, list_labels
    
class BeatlesDataloader(MSABaseDataloader):
    """
    Dataloader for the Beatles dataset.
    """
    name = "beatles"

    def __init__(self, datapath, cache_path = None, download=False, sr=44100, verbose = False): # feature, 
        """
        Constructor of the BeatlesDataloader class.

        Parameters
        ----------
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False
        """
        if self.name is None: # Should never happen
            raise ValueError("Name of the dataset is not set.")

        # If a cache path is provided, the cache path is set to the cache path of the dataset
        new_cache_path = cache_path
        if cache_path is not None and self.name not in cache_path:
            new_cache_path = f"{cache_path}/{self.name}" # Add the dataset name if not already in the cache path
        
        # Initialize the base dataloader
        super().__init__(cache_path=new_cache_path, sr=sr, verbose=verbose) # feature is not used here
        
        self.datapath = datapath
        beatles = mirdata.initialize('beatles', data_home = datapath)
        if download:
            beatles.download()            
        self.all_tracks = beatles.load_tracks()
        self.indexes = beatles.track_ids

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        # Compute the audio
        sig, _ = librosa.load(track.audio_path, sr=self.sr, mono=True)

        # Get the annotations
        len_signal = len(sig)/self.sr
        annotations_intervals, labels = self.get_annotations(track, len_signal=len_signal)

        # Return signal, track_id, annotations, labels, and signal length
        return sig, track, annotations_intervals, labels, len_signal

    def get_annotations(self, track, len_signal):
        """
        Returns the raw annotations and labels of the track (no trimming applied).
        """
        intervals = track.sections.intervals
        labels = track.sections.labels
        assert len(intervals) == len(labels), f"Number of segments ({len(intervals)}) is different from number of labels ({len(labels)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary."

        return np.asarray(intervals, dtype=float), list(labels)
    
    # def __len__(self):
    #     """
    #     Return the number of tracks in the dataset.
    #     """
    #     return len(self.all_tracks) # Why not just len(indexes, an in the base class?)

class HarmonixTrack: # Creating a special Track object for Hamronix, because it is not supported by mirdata
    def __init__(self, track_id, audio_path):
        self.track_id = track_id
        self.audio_path = audio_path

class HarmonixDataloader(MSABaseDataloader):
    """
    Dataloader for the Harmonix dataset.
    """
    name = "harmonix"

    def __init__(self, datapath, cache_path = None, download=False, sr=44100, verbose = False): # feature, 
        """
        Constructor of the HarmonixDataloader class.

        Parameters
        ----------
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False
        """
        if self.name is None: # Should never happen
            raise ValueError("Name of the dataset is not set.")

        # If a cache path is provided, the cache path is set to the cache path of the dataset
        new_cache_path = cache_path
        if cache_path is not None and self.name not in cache_path:
            new_cache_path = f"{cache_path}/{self.name}" # Add the dataset name if not already in the cache path
        
        # Initialize the base dataloader
        super().__init__(cache_path=new_cache_path, sr=sr, verbose=verbose) # feature is not used here
        
        self.datapath = datapath
        # Harmonix is not on mirdata
        # harmonix = mirdata.initialize('harmonix', data_home = datapath)
        raw_audio_files_list = os.listdir(f"{self.datapath}/audio")
        raw_audio_files_list = [f for f in raw_audio_files_list if f.endswith(".mp3")]
        self.all_tracks = {}
        self.indexes = []
        for path in raw_audio_files_list:
            track_id = path.split(".mp3")[0]
            self.all_tracks[track_id] = HarmonixTrack(track_id, f"{self.datapath}/audio/{path}")
            self.indexes.append(track_id)

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        # Compute the audio
        sig, _ = librosa.load(track.audio_path, sr=self.sr, mono=True)

        # Get the annotations
        len_signal = len(sig)/self.sr
        annotations_intervals, labels = self.get_annotations(track, len_signal=len_signal)

        return sig, track, annotations_intervals, labels, len_signal

    def get_annotations(self, track, len_signal):
        """
        Returns the raw annotations and labels of the track.
        For Harmonix, silences at the edges (leading/trailing segments with "silence" or "end" labels)
        are automatically trimmed to avoid issues with boundary-based evaluation.
        """
        annot_path = f"{self.datapath}/annotations/harmonix_segments/{track.track_id}.txt"
        all_annot, labels = self.get_harmonix_annotated_segments_from_txt(annot_path, return_labels=True)
        if len(labels) > len(all_annot):
            if len(labels) - len(all_annot) == 1:
                labels = labels[:-1] # Just remove the last label if there is one more label than segments, because, in Harmonix, annotations are given as boundaries, thus leading to one extra label (the end one).
            else:
                raise ValueError(f"Number of labels ({len(labels)}) is greater than number of segments ({len(all_annot)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary.")
        
        assert len(all_annot) == len(labels), f"Number of segments ({len(all_annot)}) is different from number of labels ({len(labels)}) for track {track.track_id}. This is probably a mistake in the annotation file, but it is not clear how to handle it. Please check the annotation file and fix it if necessary."

        # Trim leading/trailing segments marked as silent (labels "silence", "#", "end", etc.)
        # This is done at load time for Harmonix, so my_trim logic is not applied to Harmonix
        all_annot, labels = trimming_utils.trim_silent_segments(np.asarray(all_annot, dtype=float), list(labels), len_signal=len_signal)

        return np.asarray(all_annot, dtype=float), list(labels)

    def get_harmonix_annotated_segments_from_txt(self, path, return_labels=False):
        file_seg = open(path)
        boundaries = []
        all_labels = []
        
        for part in file_seg.readlines():
            tupl = part.split("\t")
            boundaries.append(round(float(tupl[0]), 3))
            all_labels.append(tupl[1])

        segments = dm.frontiers_to_segments(boundaries)
        segments = np.array(segments)

        if return_labels:
            return segments, all_labels
        return segments
