print("Starting computation of embeddings with music2latent...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import numpy as np

model_name = "music2latent"
dataset_name = "harmonix"
time_reduction_method="mean"

def load_model():
    from music2latent import EncoderDecoder
    encdec = EncoderDecoder()
    time_axis=1 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return encdec, None, time_axis

def embed_this_signal(signal, model, processor=None, extract_features=False):
    try:
        latent = model.encode(signal, extract_features=extract_features)
    except RuntimeError: # Signal is too short, needs to be padded
        signal = np.concatenate([signal, np.zeros(34304 - len(signal))]) # Music2latent was trained on samples of size 34304.
        latent = model.encode(signal, extract_features=extract_features)
    return latent.squeeze(0)

print("Loading model...")
model, processor, time_axis = load_model()
sr = 44100

print("Computing embeddings...")
for extract_features in [False, True]:
    generic_compute.compute_barwise_embeddings(
        embed_fn = embed_this_signal,
        model_name = model_name,
        dataset_name = dataset_name,
        model = model,
        processor=processor,
        sr=sr,
        time_reduction_method=time_reduction_method,
        # save_suffix="",
        time_axis=time_axis,
        verbose=True,
        extract_features=extract_features,
    )