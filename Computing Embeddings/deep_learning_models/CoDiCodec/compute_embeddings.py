print("Starting computation of embeddings with codicodec...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute

model_name = "codicodec"
dataset_name = "harmonix"
sr = 44100
time_reduction_method="mean"

def load_model():
    from codicodec import EncoderDecoder
    codicodec_encdec = EncoderDecoder()
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return codicodec_encdec, None, time_axis

def embed_this_signal(signal, model, processor=None, discrete=False):
    latent = model.encode(signal, discrete=discrete)
    latent = latent.squeeze(0)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    return latent

print("Loading model...")
model, processor, time_axis = load_model()

print("Computing embeddings...")
for discrete in [False, True]:
    print(f"Discrete: {discrete}")
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
        discrete=discrete,
    )