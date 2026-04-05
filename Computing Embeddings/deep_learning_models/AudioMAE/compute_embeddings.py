print("Starting computation of embeddings with AudioMAE...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
from einops import rearrange
import torch

model_name = "AudioMAE"
dataset_name = "harmonix"
sr = 16000
time_reduction_method="mean"

def load_model(cache_dir="/Brain/public/models", checkpoint_name="hance-ai/audiomae"):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(f"{cache_dir}/{checkpoint_name}", trust_remote_code=True).to("cuda")  # load the pretrained model
    model.eval()
    time_axis=2 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, None, time_axis

def embed_this_signal(audio, model, processor=None):
    # model.eval()
    audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio[None,:] # Adding a channel dim, important for the model.
    melspec = model.encoder.waveform_to_melspec(audio)  # (length, n_freq_bins) = (1024, 128)
    melspec = melspec[None,None,:,:]  # (1, 1, length, n_freq_bins) = (1, 1, 1024, 128)
    z = model.encoder.forward_features(melspec.to("cuda")).to("cpu")  # (b, 1+n, d); d=768
    z = z[:,1:,:]  # (b n d); remove [CLS], the class token

    b, c, w, h = melspec.shape  # w: temporal dim; h:freq dim
    wprime = round(w / model.encoder.patch_embed.patch_size[0])  # width in the latent space
    hprime = round(h / model.encoder.patch_embed.patch_size[1])  # height in the latent space

    # reconstruct the temporal and freq dims
    z = rearrange(z, 'b (w h) d -> b d h w', h=hprime)  # (b d h' w')

    # remove the batch dim
    z = z[0]  # (d h' w')
    return z.detach()  # (d h' w')

print("Loading model...")
model, processor, time_axis = load_model()

print("Computing embeddings...")
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
)