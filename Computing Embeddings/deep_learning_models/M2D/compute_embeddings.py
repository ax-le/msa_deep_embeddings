print("Starting computation of embeddings with m2d...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch
from functools import partial

model_name = "m2d"
dataset_name = "harmonix"
sr = 16000
time_reduction_method="mean"

def load_model(embed_type="audio", cache_dir="/Brain/public/models/m2d", checkpoint_name="m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"):
    from examples.portable_m2d import PortableM2D  # The portable_m2d is a simple one-file loader.
    if embed_type == "audio":
        model = PortableM2D(f'{cache_dir}/{checkpoint_name}')
    elif embed_type == "audio_text":
        model = PortableM2D(f'{cache_dir}/{checkpoint_name}', flat_features=True)
    else:
        raise ValueError(f"Unknown embed type: {embed_type}")
    model.eval()
    model.to("cuda")
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, None, time_axis

def embed_this_signal(signal, model, processor=None, embed_type="audio"):
    # model = model.to(device).eval()
    wavs = torch.tensor(signal).unsqueeze(0).to("cuda")
    with torch.no_grad():
        frame_level = model(wavs)
    if embed_type == "audio_text":
        frame_level = model.backbone.audio_proj(frame_level)  # to CLAP embeddings
    if frame_level.ndim > 2:
        frame_level = frame_level.squeeze(0)
    elif frame_level.ndim == 1:
        frame_level = frame_level.unsqueeze(0)
    return frame_level.detach()

print("Computing embeddings...")
for embed_type in ["audio_text", "audio"]:
    print(f"Computing embeddings for {embed_type}...")
    print("Loading model...")
    model, processor, time_axis = load_model(embed_type)

    embed_fn = partial(embed_this_signal, embed_type=embed_type)
    generic_compute.compute_barwise_embeddings(
        embed_fn = embed_fn,
        model_name = model_name,
        dataset_name = dataset_name,
        model = model,
        processor=processor,
        sr=sr,
        time_reduction_method=time_reduction_method,
        save_suffix=f"_space_{embed_type}",
        time_axis=time_axis,
        verbose=True,
    )