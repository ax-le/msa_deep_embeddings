print("Starting computation of embeddings with matpac...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch

# model_name = "matpac"
dataset_name = "harmonix"
sr = 16000
time_reduction_method="mean"

def load_model(model_name="matpac", cache_dir="/Brain/public/models/matpac"):
    from matpac.model import get_matpac
    if model_name == "matpac":
        ckpt_path = f"{cache_dir}/matpac_10_2048.pt"
    elif model_name == "matpac_plus":
        ckpt_path = f"{cache_dir}/matpac_plus_music_6s_2048_enconly.pt" #matpac_plus_music_6s_2048_enconly.pt
    else:
        raise ValueError(f"Unknown model name {model_name}. Expecting matpac or matpac++")
    model = get_matpac(checkpoint_path=ckpt_path)

    model.eval()
    model.to("cuda")
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, None, time_axis

def embed_this_signal(signal, model, processor):
    if len(signal.shape) == 1:
        signal = signal[None,:] # Adding a channel dim, important for the model.
    signal = torch.Tensor(signal).cuda()
    emb, _ = model(signal)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return emb.detach()

print("Computing embeddings...")
for model_name in ["matpac", "matpac_plus"]:
    print(f"Computing embeddings for {model_name}...")
    print("Loading model...")
    model, processor, time_axis = load_model(model_name)

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