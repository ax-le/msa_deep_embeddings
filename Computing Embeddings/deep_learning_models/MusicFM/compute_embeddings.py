print("Starting computation of embeddings with MusicFM...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch
import sys

model_name = "musicfm"
dataset_name = "harmonix"
sr = 24000
time_reduction_method="mean"

proj_path = "/Brain/private/a23marmo/projects/ssl_representation" # path where you cloned musicfm
sys.path.append(proj_path)

def load_model(cache_dir="/Brain/public/models/musicfm"):
    from musicfm.model.musicfm_25hz import MusicFM25Hz

    model = MusicFM25Hz(
        is_flash=False,
        stat_path=f"{cache_dir}/msd_stats.json",
        model_path=f"{cache_dir}/pretrained_msd.pt",
    )
    model = model.cuda()

    model.eval()
    model.to("cuda")
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, None, time_axis

def embed_this_signal(signal, model, processor):
    wav = torch.Tensor(signal).unsqueeze(0).cuda()
    emb = model.get_latent(wav, layer_ix=7)
    # # Sequence-level representation
    # seq_emb = emb.mean(-1) # (batch, time, channel) -> (batch, channel)
    return emb.squeeze(0).detach()

print("Computing embeddings...")
print("Loading model...")
model, processor, time_axis = load_model()

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