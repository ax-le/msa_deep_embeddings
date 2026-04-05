print("Starting computation of embeddings with MuQ...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch
from functools import partial

model_name = "OpenMuQ"
dataset_name = "harmonix"
sr = 24000
time_reduction_method="mean"

def load_model(embed_type="audio", cache_dir="/Brain/public/models", model_name = "OpenMuQ"):
    from muq import MuQ, MuQMuLan
    if embed_type == "audio":
        checkpoint_name = "MuQ-large-msd-iter"
        model = MuQ.from_pretrained(f"/Brain/public/models/{model_name}/{checkpoint_name}", cache_dir=f"{cache_dir}/{model_name}/{checkpoint_name}")
    elif embed_type == "audio_text":
        checkpoint_name = "MuQ-MuLan-large"

        # Should be set as the model name and not a path,
        # due to a lack of model.safetensors 
        # (ugly trick, but works, because hf will try to load a model.bin if
        # model_name is the name of the HF model and not a dir). 
        # (Also following env in https://github.com/tencent-ailab/MuQ/issues/13)
        model = MuQMuLan.from_pretrained(f"{model_name}/{checkpoint_name}", cache_dir=f"{cache_dir}/{model_name}/{checkpoint_name}")
    else:
        raise ValueError(f"Unknown embed type: {embed_type}")
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    model.eval()
    model.to("cuda")
    return model, None, time_axis

def embed_this_signal(signal, model, processor, embed_type="audio"):
    wavs = torch.tensor(signal).unsqueeze(0).to("cuda") 
    with torch.no_grad():
        if embed_type == "audio_text":
            # MuQ-MuLaN
            audio_embeds = model(wavs = wavs)
        elif embed_type == "audio":
            # MuQ
            output = model(wavs, output_hidden_states=False)
            audio_embeds=output.last_hidden_state
            audio_embeds = audio_embeds.squeeze(0)
            # audio_embeds = audio_embeds.mean(axis=1)
        else:
            raise ValueError(f"Unknown embed type: {embed_type}")

    return audio_embeds.detach()

if __name__ == "__main__":
    print("Computing embeddings...")
    for embed_type in ["audio", "audio_text"]:
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