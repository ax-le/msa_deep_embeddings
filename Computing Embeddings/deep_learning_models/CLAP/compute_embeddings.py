print("Starting computation of embeddings with CLAP...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch
from functools import partial

model_name = "CLAP"
dataset_name = "salami"
sr = 48000
time_reduction_method="mean"

def load_model(checkpoint_name="laion/clap-htsat-fused", cache_dir="/Brain/public/models/laion"):
    from transformers import ClapAudioModelWithProjection, ClapProcessor

    model = ClapAudioModelWithProjection.from_pretrained(f"/Brain/public/models/{checkpoint_name}", cache_dir=f"{cache_dir}/{checkpoint_name}")
    processor = ClapProcessor.from_pretrained(f"/Brain/public/models/{checkpoint_name}", cache_dir=f"{cache_dir}/{checkpoint_name}")

    model.eval()
    model.to("cuda")
    time_axis=0 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, processor, time_axis

def embed_this_signal(signal, model, processor, sr=48000):

    # Tokenize the audio for CLAP
    inputs_audio = processor(audio=signal, sampling_rate=sr, return_tensors="pt")

    # Cast the inputs for CPU or GPU inference        
    for key, value in inputs_audio.items():
        inputs_audio[key] = value.to("cuda")

    # Run the model 
    # with torch.inference_mode():
    outputs = model(**inputs_audio)
    audio_embedding = outputs.audio_embeds.detach()

    return audio_embedding


print("Computing embeddings...")
for checkpoint_name in ["laion/clap-htsat-fused", "laion/clap-htsat-unfused", "laion/larger_clap_general"]:
    print(f"Computing embeddings for {checkpoint_name}...")
    print("Loading model...")
    model, processor, time_axis = load_model(checkpoint_name)

    generic_compute.compute_barwise_embeddings(
        embed_fn = embed_this_signal,
        model_name = f"{model_name}/{checkpoint_name}",
        dataset_name = dataset_name,
        model = model,
        processor=processor,
        sr=sr,
        time_reduction_method=time_reduction_method,
        # save_suffix="",
        time_axis=time_axis,
        verbose=True,
    )