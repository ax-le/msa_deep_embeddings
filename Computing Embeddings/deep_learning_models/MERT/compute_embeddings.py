print("Starting computation of embeddings with CLAP...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute
import torch
from functools import partial

model_name = "MERT"
dataset_name = "salami"
sr = 48000
time_reduction_method="mean"

def load_model(checkpoint_name="m-a-p/MERT-v1-330M", cache_dir="/Brain/public/models/laion"):
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import AutoModel

    model = AutoModel.from_pretrained(f"/Brain/public/models/{checkpoint_name}", trust_remote_code=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"/Brain/public/models/{checkpoint_name}", trust_remote_code=True)
    
    model.eval()
    model.to("cuda")
    time_axis=1 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, feature_extractor, time_axis

def embed_this_signal(signal, model, processor, sr):
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt")
    for key, val in inputs.items():
        inputs[key] = val.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # take a look at the output shape, there are 25 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    return all_layer_hidden_states.detach()

print("Computing embeddings...")
for checkpoint_name in ["m-a-p/MERT-v1-95M"]:
    print(f"Computing embeddings for {checkpoint_name}...")
    print("Loading model...")
    model, processor, time_axis = load_model(checkpoint_name)
    sr = processor.sampling_rate
    embed_fn = partial(embed_this_signal, sr=sr)

    generic_compute.compute_barwise_embeddings(
        embed_fn = embed_fn,
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