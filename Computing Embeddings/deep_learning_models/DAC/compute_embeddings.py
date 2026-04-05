print("Starting computation of embeddings with DAC...")
print("Imports...")
import embed_utils.compute_embeddings as generic_compute

model_name = "dac"
dataset_name = "salami"
time_reduction_method="mean"

def load_model(cache_path="/Brain/public/models", checkpoint="descript/dac_44khz"):
    from transformers import DacModel, AutoProcessor
    model = DacModel.from_pretrained(f"{cache_path}/{checkpoint}")
    processor = AutoProcessor.from_pretrained(f"{cache_path}/{checkpoint}")
    time_axis=1 # Without batch, because the batch dimension is squeezed in embed_this_signal
    return model, processor, time_axis

def embed_this_signal(signal, model, processor):
    inputs = processor(raw_audio=signal, sampling_rate=processor.sampling_rate, return_tensors="pt")
    encoder_outputs = model.encode(inputs["input_values"])
    # Get the intermediate audio codes
    return encoder_outputs.audio_codes.squeeze()

print("Loading model...")
model, processor, time_axis = load_model()
sr = processor.sampling_rate

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