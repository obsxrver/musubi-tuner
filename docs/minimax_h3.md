# MiniMax H3

## Scope

Musubi Tuner supports experimental LoRA training of the MiniMax H3 **FL2VA** checkpoint in joint text-to-video-with-audio (T2VA) mode. Video and stereo audio are trained together. The separate Ref2VA checkpoint and reference-media conditioning are not supported yet, nor is sample generation during training.

H3 is exceptionally large. BF16 is currently required for the transformer; the ComfyUI INT8/ConvRot and NVFP4 checkpoints cannot be trained by this implementation. Gradient checkpointing and block swap are strongly recommended.

## Model files

Either model distribution can be used:

- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3): download the complete `FL2VA` directory. Pass the repository directory to every model argument; the scripts locate each component and all of its shards.
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3): download these individual files:
  - `diffusion_models/minimax_h3_fl2va_bf16.safetensors`
  - `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`
  - `vae/minimax_h3_video_vae_fp16.safetensors`
  - `vae/minimax_h3_audio_vae_fp32.safetensors`

For example, the official files can be downloaded with:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir models/MiniMax-H3
```

## Dataset

Use a video dataset. H3 treats video as 24 fps and accepts frame counts of `17*n+5`: `5, 22, 39, 56, 73, 90, 107, 124, ...`. Other requested lengths are rounded down. `5` frames is supported by the VAE, but longer clips are normally more useful for training.

Set `source_fps` to the actual frame rate of the source videos so video resampling and audio crops stay synchronized. Source audio is resampled to stereo 32 kHz. A video with no audio track gets a matching silent audio latent.

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
video_directory = "/path/to/videos"
cache_directory = "/path/to/cache"
target_frames = [56, 90, 124]
frame_extraction = "head"
source_fps = 30.0
```

Spatial bucket sizes are aligned to 32 pixels (video-VAE compression 16 multiplied by the transformer's spatial patch size 2).

## Pre-caching

Official repository layout:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config path/to/dataset.toml \
  --vae models/MiniMax-H3 \
  --audio_vae models/MiniMax-H3 \
  --vae_dtype bfloat16

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config path/to/dataset.toml \
  --text_encoder models/MiniMax-H3 \
  --tokenizer models/MiniMax-H3/FL2VA/tokenizer \
  --text_encoder_dtype bfloat16
```

For ComfyUI weights, give each corresponding `.safetensors` path instead. `--tokenizer` may point to the official `FL2VA/tokenizer` directory; if omitted, it is loaded from the official Hugging Face repository.

The text cache contains raw-prompt Qwen3-VL features immediately after layer 50, before the final RMSNorm. Chat templates and automatic special tokens are intentionally not applied, matching H3 FL2VA inference.

## Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dit models/MiniMax-H3 \
  --dataset_config path/to/dataset.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --discrete_flow_shift 12 \
  --weighting_scheme none --audio_loss_weight 1.0 \
  --gradient_checkpointing --blocks_to_swap 40 \
  --network_module networks.lora --network_dim 32 \
  --optimizer_type adamw8bit --learning_rate 1e-4 \
  --max_train_epochs 16 --save_every_n_epochs 1 \
  --output_dir path/to/output --output_name h3-lora
```

H3 uses a video sigma shift of 12 and an audio sigma shift of 3. The trainer samples the video schedule, maps the same base time to the audio schedule, noises both cached streams, and optimizes both raw `clean-noise` velocity targets. `--audio_loss_weight` controls the audio term relative to video.

Notes:

- Loader batches larger than one are accepted, but each packed H3 sequence is evaluated serially because sequence lengths are shape-dependent.
- `--blocks_to_swap` can be at most 48 for the 50-block transformer.
- `--fp8_base` and `--fp8_scaled` are rejected. Use the BF16 FL2VA checkpoint.
- Omit `--sample_prompts`; in-training H3 sampling is not implemented.
- Cached audio is part of every sample. Audio-free training is represented by silent source tracks rather than dropping the audio stream.

