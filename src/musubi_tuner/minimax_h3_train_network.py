from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3 import minimax_h3_utils
from musubi_tuner.minimax_h3.model import KEYFRAME_CLEAN, MiniMaxH3Model, time_shift_sigma
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MiniMax H3 does not support the FP8 flags. Pass a Comfy INT8 ConvRot checkpoint to --dit directly.")
        if args.mixed_precision not in (None, "bf16"):
            raise ValueError("MiniMax H3 training requires --mixed_precision bf16")
        args.mixed_precision = "bf16"
        if args.video_flow_shift <= 0 or args.audio_flow_shift <= 0:
            raise ValueError("MiniMax H3 flow shifts must be positive")
        if args.audio_loss_weight < 0:
            raise ValueError("--audio_loss_weight must be non-negative")
        self.dit_dtype = torch.bfloat16
        args.dit_dtype = "bfloat16"
        self._i2v_training = args.i2v
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = args.video_flow_shift
        self.vae_frame_stride = 17

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        raise ValueError("In-training sample generation is not implemented for MiniMax H3; omit --sample_prompts")

    def do_inference(self, *args, **kwargs):
        raise NotImplementedError("MiniMax H3 sampling is not part of the training integration")

    def load_vae(self, args, vae_dtype, vae_path):
        raise NotImplementedError("MiniMax H3 VAE decoding is not needed for training without --sample_prompts")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        if dit_weight_dtype not in (None, torch.bfloat16):
            raise ValueError(f"MiniMax H3 compute must use BF16, got {dit_weight_dtype}")
        model = minimax_h3_utils.load_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        model.sigma_shift_video = args.video_flow_shift
        model.sigma_shift_audio = args.audio_flow_shift
        return model

    def compile_transformer(self, args, transformer):
        model: MiniMaxH3Model = transformer
        return model_utils.compile_transformer(
            args, model, [model.token_refiner.blocks, model.blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # Both H3 VAEs store already-normalized latents in the cache.
        return latents

    def call_dit(
        self,
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
        **kwargs,
    ):
        raise RuntimeError("MiniMax H3 uses its joint audio/video process_batch implementation")

    @staticmethod
    def _weighted_mse(pred, target, sigma, weighting_scheme):
        loss = F.mse_loss(pred, target, reduction="none")
        if weighting_scheme == "sigma_sqrt":
            weight = sigma.float().clamp_min(1e-6).pow(-2)
        elif weighting_scheme == "cosmap":
            weight = 2 / (torch.pi * (1 - 2 * sigma.float() + 2 * sigma.float().square()))
        else:
            weight = None
        if weight is not None:
            loss = loss * weight.view(-1, *([1] * (loss.ndim - 1)))
        return loss.mean()

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        if "latents_audio" not in batch or "h3_text_embed" not in batch:
            raise ValueError("H3 audio and text caches are missing; run both minimax_h3 cache commands first")
        if self.i2v_training and "latents_image" not in batch:
            raise ValueError("H3 I2V reference latents are missing; rerun minimax_h3_cache_latents.py with --i2v")
        sigma_video, _timesteps = self.get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            batch["timesteps"],
            noise_scheduler,
            accelerator.device,
            dit_dtype,
            return_sigmas=True,
        )

        clean_video = latents.to(accelerator.device, dtype=network_dtype)
        noise_video = noise.to(accelerator.device, dtype=network_dtype)
        clean_audio = batch["latents_audio"].to(accelerator.device, dtype=network_dtype)
        noise_audio = torch.randn_like(clean_audio)
        sigma_video = sigma_video.to(accelerator.device)
        sigma_audio = time_shift_sigma(sigma_video, args.video_flow_shift, args.audio_flow_shift)
        sv = sigma_video.view(-1, 1, 1, 1, 1).to(network_dtype)
        sa = sigma_audio.view(-1, 1, 1, 1).to(network_dtype)
        noisy_video = (1 - sv) * clean_video + sv * noise_video
        noisy_audio = (1 - sa) * clean_audio + sa * noise_audio

        contexts = [value.to(accelerator.device, dtype=network_dtype) for value in batch["h3_text_embed"]]
        tags = [value.to(accelerator.device, dtype=torch.long) for value in batch.get("h3_token_tags", [])]
        if not tags:
            tags = [torch.ones(value.shape[0], device=accelerator.device, dtype=torch.long) for value in contexts]
        vision_context = [bool((value == 0).any()) for value in tags]
        if self.i2v_training and not all(vision_context):
            raise ValueError(
                "H3 I2V text caches have no first-frame vision rows; rerun "
                "minimax_h3_cache_text_encoder_outputs.py with --i2v"
            )
        if not self.i2v_training and any(vision_context):
            raise ValueError("H3 first-frame vision text caches require --i2v training")

        condition_video = None
        if self.i2v_training:
            clean_condition = batch["latents_image"].to(accelerator.device, dtype=network_dtype)
            condition_noise = torch.randn_like(clean_condition)
            condition_sigma = 1.0 - KEYFRAME_CLEAN
            condition_video = KEYFRAME_CLEAN * clean_condition + condition_sigma * condition_noise
        if args.gradient_checkpointing:
            noisy_video.requires_grad_(True)
            noisy_audio.requires_grad_(True)
            if condition_video is not None:
                condition_video.requires_grad_(True)
            for value in contexts:
                value.requires_grad_(True)

        with accelerator.autocast():
            pred_video, pred_audio = transformer(
                noisy_video,
                noisy_audio,
                sigma_video,
                contexts,
                tags,
                condition_video=condition_video,
            )
        target_video = clean_video - noise_video
        target_audio = clean_audio - noise_audio
        video_loss = self._weighted_mse(pred_video.to(network_dtype), target_video, sigma_video, args.weighting_scheme)
        audio_loss = self._weighted_mse(pred_audio.to(network_dtype), target_audio, sigma_audio, args.weighting_scheme)
        loss = (video_loss + args.audio_loss_weight * audio_loss) / (1.0 + args.audio_loss_weight)
        return loss, {"loss/video": float(video_loss.detach()), "loss/audio": float(audio_loss.detach())}


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="train H3 FL2VA using each target video's first frame as the reference image",
    )
    parser.add_argument("--video_flow_shift", type=float, default=12.0, help="H3 video sigma shift")
    parser.add_argument("--audio_flow_shift", type=float, default=3.0, help="H3 audio sigma shift")
    parser.add_argument("--audio_loss_weight", type=float, default=1.0, help="relative audio reconstruction loss weight")
    parser.set_defaults(timestep_sampling="shift", discrete_flow_shift=12.0, mixed_precision="bf16")
    return parser


def main():
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
