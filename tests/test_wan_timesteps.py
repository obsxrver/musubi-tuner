from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.wan_train_network import WanNetworkTrainer


def _uniform_args():
    return SimpleNamespace(
        timestep_sampling="uniform",
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
    )


def test_base_timestep_sampler_can_return_sigmas_without_mixing_latents():
    trainer = NetworkTrainer()
    trainer.num_timestep_buckets = None
    latents = torch.zeros(2, 1, 1, 1, 1)
    noise = torch.ones_like(latents)

    sigmas, timesteps = trainer.get_noisy_model_input_and_timesteps(
        _uniform_args(),
        noise,
        latents,
        [0.2, 0.7],
        None,
        torch.device("cpu"),
        torch.float32,
        return_sigmas=True,
    )

    assert torch.allclose(sigmas, torch.tensor([0.2, 0.7]))
    assert torch.allclose(timesteps, torch.tensor([201.0, 701.0]))


@pytest.mark.parametrize(
    ("responses", "expected_sigmas", "expected_timesteps", "is_high_noise"),
    [
        (
            [
                (torch.tensor([0.2, 0.8, 0.1]), torch.tensor([201.0, 801.0, 101.0])),
                (torch.tensor([0.3]), torch.tensor([301.0])),
            ],
            torch.tensor([0.2, 0.3, 0.1]),
            torch.tensor([201.0, 301.0, 101.0]),
            False,
        ),
        (
            [
                (torch.tensor([0.8, 0.2, 0.9]), torch.tensor([801.0, 201.0, 901.0])),
                (torch.tensor([0.7]), torch.tensor([701.0])),
            ],
            torch.tensor([0.8, 0.7, 0.9]),
            torch.tensor([801.0, 701.0, 901.0]),
            True,
        ),
    ],
)
def test_wan_high_low_timestep_sampling_filters_candidates_in_batches(
    responses, expected_sigmas, expected_timesteps, is_high_noise
):
    trainer = WanNetworkTrainer()
    trainer.high_low_training = True
    trainer.timestep_boundary = 0.5
    trainer.num_timestep_buckets = None

    responses = iter(responses)
    sampled_batch_sizes = []

    def fake_sample(
        self,
        args,
        noise,
        latents,
        timesteps,
        noise_scheduler,
        device,
        dtype,
        return_sigmas=False,
    ):
        assert return_sigmas
        sampled_batch_sizes.append(noise.shape[0])
        sigmas, sampled_timesteps = next(responses)
        assert sigmas.shape[0] == noise.shape[0]
        return sigmas, sampled_timesteps

    latents = torch.zeros(3, 1, 1, 1, 1)
    noise = torch.ones_like(latents)
    with patch.object(NetworkTrainer, "get_noisy_model_input_and_timesteps", autospec=True, side_effect=fake_sample):
        noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
            SimpleNamespace(), noise, latents, None, None, torch.device("cpu"), torch.float32
        )

    assert trainer.next_model_is_high_noise is is_high_noise
    assert sampled_batch_sizes == [3, 1]
    assert torch.allclose(noisy_model_input.flatten(), expected_sigmas)
    assert torch.equal(timesteps, expected_timesteps)
