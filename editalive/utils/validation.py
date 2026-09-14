import math


def validate_resolution(long_edge, short_edge):
    if short_edge <= 0 or long_edge < short_edge:
        raise ValueError("Output edges must satisfy long_edge >= short_edge > 0.")
    if long_edge % 16 or short_edge % 16:
        raise ValueError("Output edges must be divisible by 16.")


def validate_sampling(sampling_steps, custom_timesteps, shift, num_train_timesteps=1000):
    if sampling_steps <= 0:
        raise ValueError("sample_steps must be positive.")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("sample_shift must be finite and positive.")
    if custom_timesteps is not None:
        if len(custom_timesteps) == 0 or any(
            not 0 < t <= num_train_timesteps for t in custom_timesteps
        ):
            raise ValueError(f"custom_timesteps must be in (0, {num_train_timesteps}].")
        if any(a <= b for a, b in zip(custom_timesteps, custom_timesteps[1:])):
            raise ValueError("custom_timesteps must be strictly decreasing.")
