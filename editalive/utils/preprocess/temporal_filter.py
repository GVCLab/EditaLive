import numpy as np


def _alpha(cutoff, freq):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau * freq)


def _one_euro_forward(x, freq, min_cutoff, beta, d_cutoff, weights=None):
    """Filter x of shape (T, D) along axis 0. weights: (T, D) in [0,1] or None.

    `weights` (typically keypoint confidence) attenuates how much a sample is
    trusted: w=1 is the plain filter, w->0 holds the previous estimate. This
    stops a single low-confidence frame from yanking the trajectory.
    """
    T = x.shape[0]
    out = np.empty_like(x)
    out[0] = x[0]
    x_prev = x[0]
    dx_prev = np.zeros_like(x[0])

    a_d = _alpha(d_cutoff, freq)

    for t in range(1, T):
        dx = (x[t] - x_prev) * freq
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev

        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = _alpha(cutoff, freq)

        if weights is not None:
            a = a * weights[t]

        x_hat = a * x[t] + (1.0 - a) * x_prev

        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat

    return out


def one_euro_filter(x, freq=16.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0,
                    weights=None, bidirectional=True):
    """Smooth a (T, D) array along time.

    Args:
        x: (T, D) float array. For keypoints, flatten (T, K, 2) -> (T, K*2).
        freq: sampling rate in Hz (video fps).
        min_cutoff: cutoff at zero speed. Lower = smoother but laggier.
        beta: speed coupling. Higher = follows fast motion more eagerly.
        d_cutoff: cutoff of the internal speed low-pass.
        weights: optional (T, D) confidence in [0, 1].
        bidirectional: average a forward and a backward pass (zero phase lag).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (T, D), got {x.shape}")
    if x.shape[0] < 2:
        return x.copy()

    fwd = _one_euro_forward(x, freq, min_cutoff, beta, d_cutoff, weights)
    if not bidirectional:
        return fwd

    w_rev = None if weights is None else weights[::-1]
    bwd = _one_euro_forward(x[::-1], freq, min_cutoff, beta, d_cutoff, w_rev)[::-1]
    return 0.5 * (fwd + bwd)


# Wholebody-133 keypoint groups, matching split_kp2ds_for_aa in pose2d_utils.py.
# HEAD is the nose/eyes/ears subset of the body group (indices 0-4): those land in
# `keypoints_body` and so are the only head points that actually get rendered into
# src_pose.mp4. They jitter with the face (measured 1.6-3.6 px/frame accel, versus
# the face centroid's 1.4-3.3), because both come from the same tiny corner of the
# 256x192 ViTPose crop -- so they need the face's parameters, not the body's.
KP_GROUPS = {
    "head": list(range(0, 5)),
    "body": list(range(5, 22)),
    "face": list(range(22, 91)),
    "hands": list(range(91, 133)),
}

# Per-group (min_cutoff, beta). Rationale:
#   head/face -- jitter is 79-94% rigid translation of the whole face, not
#     expression, and the face landmarks are consumed only as a min/max bbox
#     (get_face_bboxes), so there is no fine detail to preserve. Filter hard.
#   body -- 1.0/0.05 already lands at 41-58% of raw accel with no visible lag.
#   hands -- ViTPose flips between wrong hand hypotheses at 20-36 px/frame; a
#     smoother reads that as genuine speed and passes it through. Tightening the
#     cutoff here buys little and costs real motion, so leave it at the default.
KP_GROUP_PARAMS = {
    "head":  (0.4, 0.01),
    "body":  (1.0, 0.05),
    "face":  (0.4, 0.01),
    "hands": (1.0, 0.05),
}


def smooth_kp2ds(kp2ds, freq=16.0, min_cutoff=1.0, beta=0.05,
                 use_confidence=True, conf_floor=0.3, bidirectional=True,
                 group_params=None):
    """Smooth a (T, K, 3) keypoint sequence: xy is filtered, confidence is not.

    Coordinates are expected in PIXELS (that is what Pose2d produces before
    load_pose_metas_from_kp2ds_seq normalises by width/height), so `beta` is
    tuned against pixel-scale speeds.

    group_params: optional {group_name: (min_cutoff, beta)} to filter regions of
        the skeleton with different strengths -- the head/face need much heavier
        smoothing than the body (see KP_GROUP_PARAMS). Pass KP_GROUP_PARAMS for
        the tuned defaults, or None to use one (min_cutoff, beta) for all points.
        Only applies to the 133-keypoint wholebody layout; ignored otherwise.
    """
    kp2ds = np.asarray(kp2ds, dtype=np.float64)
    if kp2ds.ndim != 3 or kp2ds.shape[-1] < 3:
        raise ValueError(f"expected (T, K, 3+), got {kp2ds.shape}")

    T, K = kp2ds.shape[0], kp2ds.shape[1]

    weights_full = None
    if use_confidence:
        conf = np.clip(kp2ds[..., 2], 0.0, 1.0)
        conf = conf_floor + (1.0 - conf_floor) * conf
        weights_full = conf

    out = kp2ds.copy()

    if group_params and K == 133:
        groups = [(KP_GROUPS[n], p) for n, p in group_params.items() if n in KP_GROUPS]
        covered = sorted(i for idx, _ in groups for i in idx)
        missing = sorted(set(range(K)) - set(covered))
        if missing:
            groups.append((missing, (min_cutoff, beta)))
    else:
        groups = [(list(range(K)), (min_cutoff, beta))]

    for idx, (g_min_cutoff, g_beta) in groups:
        idx = np.asarray(idx, dtype=int)
        n = len(idx)
        xy = kp2ds[:, idx, :2].reshape(T, n * 2)
        w = None
        if weights_full is not None:
            w = np.repeat(weights_full[:, idx], 2, axis=1).reshape(T, n * 2)
        sm = one_euro_filter(xy, freq=freq, min_cutoff=g_min_cutoff, beta=g_beta,
                             weights=w, bidirectional=bidirectional)
        out[:, idx, :2] = sm.reshape(T, n, 2)

    return out


def smooth_face_bboxes(bboxes, image_shape, freq=16.0, min_cutoff=0.5,
                       beta=0.02, bidirectional=True):
    """Smooth a sequence of get_face_bboxes() results: [min_x, max_x, min_y, max_y].

    get_face_bboxes reduces 68 face landmarks to their min/max, so the crop is an
    *extremum* statistic: one noisy landmark on one frame moves the whole box, and
    the 512x512 face crop jumps with it. Filtering the box directly is both cheaper
    (4 channels instead of 136) and better targeted than filtering the landmarks,
    since the landmark positions themselves are never used for anything else.

    Note the channel order is (x1, x2, y1, y2), not xyxy -- irrelevant to a
    per-channel filter, but it means smooth_bboxes' docstring does not apply.

    Returns int boxes clamped to the frame and guaranteed non-empty, since the
    result indexes directly into the frame (frame[y1:y2, x1:x2]).
    """
    arr = np.asarray(bboxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected (T, 4), got {arr.shape}")

    h, w = image_shape
    sm = one_euro_filter(arr, freq=freq, min_cutoff=min_cutoff, beta=beta,
                         bidirectional=bidirectional)

    out = np.rint(sm).astype(int)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)          # x1
    out[:, 1] = np.clip(out[:, 1], 1, w)              # x2
    out[:, 2] = np.clip(out[:, 2], 0, h - 1)          # y1
    out[:, 3] = np.clip(out[:, 3], 1, h)              # y2
    # Rounding/clamping can collapse a box; keep at least one pixel of extent.
    out[:, 1] = np.maximum(out[:, 1], out[:, 0] + 1)
    out[:, 3] = np.maximum(out[:, 3], out[:, 2] + 1)
    return out
