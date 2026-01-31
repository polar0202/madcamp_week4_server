"""AdaIN functions for arbitrary style transfer."""

import torch


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for feature normalization."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Apply AdaIN to align content features with style statistics."""
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adaptive_instance_normalization_with_stats(
    content_feat, style_mean, style_std, eps=1e-5
):
    """AdaIN using pre-computed style mean/std (skips style encoding)."""
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat, eps=eps)
    normalized_feat = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    """Calculate mean and std for 3D feature (C, H, W)."""
    assert feat.size()[0] == 3
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    """Matrix square root via SVD."""
    U, D, Vh = torch.linalg.svd(x)
    return torch.mm(torch.mm(U, torch.diag(D.pow(0.5))), Vh)


def coral(source, target):
    """CORAL: Preserve content image color while applying style."""
    # source, target: (C, H, W)
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3, device=source.device)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3, device=target.device)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.linalg.inv(_mat_sqrt(source_f_cov_eye)), source_f_norm)
    )

    source_f_transfer = (
        source_f_norm_transfer * target_f_std.expand_as(source_f_norm)
        + target_f_mean.expand_as(source_f_norm)
    )
    return source_f_transfer.view(source.size())
