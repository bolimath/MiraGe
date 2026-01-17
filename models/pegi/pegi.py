from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from lpipsPyTorch import lpips as lpips_fn
except ImportError:  # pragma: no cover - optional dependency
    lpips_fn = None


@dataclass
class PEGIConfig:
    image_height: int
    image_width: int
    num_gaussians: int = 512
    beta: float = 8.0
    light_dim: int = 3
    specular_power: float = 32.0
    lambda_rec: float = 1.0
    lambda_lpips: float = 0.2
    lambda_normal: float = 0.1
    lambda_depth: float = 0.1
    lambda_light: float = 0.1
    lambda_edit: float = 0.1
    lambda_comp: float = 0.1


class PEGIModel(nn.Module):
    """PEGI: Physically-Grounded Editable Gaussian Image."""

    def __init__(self, config: PEGIConfig):
        super().__init__()
        self.config = config
        num = config.num_gaussians

        self.mu = nn.Parameter(torch.rand(num, 2))
        self.log_sigma = nn.Parameter(torch.zeros(num, 2))
        self.theta = nn.Parameter(torch.zeros(num, 1))
        self.depth = nn.Parameter(torch.zeros(num, 1))
        self.normal = nn.Parameter(F.normalize(torch.randn(num, 3), dim=-1))
        self.albedo = nn.Parameter(torch.rand(num, 3))
        self.roughness = nn.Parameter(torch.zeros(num, 1))
        self.opacity = nn.Parameter(torch.rand(num, 1))

        self.light = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))
        self.ambient = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

    def gaussian_weights(self, grid: torch.Tensor) -> torch.Tensor:
        mu = self.mu.unsqueeze(0)
        diff = grid.unsqueeze(2) - mu
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        cos_t = torch.cos(self.theta).unsqueeze(0)
        sin_t = torch.sin(self.theta).unsqueeze(0)
        rot = torch.stack(
            [
                torch.cat([cos_t, -sin_t], dim=-1),
                torch.cat([sin_t, cos_t], dim=-1),
            ],
            dim=-2,
        )
        diff_rot = torch.einsum("bhwn, bhnm -> bhwm", diff, rot)
        exponent = -0.5 * (diff_rot / (sigma + 1e-6)).pow(2).sum(dim=-1)
        return torch.exp(exponent)

    def visibility(self, weights: torch.Tensor) -> torch.Tensor:
        depth = self.depth.view(1, 1, 1, -1)
        logits = -self.config.beta * depth + torch.log(weights + 1e-8)
        return F.softmax(logits, dim=-1)

    def shading(self) -> torch.Tensor:
        light_dir = F.normalize(self.light, dim=0)
        normals = F.normalize(self.normal, dim=-1)
        ndotl = torch.clamp((normals * light_dir).sum(dim=-1, keepdim=True), min=0.0)
        diffuse = self.albedo * ndotl
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=normals.device)
        half_vec = F.normalize(light_dir + view_dir, dim=0)
        spec = torch.clamp((normals * half_vec).sum(dim=-1, keepdim=True), min=0.0)
        specular = spec.pow(self.config.specular_power) * (1.0 - self.roughness)
        return diffuse + specular + self.ambient

    def shading_factor(self, light_dir: torch.Tensor) -> torch.Tensor:
        normals = F.normalize(self.normal, dim=-1)
        ndotl = torch.clamp((normals * light_dir).sum(dim=-1, keepdim=True), min=0.0)
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=normals.device)
        half_vec = F.normalize(light_dir + view_dir, dim=0)
        spec = torch.clamp((normals * half_vec).sum(dim=-1, keepdim=True), min=0.0)
        specular = spec.pow(self.config.specular_power) * (1.0 - self.roughness)
        return ndotl + specular

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        weights = self.gaussian_weights(grid)
        vis = self.visibility(weights)
        color = self.shading()
        opacity = torch.sigmoid(self.opacity).view(1, 1, 1, -1)
        colors = color.view(1, 1, 1, -1, 3)
        contrib = weights.unsqueeze(-1) * vis.unsqueeze(-1) * opacity.unsqueeze(-1) * colors
        return contrib.sum(dim=3).clamp(0.0, 1.0)

    def render(self, height: int, width: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.mu.device
        grid = make_pixel_grid(height, width, device=device)
        return self.forward(grid)


def make_pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1)


def edit_transform(model: PEGIModel, translation: Tuple[float, float]) -> Dict[str, torch.Tensor]:
    mu = model.mu + torch.tensor(translation, device=model.mu.device)
    mu = mu.clamp(0.0, 1.0)
    return {"mu": mu}


def swap_depth(model: PEGIModel, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    depth = model.depth.clone()
    if indices.numel() >= 2:
        depth[indices[0]], depth[indices[1]] = depth[indices[1]], depth[indices[0]]
    return {"depth": depth}


def remove_gaussians(model: PEGIModel, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    opacity = model.opacity.clone()
    opacity[mask] = -10.0
    return {"opacity": opacity}


def normal_smoothness(normals: torch.Tensor) -> torch.Tensor:
    diffs = normals[:, None, :] - normals[None, :, :]
    return diffs.pow(2).sum(dim=-1).mean()


def depth_smoothness(depth: torch.Tensor) -> torch.Tensor:
    diffs = depth[:, None] - depth[None, :]
    return diffs.pow(2).mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if lpips_fn is None:
        return torch.tensor(0.0, device=pred.device)
    pred_in = pred.permute(0, 3, 1, 2)
    target_in = target.permute(0, 3, 1, 2)
    return lpips_fn(pred_in, target_in)


def apply_edit(model: PEGIModel, edits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in edits.items()}


def train_pegi(
    model: PEGIModel,
    image: torch.Tensor,
    steps: int = 1000,
    lr: float = 1e-3,
    seed: int = 13,
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = image.device
    grid = make_pixel_grid(image.shape[1], image.shape[2], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": []}

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(grid)
        weights = model.gaussian_weights(grid)
        rec = reconstruction_loss(pred, image)
        lpips_val = lpips_loss(pred.unsqueeze(0), image.unsqueeze(0))
        normal_reg = normal_smoothness(model.normal)
        depth_reg = depth_smoothness(model.depth)

        light_noise = torch.randn_like(model.light) * 0.05
        light_alt = F.normalize(model.light + light_noise, dim=0)
        original_light = model.light.clone()
        original_factor = model.shading_factor(F.normalize(original_light, dim=0))
        model.light.data.copy_(light_alt)
        alt_factor = model.shading_factor(light_alt)
        epsilon = 1e-4
        albedo_est = (model.albedo * original_factor) / (original_factor + epsilon)
        albedo_alt_est = (model.albedo * alt_factor) / (alt_factor + epsilon)
        albedo_consistency = F.l1_loss(albedo_est, albedo_alt_est)
        model.light.data.copy_(original_light)

        translation = (0.05, -0.05)
        edit = edit_transform(model, translation)
        edited_params = apply_edit(model, edit)
        original_mu = model.mu.clone()
        model.mu.data.copy_(edited_params["mu"])
        edited_pred = model(grid)
        edited_weights = model.gaussian_weights(grid)
        model.mu.data.copy_(original_mu)

        diff_mask = (weights - edited_weights).abs().sum(dim=-1, keepdim=True)
        local_mask = (diff_mask > diff_mask.mean()).float()
        local_loss = ((pred - edited_pred).abs() * (1.0 - local_mask)).mean()

        loss = (
            model.config.lambda_rec * rec
            + model.config.lambda_lpips * lpips_val
            + model.config.lambda_normal * normal_reg
            + model.config.lambda_depth * depth_reg
            + model.config.lambda_light * albedo_consistency
            + model.config.lambda_edit * local_loss
        )
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())

    return history
