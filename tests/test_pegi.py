import os
import sys
import torch
import pytest

# Ensure repository root is on sys.path so we can import models.pegi.pegi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pegi.pegi import (
    make_pixel_grid,
    PEGIConfig,
    PEGIModel,
    edit_transform,
    swap_depth,
    remove_gaussians,
    train_pegi,
)

def test_make_pixel_grid_shape_and_range():
    h, w = 4, 6
    grid = make_pixel_grid(h, w, device=torch.device("cpu"))
    assert grid.shape == (h, w, 2)
    assert torch.all(grid >= 0.0) and torch.all(grid <= 1.0)

def test_pegi_render_shape_and_value_range():
    h, w = 8, 8
    cfg = PEGIConfig(image_height=h, image_width=w, num_gaussians=8)
    model = PEGIModel(cfg)
    out = model.render(h, w, device=torch.device("cpu"))
    # Expected: height x width x 3
    assert out.ndim == 3
    assert out.shape == (h, w, 3)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

def test_edit_transform_clamps_mu():
    cfg = PEGIConfig(image_height=4, image_width=4, num_gaussians=2)
    model = PEGIModel(cfg)
    # Put mu near the edge and translate outside [0,1]
    model.mu.data.copy_(torch.tensor([[0.95, 0.95], [0.02, 0.01]]))
    translation = (0.2, 0.2)
    edits = edit_transform(model, translation)
    mu = edits["mu"]
    assert torch.all(mu >= 0.0) and torch.all(mu <= 1.0)

def test_swap_depth_swaps_entries():
    cfg = PEGIConfig(image_height=4, image_width=4, num_gaussians=3)
    model = PEGIModel(cfg)
    # Set known depths
    model.depth.data.copy_(torch.tensor([[0.1], [0.2], [0.3]]))
    indices = torch.tensor([0, 2])
    out = swap_depth(model, indices)
    depth = out["depth"].view(-1)
    assert pytest.approx(depth[0].item(), rel=1e-6) == 0.3
    assert pytest.approx(depth[2].item(), rel=1e-6) == 0.1
    # middle element unchanged
    assert pytest.approx(depth[1].item(), rel=1e-6) == 0.2

def test_remove_gaussians_sets_opacity_low():
    cfg = PEGIConfig(image_height=4, image_width=4, num_gaussians=4)
    model = PEGIModel(cfg)
    # mask to remove first two gaussians
    mask = torch.tensor([True, True, False, False])
    out = remove_gaussians(model, mask)
    opacity = out["opacity"].view(-1)
    assert opacity[0].item() == pytest.approx(-10.0)
    assert opacity[1].item() == pytest.approx(-10.0)
    # others unchanged (should be in original range 0..1 before sigmoid)
    assert opacity[2].item() != pytest.approx(-10.0)

def test_train_pegi_runs_short():
    h, w = 6, 6
    cfg = PEGIConfig(image_height=h, image_width=w, num_gaussians=6, lambda_lpips=0.0)
    model = PEGIModel(cfg)
    # small random image (H, W, 3)
    torch.manual_seed(0)
    image = torch.rand(h, w, 3)
    history = train_pegi(model, image, steps=2, lr=1e-3, seed=0)
    assert "loss" in history
    assert len(history["loss"]) == 2
    assert all(isinstance(x, float) or isinstance(x, (int,)) for x in history["loss"])
