import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from models.vlaai_lite import VLAAILite

def get_params(width):
    spatial_dim = 32 * width
    temporal_dim = 64 * width
    model = VLAAILite(in_channels=8, spatial_dim=spatial_dim, temporal_dim=temporal_dim, max_dilation=8)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Width 1x: {get_params(1):,}")
print(f"Width 2x: {get_params(2):,}")
print(f"Width 4x: {get_params(4):,}")
