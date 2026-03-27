"""
ground_model.py — Build and apply a reusable ground height model from consolidated voxels
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def build_ground_model(vdf, tile_size=10.0, percentile=5):
    """
    From a consolidated voxel DataFrame, build a ground height lookup.
    
    Returns a function: ground_z = f(x_world, y_world) that gives the
    estimated ground height at any world XY coordinate.
    
    Args:
        vdf: DataFrame with columns world_x, world_y, world_z, hit_count
        tile_size: size of XY tiles in meters for ground estimation
        percentile: use this percentile of z_min within each tile as ground
    
    Returns:
        ground_fn: callable(x, y) -> z_ground  (works with arrays)
        ground_info: dict with metadata for debugging
    """
    x = vdf['world_x'].values
    y = vdf['world_y'].values
    z = vdf['z_min'].values  # lowest z within each voxel
    
    # Build tile grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_edges = np.arange(x_min, x_max + tile_size, tile_size)
    y_edges = np.arange(y_min, y_max + tile_size, tile_size)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Compute ground height per tile
    ground_grid = np.full((len(x_centers), len(y_centers)), np.nan)
    
    x_idx = np.clip(
        np.searchsorted(x_edges, x, side='right') - 1,
        0, len(x_centers) - 1
    )
    y_idx = np.clip(
        np.searchsorted(y_edges, y, side='right') - 1,
        0, len(y_centers) - 1
    )
    
    for xi in range(len(x_centers)):
        for yi in range(len(y_centers)):
            mask = (x_idx == xi) & (y_idx == yi)
            if mask.sum() > 0:
                ground_grid[xi, yi] = np.percentile(z[mask], percentile)
    
    # Fill NaN tiles with nearest neighbor
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(ground_grid)
    if nan_mask.any() and not nan_mask.all():
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        ground_grid = ground_grid[tuple(indices)]
    elif nan_mask.all():
        ground_grid[:] = z.min()
    
    # Build interpolator
    interpolator = RegularGridInterpolator(
        (x_centers, y_centers), ground_grid,
        method='linear', bounds_error=False,
        fill_value=np.nanmedian(ground_grid)
    )
    
    def ground_fn(x_query, y_query):
        """Return ground z for arrays of x, y world coordinates."""
        pts = np.column_stack([x_query, y_query])
        return interpolator(pts)
    
    return ground_fn, {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'tile_size': tile_size,
        'ground_grid': ground_grid,
        'x_centers': x_centers,
        'y_centers': y_centers,
    }