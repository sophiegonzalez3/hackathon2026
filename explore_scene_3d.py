"""
explore_scene_3d.py — Immersive First-Person 3D Scene Explorer
===============================================================
Creates a video game-like exploration experience with:
  - WASD movement + mouse look (first-person controls)
  - Sprint with Shift, fly up/down with Space/Ctrl
  - Point cloud and 3D bounding boxes
  - Minimap, crosshair, coordinates HUD
  - Collision-free flying exploration

Usage:
    python explore_scene_3d.py \
        --voxels /content/hackathon/consolidated/scene_3_voxels.csv \
        --bboxes /content/hackathon/gt_runs/gt_bboxes_run_05_merge_clean.csv \
        --scene 3 \
        --output /content/hackathon/scene_3_explorer.html
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Class definitions
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_COLORS_HEX = {
    'Background':    0x555555,
    'Antenna':       0x3498db,
    'Cable':         0xf39c12,
    'Electric Pole': 0x9b59b6,
    'Wind Turbine':  0x2ecc71,
}

BBOX_COLORS_HEX = {
    'Antenna':       0x2980b9,
    'Cable':         0xd68910,
    'Electric Pole': 0x7d3c98,
    'Wind Turbine':  0x1e8449,
}


def load_data(voxels_path, bboxes_path, scene_num, max_points=200000, obstacles_only=False):
    """Load voxels and bounding boxes."""
    
    # Load voxels
    print(f"Loading voxels from {voxels_path}...")
    voxels_df = pd.read_csv(voxels_path)
    print(f"  ✓ {len(voxels_df):,} voxels loaded")
    print(f"  Columns: {voxels_df.columns.tolist()}")
    
    # Filter to obstacles only if requested
    if obstacles_only and 'is_obstacle' in voxels_df.columns:
        n_before = len(voxels_df)
        voxels_df = voxels_df[voxels_df['is_obstacle'] == True].copy()
        print(f"  Filtered to obstacles only: {len(voxels_df):,} / {n_before:,}")
    elif obstacles_only and 'class_id' in voxels_df.columns:
        n_before = len(voxels_df)
        voxels_df = voxels_df[voxels_df['class_id'] >= 0].copy()
        print(f"  Filtered to obstacles only (class_id >= 0): {len(voxels_df):,} / {n_before:,}")
    
    # Auto-detect coordinate columns
    cols = voxels_df.columns.tolist()
    
    # Try common column name patterns (prioritize world coordinates)
    x_candidates = ['world_x', 'x', 'X', 'voxel_x', 'cx', 'center_x', 'pos_x']
    y_candidates = ['world_y', 'y', 'Y', 'voxel_y', 'cy', 'center_y', 'pos_y']
    z_candidates = ['world_z', 'z', 'Z', 'voxel_z', 'cz', 'center_z', 'pos_z']
    
    x_col = next((c for c in x_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    z_col = next((c for c in z_candidates if c in cols), None)
    
    # If not found, try to find columns containing x, y, z (but not vx, vy, vz which are indices)
    if x_col is None:
        x_col = next((c for c in cols if 'x' in c.lower() and 'max' not in c.lower() and c not in ['vx']), None)
    if y_col is None:
        y_col = next((c for c in cols if 'y' in c.lower() and 'max' not in c.lower() and c not in ['vy']), None)
    if z_col is None:
        z_col = next((c for c in cols if 'z' in c.lower() and 'max' not in c.lower() and c not in ['vz', 'z_min', 'z_max', 'z_range', 'z_mean', 'z_std']), None)
    
    # Last resort: assume first 3 numeric columns are x, y, z
    if x_col is None or y_col is None or z_col is None:
        numeric_cols = voxels_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            print(f"  ⚠ Could not identify x/y/z columns, using first 3 numeric: {numeric_cols[:3]}")
            x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        else:
            raise ValueError(f"Cannot identify coordinate columns in: {cols}")
    
    print(f"  Using columns: x={x_col}, y={y_col}, z={z_col}")
    xyz = voxels_df[[x_col, y_col, z_col]].values
    
    # Get labels
    if 'label' in voxels_df.columns:
        labels = voxels_df['label'].values
    elif 'class_label' in voxels_df.columns:
        labels = voxels_df['class_label'].values
    elif 'class_id' in voxels_df.columns:
        id_map = {-1: 'Background', 0: 'Antenna', 1: 'Cable', 2: 'Electric Pole', 3: 'Wind Turbine'}
        labels = np.array([id_map.get(int(cid), 'Background') for cid in voxels_df['class_id']])
    else:
        labels = np.array(['Background'] * len(xyz))
    
    # Subsample if needed
    if len(xyz) > max_points:
        print(f"  Subsampling to {max_points:,} points...")
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        labels = labels[idx]
    
    # Load bboxes
    print(f"Loading bboxes from {bboxes_path}...")
    bboxes_df = pd.read_csv(bboxes_path)
    scene_name = f'scene_{scene_num}'
    bboxes_df = bboxes_df[bboxes_df['scene'] == scene_name].copy()
    print(f"  ✓ {len(bboxes_df)} bboxes for {scene_name}")
    
    # Transform bboxes from ego frame to world frame
    if 'ego_x' in bboxes_df.columns and 'ego_y' in bboxes_df.columns:
        print("  Transforming bboxes from ego frame to world frame...")
        
        # Get ego pose for each bbox
        # NOTE: ego coordinates appear to be in CENTIMETERS, convert to meters
        ego_x = bboxes_df['ego_x'].values / 100.0  # cm -> m
        ego_y = bboxes_df['ego_y'].values / 100.0  # cm -> m
        ego_z = bboxes_df['ego_z'].values / 100.0 if 'ego_z' in bboxes_df.columns else np.zeros(len(bboxes_df))
        
        # NOTE: ego_yaw appears to be in DEGREES, convert to radians
        ego_yaw_deg = bboxes_df['ego_yaw'].values if 'ego_yaw' in bboxes_df.columns else np.zeros(len(bboxes_df))
        ego_yaw = np.deg2rad(ego_yaw_deg)
        
        # Get bbox centers in ego frame (these are already in meters)
        bbox_x = bboxes_df['bbox_center_x'].values
        bbox_y = bboxes_df['bbox_center_y'].values
        bbox_z = bboxes_df['bbox_center_z'].values
        bbox_yaw = bboxes_df['bbox_yaw'].values if 'bbox_yaw' in bboxes_df.columns else np.zeros(len(bboxes_df))
        
        print(f"    Sample ego pose (converted): x={ego_x[0]:.1f}m, y={ego_y[0]:.1f}m, z={ego_z[0]:.1f}m, yaw={np.rad2deg(ego_yaw[0]):.1f}°")
        print(f"    Sample bbox (ego frame): x={bbox_x[0]:.1f}, y={bbox_y[0]:.1f}, z={bbox_z[0]:.1f}")
        
        # Transform to world frame: rotate by ego_yaw, then translate by ego position
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        
        # Rotated bbox center
        world_bbox_x = cos_yaw * bbox_x - sin_yaw * bbox_y + ego_x
        world_bbox_y = sin_yaw * bbox_x + cos_yaw * bbox_y + ego_y
        world_bbox_z = bbox_z + ego_z
        
        # World yaw = ego_yaw + bbox_yaw
        world_bbox_yaw = ego_yaw + bbox_yaw
        
        print(f"    Sample bbox (world frame): x={world_bbox_x[0]:.1f}, y={world_bbox_y[0]:.1f}, z={world_bbox_z[0]:.1f}")
        
        # Update dataframe
        bboxes_df['bbox_center_x'] = world_bbox_x
        bboxes_df['bbox_center_y'] = world_bbox_y
        bboxes_df['bbox_center_z'] = world_bbox_z
        bboxes_df['bbox_yaw'] = world_bbox_yaw
        
        print(f"  ✓ Transformed {len(bboxes_df)} bboxes to world coordinates")
    else:
        print("  ⚠ No ego pose columns found, using bbox coordinates as-is")
    
    return xyz, labels, bboxes_df


def generate_html(xyz, labels, bboxes_df, scene_num):
    """Generate the immersive 3D explorer HTML."""
    
    # Prepare point data by class
    points_by_class = {}
    for cls in ['Background', 'Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']:
        mask = labels == cls
        if mask.any():
            pts = xyz[mask].tolist()
            points_by_class[cls] = pts
    
    # Prepare bbox data
    bboxes_list = []
    for _, row in bboxes_df.iterrows():
        bboxes_list.append({
            'cx': float(row['bbox_center_x']),
            'cy': float(row['bbox_center_y']),
            'cz': float(row['bbox_center_z']),
            'w': float(row['bbox_width']),
            'l': float(row['bbox_length']),
            'h': float(row['bbox_height']),
            'yaw': float(row.get('bbox_yaw', 0)),
            'label': row['class_label'],
        })
    
    # Calculate scene bounds for spawn position
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    
    print(f"\n  Point cloud bounds:")
    print(f"    X: [{x_min:.1f}, {x_max:.1f}]")
    print(f"    Y: [{y_min:.1f}, {y_max:.1f}]")
    print(f"    Z: [{z_min:.1f}, {z_max:.1f}]")
    
    if len(bboxes_list) > 0:
        bbox_x = [b['cx'] for b in bboxes_list]
        bbox_y = [b['cy'] for b in bboxes_list]
        bbox_z = [b['cz'] for b in bboxes_list]
        print(f"  Bbox centers range:")
        print(f"    X: [{min(bbox_x):.1f}, {max(bbox_x):.1f}]")
        print(f"    Y: [{min(bbox_y):.1f}, {max(bbox_y):.1f}]")
        print(f"    Z: [{min(bbox_z):.1f}, {max(bbox_z):.1f}]")
    
    spawn_x = (x_min + x_max) / 2
    spawn_y = (y_min + y_max) / 2
    spawn_z = z_max + 20  # Start above the scene
    
    print(f"  Spawn position: ({spawn_x:.1f}, {spawn_y:.1f}, {spawn_z:.1f})")
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎮 Scene {scene_num} Explorer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            overflow: hidden; 
            background: #000; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        #container {{ width: 100vw; height: 100vh; }}
        
        /* Crosshair */
        #crosshair {{
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 100;
        }}
        #crosshair::before, #crosshair::after {{
            content: '';
            position: absolute;
            background: rgba(255,255,255,0.8);
        }}
        #crosshair::before {{
            width: 20px; height: 2px;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
        }}
        #crosshair::after {{
            width: 2px; height: 20px;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
        }}
        
        /* HUD */
        #hud {{
            position: fixed;
            bottom: 20px; left: 20px;
            color: #fff;
            font-size: 14px;
            text-shadow: 1px 1px 2px #000;
            z-index: 100;
            background: rgba(0,0,0,0.6);
            padding: 15px 20px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #hud .title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #3498db;
        }}
        #hud .coords {{
            font-family: 'Courier New', monospace;
            color: #2ecc71;
        }}
        #hud .speed {{
            color: #f39c12;
            margin-top: 5px;
        }}
        
        /* Controls help */
        #controls {{
            position: fixed;
            top: 20px; right: 20px;
            color: #fff;
            font-size: 12px;
            z-index: 100;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            max-width: 220px;
        }}
        #controls h3 {{
            margin-bottom: 10px;
            color: #3498db;
            font-size: 14px;
        }}
        #controls .key {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px;
            font-family: monospace;
        }}
        
        /* Minimap */
        #minimap {{
            position: fixed;
            bottom: 20px; right: 20px;
            width: 180px; height: 180px;
            background: rgba(0,0,0,0.8);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            z-index: 100;
            overflow: hidden;
        }}
        #minimap canvas {{
            width: 100%; height: 100%;
        }}
        
        /* Legend */
        #legend {{
            position: fixed;
            top: 20px; left: 20px;
            color: #fff;
            font-size: 13px;
            z-index: 100;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #legend h3 {{
            margin-bottom: 10px;
            color: #fff;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            cursor: pointer;
            opacity: 1;
            transition: opacity 0.2s;
        }}
        .legend-item.hidden {{
            opacity: 0.4;
        }}
        .legend-color {{
            width: 16px; height: 16px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        
        /* Start screen */
        #start-screen {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            color: #fff;
        }}
        #start-screen h1 {{
            font-size: 48px;
            margin-bottom: 20px;
            color: #3498db;
        }}
        #start-screen p {{
            font-size: 18px;
            margin-bottom: 30px;
            color: #aaa;
        }}
        #start-btn {{
            padding: 15px 40px;
            font-size: 20px;
            background: #3498db;
            color: #fff;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
        }}
        #start-btn:hover {{
            background: #2980b9;
            transform: scale(1.05);
        }}
        
        /* Loading */
        #loading {{
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: #fff;
            font-size: 24px;
            z-index: 500;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="crosshair"></div>
    
    <div id="hud">
        <div class="title">🎮 Scene {scene_num} Explorer</div>
        <div class="coords">
            X: <span id="pos-x">0.0</span> m<br>
            Y: <span id="pos-y">0.0</span> m<br>
            Z: <span id="pos-z">0.0</span> m
        </div>
        <div class="speed">Speed: <span id="speed-display">10</span> m/s</div>
    </div>
    
    <div id="controls">
        <h3>🎮 Controls (AZERTY)</h3>
        <span class="key">Z</span><span class="key">Q</span><span class="key">S</span><span class="key">D</span> Move<br>
        <span class="key">Space</span> Fly Up<br>
        <span class="key">Ctrl</span> Fly Down<br>
        <span class="key">R</span> / <span class="key">E</span> Rotate Left/Right<br>
        <span class="key">Shift</span> Sprint<br>
        <span class="key">Scroll</span> Adjust Speed<br>
        <span class="key">T</span> Reset Position<br>
        <span class="key">B</span> Toggle BBoxes<br>
        <span class="key">ESC</span> Release Mouse
    </div>
    
    <div id="legend">
        <h3>📊 Classes</h3>
        <div class="legend-item" data-class="Background">
            <div class="legend-color" style="background:#555555"></div>
            <span>Background</span>
        </div>
        <div class="legend-item" data-class="Antenna">
            <div class="legend-color" style="background:#3498db"></div>
            <span>Antenna</span>
        </div>
        <div class="legend-item" data-class="Cable">
            <div class="legend-color" style="background:#f39c12"></div>
            <span>Cable</span>
        </div>
        <div class="legend-item" data-class="Electric Pole">
            <div class="legend-color" style="background:#9b59b6"></div>
            <span>Electric Pole</span>
        </div>
        <div class="legend-item" data-class="Wind Turbine">
            <div class="legend-color" style="background:#2ecc71"></div>
            <span>Wind Turbine</span>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
        <h3>📦 Bounding Boxes</h3>
        <div class="legend-item" data-bbox="all" id="bbox-toggle">
            <div class="legend-color" style="background:#e74c3c; border: 2px solid #fff;"></div>
            <span>Show All BBoxes</span>
        </div>
        <div class="legend-item" data-bbox="Antenna">
            <div class="legend-color" style="background:#2980b9; border: 2px dashed #fff;"></div>
            <span>Antenna BBox</span>
        </div>
        <div class="legend-item" data-bbox="Cable">
            <div class="legend-color" style="background:#d68910; border: 2px dashed #fff;"></div>
            <span>Cable BBox</span>
        </div>
        <div class="legend-item" data-bbox="Electric Pole">
            <div class="legend-color" style="background:#7d3c98; border: 2px dashed #fff;"></div>
            <span>Pole BBox</span>
        </div>
        <div class="legend-item" data-bbox="Wind Turbine">
            <div class="legend-color" style="background:#1e8449; border: 2px dashed #fff;"></div>
            <span>Turbine BBox</span>
        </div>
    </div>
    
    <div id="minimap">
        <canvas id="minimap-canvas"></canvas>
    </div>
    
    <div id="start-screen">
        <h1>🗺️ Scene {scene_num}</h1>
        <p>Explore the LiDAR point cloud in first-person</p>
        <button id="start-btn">🎮 Click to Start</button>
    </div>
    
    <div id="loading">Loading point cloud...</div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // ═══════════════════════════════════════════════════════════════════
        // DATA
        // ═══════════════════════════════════════════════════════════════════
        const POINTS_BY_CLASS = {json.dumps(points_by_class)};
        const BBOXES = {json.dumps(bboxes_list)};
        const SPAWN = {{ x: {spawn_x}, y: {spawn_y}, z: {spawn_z} }};
        const BOUNDS = {{
            xMin: {x_min}, xMax: {x_max},
            yMin: {y_min}, yMax: {y_max},
            zMin: {z_min}, zMax: {z_max}
        }};
        
        const CLASS_COLORS = {{
            'Background': 0x555555,
            'Antenna': 0x3498db,
            'Cable': 0xf39c12,
            'Electric Pole': 0x9b59b6,
            'Wind Turbine': 0x2ecc71
        }};
        
        const BBOX_COLORS = {{
            'Antenna': 0x2980b9,
            'Cable': 0xd68910,
            'Electric Pole': 0x7d3c98,
            'Wind Turbine': 0x1e8449
        }};

        // ═══════════════════════════════════════════════════════════════════
        // THREE.JS SETUP
        // ═══════════════════════════════════════════════════════════════════
        let scene, camera, renderer;
        let pointClouds = {{}};
        let bboxObjects = {{'all': [], 'Antenna': [], 'Cable': [], 'Electric Pole': [], 'Wind Turbine': []}};
        let bboxVisible = {{'all': true, 'Antenna': true, 'Cable': true, 'Electric Pole': true, 'Wind Turbine': true}};
        let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false;
        let moveUp = false, moveDown = false, sprint = false;
        let rotateLeft = false, rotateRight = false;
        let baseSpeed = 10;
        let rotationSpeed = 1.5;
        let velocity = new THREE.Vector3();
        let direction = new THREE.Vector3();
        let euler = new THREE.Euler(0, 0, 0, 'YXZ');
        let isLocked = false;
        let prevTime = performance.now();

        // Minimap
        let minimapCtx;
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            scene.fog = new THREE.Fog(0x0a0a0a, 100, 500);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
            camera.position.set(SPAWN.x, SPAWN.y, SPAWN.z);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            // Grid helper
            const gridSize = Math.max(BOUNDS.xMax - BOUNDS.xMin, BOUNDS.yMax - BOUNDS.yMin);
            const gridHelper = new THREE.GridHelper(gridSize * 1.5, 50, 0x333333, 0x222222);
            gridHelper.position.set((BOUNDS.xMin + BOUNDS.xMax) / 2, BOUNDS.zMin - 1, (BOUNDS.yMin + BOUNDS.yMax) / 2);
            gridHelper.rotation.x = Math.PI / 2;  // Rotate to XY plane (Z-up)
            scene.add(gridHelper);
            
            // Load points
            loadPointClouds();
            
            // Load bboxes
            loadBoundingBoxes();
            
            // Setup minimap
            setupMinimap();
            
            // Event listeners
            setupControls();
            
            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            // Start animation
            animate();
        }}
        
        function loadPointClouds() {{
            for (const [className, points] of Object.entries(POINTS_BY_CLASS)) {{
                if (points.length === 0) continue;
                
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(points.length * 3);
                
                for (let i = 0; i < points.length; i++) {{
                    positions[i * 3] = points[i][0];
                    positions[i * 3 + 1] = points[i][1];
                    positions[i * 3 + 2] = points[i][2];
                }}
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                const size = className === 'Background' ? 1.5 : 2.5;
                const opacity = className === 'Background' ? 0.4 : 0.9;
                
                const material = new THREE.PointsMaterial({{
                    color: CLASS_COLORS[className],
                    size: size,
                    sizeAttenuation: true,
                    transparent: true,
                    opacity: opacity,
                }});
                
                const pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);
                pointClouds[className] = pointCloud;
            }}
            console.log('Point clouds loaded');
        }}
        
        function loadBoundingBoxes() {{
            console.log('Loading bboxes, count:', BBOXES.length);
            if (BBOXES.length > 0) {{
                console.log('First bbox:', BBOXES[0]);
            }}
            
            for (const bbox of BBOXES) {{
                // Skip invalid bboxes
                if (isNaN(bbox.cx) || isNaN(bbox.cy) || isNaN(bbox.cz)) {{
                    console.warn('Skipping bbox with NaN coordinates:', bbox);
                    continue;
                }}
                
                const geometry = new THREE.BoxGeometry(bbox.w, bbox.l, bbox.h);
                const edges = new THREE.EdgesGeometry(geometry);
                const material = new THREE.LineBasicMaterial({{
                    color: BBOX_COLORS[bbox.label] || 0xff0000,
                    linewidth: 3,
                    transparent: false,
                }});
                
                const wireframe = new THREE.LineSegments(edges, material);
                wireframe.position.set(bbox.cx, bbox.cy, bbox.cz);
                wireframe.rotation.z = bbox.yaw || 0;
                wireframe.visible = true;  // Ensure visible
                
                scene.add(wireframe);
                
                // Track by class
                bboxObjects['all'].push(wireframe);
                if (bboxObjects[bbox.label]) {{
                    bboxObjects[bbox.label].push(wireframe);
                }}
            }}
            console.log('Bounding boxes loaded:', bboxObjects['all'].length);
            console.log('Bbox objects by class:', Object.keys(bboxObjects).map(k => k + ': ' + bboxObjects[k].length).join(', '));
        }}
        
        function setupMinimap() {{
            const canvas = document.getElementById('minimap-canvas');
            canvas.width = 180;
            canvas.height = 180;
            minimapCtx = canvas.getContext('2d');
        }}
        
        function updateMinimap() {{
            const ctx = minimapCtx;
            const w = 180, h = 180;
            
            // Clear
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(0, 0, w, h);
            
            // Scale
            const scaleX = w / (BOUNDS.xMax - BOUNDS.xMin + 20);
            const scaleY = h / (BOUNDS.yMax - BOUNDS.yMin + 20);
            const scale = Math.min(scaleX, scaleY) * 0.9;
            
            const offsetX = w / 2;
            const offsetY = h / 2;
            const centerX = (BOUNDS.xMin + BOUNDS.xMax) / 2;
            const centerY = (BOUNDS.yMin + BOUNDS.yMax) / 2;
            
            // Draw bboxes
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            for (const bbox of BBOXES) {{
                const x = offsetX + (bbox.cx - centerX) * scale;
                const y = offsetY - (bbox.cy - centerY) * scale;
                const bw = bbox.w * scale;
                const bh = bbox.l * scale;
                ctx.strokeRect(x - bw/2, y - bh/2, bw, bh);
            }}
            
            // Draw player position
            const px = offsetX + (camera.position.x - centerX) * scale;
            const py = offsetY - (camera.position.y - centerY) * scale;
            
            // Direction indicator
            const dir = new THREE.Vector3(0, 1, 0);
            dir.applyQuaternion(camera.quaternion);
            
            ctx.fillStyle = '#3498db';
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
            
            // Direction line
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px + dir.x * 15, py - dir.y * 15);
            ctx.stroke();
        }}
        
        function setupControls() {{
            // Pointer lock
            const startScreen = document.getElementById('start-screen');
            const startBtn = document.getElementById('start-btn');
            
            startBtn.addEventListener('click', () => {{
                renderer.domElement.requestPointerLock();
            }});
            
            document.addEventListener('pointerlockchange', () => {{
                isLocked = document.pointerLockElement === renderer.domElement;
                startScreen.style.display = isLocked ? 'none' : 'flex';
            }});
            
            // Mouse move
            document.addEventListener('mousemove', (event) => {{
                if (!isLocked) return;
                
                const sensitivity = 0.002;
                euler.setFromQuaternion(camera.quaternion);
                euler.y -= event.movementX * sensitivity;
                euler.x -= event.movementY * sensitivity;
                euler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, euler.x));
                camera.quaternion.setFromEuler(euler);
            }});
            
            // Keyboard - supports both AZERTY (ZQSD) and QWERTY (WASD)
            document.addEventListener('keydown', (event) => {{
                switch (event.code) {{
                    // Forward: W (QWERTY) or Z (AZERTY)
                    case 'KeyW': case 'KeyZ': moveForward = true; break;
                    // Backward: S
                    case 'KeyS': moveBackward = true; break;
                    // Left: A (QWERTY) or Q (AZERTY)
                    case 'KeyA': case 'KeyQ': moveLeft = true; break;
                    // Right: D
                    case 'KeyD': moveRight = true; break;
                    // Fly up: Space
                    case 'Space': moveUp = true; event.preventDefault(); break;
                    // Fly down: Ctrl
                    case 'ControlLeft': case 'ControlRight': moveDown = true; event.preventDefault(); break;
                    // Sprint: Shift
                    case 'ShiftLeft': case 'ShiftRight': sprint = true; break;
                    // Rotate around Z: E (right) and A (left for QWERTY) / E and Q won't work for AZERTY
                    // Using E and R for rotation to avoid conflicts
                    case 'KeyE': rotateRight = true; break;
                    case 'KeyR': rotateLeft = true; break;
                    // Reset position: T (was R, moved to T)
                    case 'KeyT': resetPosition(); break;
                    // Toggle bboxes: B
                    case 'KeyB': toggleAllBboxes(); break;
                }}
            }});
            
            document.addEventListener('keyup', (event) => {{
                switch (event.code) {{
                    case 'KeyW': case 'KeyZ': moveForward = false; break;
                    case 'KeyS': moveBackward = false; break;
                    case 'KeyA': case 'KeyQ': moveLeft = false; break;
                    case 'KeyD': moveRight = false; break;
                    case 'Space': moveUp = false; break;
                    case 'ControlLeft': case 'ControlRight': moveDown = false; break;
                    case 'ShiftLeft': case 'ShiftRight': sprint = false; break;
                    case 'KeyE': rotateRight = false; break;
                    case 'KeyR': rotateLeft = false; break;
                }}
            }});
            
            // Scroll for speed
            document.addEventListener('wheel', (event) => {{
                baseSpeed = Math.max(1, Math.min(100, baseSpeed - event.deltaY * 0.01));
                document.getElementById('speed-display').textContent = baseSpeed.toFixed(0);
            }});
            
            // Legend toggle - Point clouds
            document.querySelectorAll('.legend-item[data-class]').forEach(item => {{
                item.addEventListener('click', () => {{
                    const className = item.dataset.class;
                    if (pointClouds[className]) {{
                        pointClouds[className].visible = !pointClouds[className].visible;
                        item.classList.toggle('hidden');
                    }}
                }});
            }});
            
            // Legend toggle - Bounding boxes
            document.querySelectorAll('.legend-item[data-bbox]').forEach(item => {{
                item.addEventListener('click', () => {{
                    const bboxClass = item.dataset.bbox;
                    
                    if (bboxClass === 'all') {{
                        // Toggle all bboxes
                        const newVisible = !bboxVisible['all'];
                        bboxVisible['all'] = newVisible;
                        bboxObjects['all'].forEach(obj => obj.visible = newVisible);
                        
                        // Update all bbox legend items
                        document.querySelectorAll('.legend-item[data-bbox]').forEach(el => {{
                            if (newVisible) {{
                                el.classList.remove('hidden');
                            }} else {{
                                el.classList.add('hidden');
                            }}
                        }});
                        
                        // Reset individual visibility
                        ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine'].forEach(cls => {{
                            bboxVisible[cls] = newVisible;
                        }});
                    }} else {{
                        // Toggle specific class bboxes
                        bboxVisible[bboxClass] = !bboxVisible[bboxClass];
                        bboxObjects[bboxClass].forEach(obj => obj.visible = bboxVisible[bboxClass]);
                        item.classList.toggle('hidden');
                    }}
                }});
            }});
            
            // Resize
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        }}
        
        function resetPosition() {{
            camera.position.set(SPAWN.x, SPAWN.y, SPAWN.z);
            euler.set(0, 0, 0);
            camera.quaternion.setFromEuler(euler);
        }}
        
        function toggleAllBboxes() {{
            const newVisible = !bboxVisible['all'];
            bboxVisible['all'] = newVisible;
            bboxObjects['all'].forEach(obj => obj.visible = newVisible);
            
            // Update all bbox legend items
            document.querySelectorAll('.legend-item[data-bbox]').forEach(el => {{
                if (newVisible) {{
                    el.classList.remove('hidden');
                }} else {{
                    el.classList.add('hidden');
                }}
            }});
            
            // Reset individual visibility
            ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine'].forEach(cls => {{
                bboxVisible[cls] = newVisible;
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            const time = performance.now();
            const delta = (time - prevTime) / 1000;
            prevTime = time;
            
            if (isLocked) {{
                // Calculate speed
                const speed = baseSpeed * (sprint ? 3 : 1);
                
                // Direction vectors
                direction.z = Number(moveForward) - Number(moveBackward);
                direction.x = Number(moveRight) - Number(moveLeft);
                direction.normalize();
                
                // Get forward and right vectors from camera
                const forward = new THREE.Vector3(0, 1, 0);
                const right = new THREE.Vector3(1, 0, 0);
                forward.applyQuaternion(camera.quaternion);
                right.applyQuaternion(camera.quaternion);
                
                // Keep movement horizontal for forward/back/strafe
                forward.z = 0;
                forward.normalize();
                right.z = 0;
                right.normalize();
                
                // Apply movement
                if (moveForward || moveBackward) {{
                    camera.position.addScaledVector(forward, direction.z * speed * delta);
                }}
                if (moveLeft || moveRight) {{
                    camera.position.addScaledVector(right, direction.x * speed * delta);
                }}
                
                // Vertical movement
                if (moveUp) camera.position.z += speed * delta;
                if (moveDown) camera.position.z -= speed * delta;
                
                // Rotation around Z axis (roll)
                if (rotateLeft || rotateRight) {{
                    const rotDir = Number(rotateRight) - Number(rotateLeft);
                    euler.setFromQuaternion(camera.quaternion);
                    euler.y += rotDir * rotationSpeed * delta;
                    camera.quaternion.setFromEuler(euler);
                }}
                
                // Update HUD
                document.getElementById('pos-x').textContent = camera.position.x.toFixed(1);
                document.getElementById('pos-y').textContent = camera.position.y.toFixed(1);
                document.getElementById('pos-z').textContent = camera.position.z.toFixed(1);
            }}
            
            // Update minimap
            updateMinimap();
            
            renderer.render(scene, camera);
        }}
        
        // Start
        init();
    </script>
</body>
</html>
'''
    return html_content


def main():
    parser = argparse.ArgumentParser(description='Generate immersive 3D scene explorer')
    parser.add_argument('--voxels', type=str, required=True,
                        help='Path to consolidated voxels CSV')
    parser.add_argument('--bboxes', type=str, required=True,
                        help='Path to bounding boxes CSV')
    parser.add_argument('--scene', type=int, required=True,
                        help='Scene number (for filtering bboxes)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path')
    parser.add_argument('--max-points', type=int, default=200000,
                        help='Maximum points to include')
    parser.add_argument('--obstacles-only', action='store_true',
                        help='Show only obstacle points (no background)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'scene_{args.scene}_explorer.html'
    
    print("=" * 70)
    print(f"  🎮 IMMERSIVE 3D SCENE EXPLORER — Scene {args.scene}")
    print("=" * 70)
    
    # Load data
    xyz, labels, bboxes_df = load_data(
        args.voxels, args.bboxes, args.scene, args.max_points, args.obstacles_only
    )
    
    # Generate HTML
    print("\nGenerating HTML explorer...")
    html_content = generate_html(xyz, labels, bboxes_df, args.scene)
    
    # Save
    with open(args.output, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Saved: {args.output}")
    print(f"\n🎮 Controls (AZERTY compatible):")
    print(f"   Z/Q/S/D   - Move (or W/A/S/D)")
    print(f"   Mouse     - Look around")
    print(f"   Space     - Fly up")
    print(f"   Ctrl      - Fly down")
    print(f"   R / E     - Rotate left / right")
    print(f"   Shift     - Sprint (3x speed)")
    print(f"   Scroll    - Adjust base speed")
    print(f"   T         - Reset position")
    print(f"   B         - Toggle all bboxes")
    print(f"   Click legend items to toggle classes/bboxes")


if __name__ == '__main__':
    main()