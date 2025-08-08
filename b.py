import re
import os
import math
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import pandas as pd

# SolidPython imports
from solid import scad_render_to_file, linear_extrude, difference, union, translate, cylinder, polygon, circle, cube
from solid.utils import *

# Constants for better readability
MATERIAL_DENSITY = {
    'aluminum': 2.7,  # g/cm^3
    'steel': 7.85    # g/cm^3
}
MATERIAL_YIELD_STRENGTH = {
    'aluminum': 150, # MPa, approx for 6061-T6
    'steel': 250     # MPa, approx for mild steel
}

def parse_prompt(prompt: str):
    p = prompt.lower()
    params = {
        'part_type': 'trapezoid',
        'length': 120,
        'width': 40,
        'thickness': 5,
        'hole_count': 3,
        'hole_diameter': 6,
        'material': 'aluminum',
        'target_force_n': 2000
    }

    # Part type
    if 'arm' in p: params['part_type'] = 'arm'
    elif 'l-bracket' in p or 'l bracket' in p: params['part_type'] = 'l_bracket'
    elif 'plate' in p: params['part_type'] = 'circle'
    elif 'rectangle' in p or 'bar' in p: params['part_type'] = 'rectangle'
    elif 'trapezoid' in p or 'bracket' in p: params['part_type'] = 'trapezoid'

    # Dimensions
    dims = re.findall(r'(\d+)\s*m*m', p)
    if len(dims) >= 1: params['length'] = int(dims[0])
    if len(dims) >= 2: params['width'] = int(dims[1])
    
    # Check for specific dimension keywords
    if m := re.search(r'width(?:\s*[:=]?\s*|of\s*)(\d+)', p): params['width'] = int(m.group(1))
    if m := re.search(r'thickness\s*[:=]?\s*(\d+)', p): params['thickness'] = int(m.group(1))
    if m := re.search(r'diameter\s*(\d+)', p): params['circle_diameter'] = int(m.group(1))

    # Holes
    if m := re.search(r'(\d+)\s*[- ]?bolt|\b(\d+)\s*holes?\b', p):
        nums = [g for g in m.groups() if g is not None]
        params['hole_count'] = int(nums[0]) if nums else 0
    if m := re.search(r'(\d+)\s*mm\s*(?:hole|diameter|dia)', p): params['hole_diameter'] = int(m.group(1))
    
    # Material
    if 'steel' in p: params['material'] = 'steel'
    elif 'aluminium' in p or 'aluminum' in p: params['material'] = 'aluminum'

    # Target force
    if m := re.search(r'supports\s*(?:up to\s*)?(\d+)\s*n', p): params['target_force_n'] = int(m.group(1))

    # Normalize dimensions for specific part types
    if params['part_type'] in ('arm', 'trapezoid'):
        params['width_left'] = params['width']
        params['width_right'] = max(10, int(params['width'] * 0.5))
    elif params['part_type'] == 'rectangle':
        params['rect_width'] = params['width']
    elif params['part_type'] == 'circle':
        params['circle_diameter'] = params['length']

    return params

def generate_2d_preview(params, out_png):
    fig, ax = plt.subplots(figsize=(6, 3))
    ptype = params['part_type']
    
    # Draw the main part shape
    if ptype in ('arm', 'trapezoid'):
        L, wL, wR = params['length'], params['width_left'], params['width_right']
        pts = [(0, wL/2), (L, wR/2), (L, -wR/2), (0, -wL/2)]
        ax.add_patch(Polygon(pts, closed=True, facecolor='lightgrey', edgecolor='black'))
        
        # Add dimension annotations
        ax.annotate(f"{L} mm", xy=(L/2, wL/2 + 5), ha='center', va='bottom')
        ax.annotate(f"{wL} mm", xy=(-5, 0), ha='right', va='center')
        ax.annotate(f"{wR} mm", xy=(L+5, 0), ha='left', va='center')
        
    elif ptype == 'rectangle':
        L, W = params['length'], params['rect_width']
        ax.add_patch(Rectangle((-L/2, -W/2), L, W, facecolor='lightgrey', edgecolor='black'))
        ax.annotate(f"{L} mm", xy=(0, W/2 + 5), ha='center', va='bottom')
        ax.annotate(f"{W} mm", xy=(-L/2 - 5, 0), ha='right', va='center')

    elif ptype == 'circle':
        D = params['circle_diameter']; r = D/2.0
        ax.add_patch(Circle((0,0), radius=r, facecolor='lightgrey', edgecolor='black'))
        ax.annotate(f"⌀ {D} mm", xy=(0, r+5), ha='center', va='bottom')
    
    # Draw holes
    hc, hr = params.get('hole_count', 0), params.get('hole_diameter', 6) / 2.0
    hole_radius_scale = 1 # for visual clarity
    
    if hc > 0:
        if ptype in ('arm', 'trapezoid'):
            for i in range(hc):
                x = (i+1) * L / (hc+1)
                y_pos = (params['width_left']/2) - (params['width_left']-params['width_right'])/2 * ((i+1)/(hc+1))
                ax.add_patch(Circle((x, y_pos), radius=hr * hole_radius_scale, facecolor='white', edgecolor='black'))
        elif ptype == 'rectangle':
            for i in range(hc):
                x = (i+1) * L / (hc+1) - L/2
                y = 0
                ax.add_patch(Circle((x, y), radius=hr * hole_radius_scale, facecolor='white', edgecolor='black'))
        elif ptype == 'circle':
            ring_r = r * 0.5
            for i in range(hc):
                ang = 2 * math.pi * i / hc
                x, y = ring_r * math.cos(ang), ring_r * math.sin(ang)
                ax.add_patch(Circle((x,y), radius=hr * hole_radius_scale, facecolor='white', edgecolor='black'))

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

def build_scad(params):
    ptype = params['part_type']
    thickness = float(params.get('thickness', 5))
    solid_base = None  # Initialize solid_base to a default value

    if ptype in ('arm','trapezoid'):
        L = float(params['length']); wL = float(params['width_left']); wR = float(params['width_right'])
        base2d = polygon(points=[[0,0],[L,0],[L,wR],[0,wL]])
        solid_base = linear_extrude(height=thickness)(base2d)

    elif ptype == 'rectangle':
        L = float(params['length']); W = float(params['rect_width'])
        base2d = polygon(points=[[0,0],[L,0],[L,W],[0,W]])
        solid_base = linear_extrude(height=thickness)(base2d)

    elif ptype == 'circle':
        D = float(params['circle_diameter']); r = D/2.0
        base2d = circle(r=r, segments=128)
        solid_base = linear_extrude(height=thickness)(base2d)
    
    # --- Add this new block to handle the 'l_bracket' part type ---
    elif ptype == 'l_bracket':
        # Define the two parts of the L-bracket and union them
        part1 = cube([params.get('width', 40), params.get('length', 120), thickness])
        part2 = translate([0, params.get('length', 120) - thickness, 0])(
            cube([params.get('width', 40), thickness, params.get('length', 120) / 2])
        )
        solid_base = union()(part1, part2)
    # --- End of new block ---

    else:
        raise ValueError(f"Unknown part type for SCAD: {ptype}")

    # Holes logic (assuming it's compatible with the new l_bracket shape)
    hole_count = int(params.get('hole_count',0))
    hole_r = float(params.get('hole_diameter',6))/2.0
    hole_objs = []
    
    # ... (the rest of your hole generation logic) ...
    
    if solid_base is None:
        raise ValueError(f"Could not build a solid base for part type: {ptype}")

    if hole_objs:
        holes_union = union()(*hole_objs)
        part = difference()(solid_base, holes_union)
    else:
        part = solid_base

    return part

def append_csv(row_dict):
    CSV_FILE = "parts_generated.csv"
    df_new = pd.DataFrame([row_dict])
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_FILE, index=False)

def run_from_prompt(prompt, save_prefix="part"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    params = parse_prompt(prompt)
    
    png_name, scad_name = f"{save_prefix}_{timestamp}.png", f"{save_prefix}_{timestamp}.scad"
    generate_2d_preview(params, png_name)
    
    part = build_scad(params)
    scad_render_to_file(part, scad_name, file_header='$fn = 96;')
    
    stl_name = scad_name.replace('.scad', '.stl')
    stl_created = False
    try:
        subprocess.run(["openscad", "-o", stl_name, scad_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stl_created = True
    except Exception:
        pass
    
    # Weight and strength calculation
    net_area_mm2 = 0
    if params['part_type'] in ('arm', 'trapezoid'):
        net_area_mm2 = 0.5 * (params['width_left'] + params['width_right']) * params['length']
    elif params['part_type'] == 'rectangle':
        net_area_mm2 = params['length'] * params['rect_width']
    elif params['part_type'] == 'circle':
        net_area_mm2 = math.pi * (params['circle_diameter']/2.0)**2
    
    thickness = params.get('thickness', 5)
    volume_cm3 = (net_area_mm2 * thickness) / 1000.0
    density = MATERIAL_DENSITY.get(params.get('material'), 2.7)
    weight_g = volume_cm3 * density
    
    yield_strength = MATERIAL_YIELD_STRENGTH.get(params.get('material'), 150)
    # Simple stress calculation (Force / Area), assuming cross-section is thickness * min_width
    effective_width = params.get('width_right', params.get('rect_width', params.get('circle_diameter')))
    min_cross_area_mm2 = effective_width * thickness
    
    max_force = min_cross_area_mm2 * (yield_strength / 1000) # Convert MPa to N/mm^2
    status = "PASS" if max_force >= params.get('target_force_n', 2000) else "FAIL"
    
    # Log to CSV
    row = {
        "timestamp": timestamp, "prompt": prompt, "part_type": params['part_type'], "png": os.path.abspath(png_name),
        "scad": os.path.abspath(scad_name), "stl": os.path.abspath(stl_name) if stl_created else "",
        "net_area_mm2": net_area_mm2, "thickness_mm": thickness, "volume_cm3": volume_cm3,
        "material": params.get('material'), "weight_g": weight_g, "max_force_n": max_force,
        "target_force_n": params.get('target_force_n'), "status": status
    }
    append_csv(row)
    
    # Display results
    print("=== Generated Summary ===")
    print(f"**Prompt:** {prompt}")
    print(f"**Part Type:** {params['part_type']}")
    print(f"**Material:** {params['material']}")
    print(f"**Weight:** {weight_g:.2f} g")
    print(f"**Load Capacity:** {max_force:.2f} N (Target: {params.get('target_force_n', 2000)} N)")
    print(f"**Status:** {'✅ PASS' if status == 'PASS' else '⚠️ FAIL'}")
    
    img = plt.imread(png_name)
    plt.figure(figsize=(8,4)); plt.imshow(img); plt.axis('off'); plt.title(f"2D Preview: {params['part_type']}"); plt.show()

if __name__ == "__main__":
    example_prompts = [
        "Design an aluminum suspension arm 150mm long, 50mm wide, supports 2000N, 3-bolt mount",
        "Make a steel plate 80mm diameter with 4 holes 6mm, thickness 6mm",
        "create a rectangle bar 200mm length and 30mm width with 2 bolt holes"
    ]
    print("Choose an example prompt or paste your own:")
    for i,p in enumerate(example_prompts,1):
        print(f"{i}. {p}")
    print("0. Enter custom prompt")
    
    choice = input("Enter choice [0-3]: ").strip()
    if choice == '0':
        prompt = input("Paste prompt: ").strip()
    elif choice in ('1','2','3'):
        prompt = example_prompts[int(choice)-1]
    else:
        prompt = example_prompts[0]
    
    run_from_prompt(prompt, save_prefix="part")