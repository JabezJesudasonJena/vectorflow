import re
import os
import math
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import pandas as pd
from solid import scad_render_to_file, linear_extrude, difference, union, translate, cylinder, polygon, circle, cube
from solid import *
from solid.utils import *

def parse_prompt(prompt: str):
    p = prompt.lower()
    params = {}
    if re.search(r'\barm\b', p):
        params['part_type'] = 'arm'
    elif re.search(r'\bl-?bracket\b|\bl bracket\b', p):
        params['part_type'] = 'l_bracket'
    elif re.search(r'\bplate\b', p):
        params['part_type'] = 'circle'
    elif re.search(r'\brectangl|bar\b', p):
        params['part_type'] = 'rectangle'
    elif re.search(r'\btrapezoid\b|\bbracket\b', p):
        params['part_type'] = 'trapezoid'
    else:
        params['part_type'] = 'trapezoid'
    m = re.search(r'(\d+)\s*mm', p)
    if m:
        params['length'] = int(m.group(1))
    else:
        m2 = re.search(r'(\d+)\s*(?:long|length|l\b)', p)
        params['length'] = int(m2.group(1)) if m2 else 120
    m = re.search(r'width(?:\s*[:=]?\s*|of\s*)(\d+)\s*mm', p)
    if m:
        params['width'] = int(m.group(1))
    else:
        mm_all = re.findall(r'(\d+)\s*mm', p)
        if len(mm_all) >= 2:
            params['width'] = int(mm_all[1])
        else:
            params['width'] = 40
    m = re.search(r'thickness\s*[:=]?\s*(\d+)\s*mm', p)
    if m:
        params['thickness'] = int(m.group(1))
    else:
        params['thickness'] = 5
    m = re.search(r'(\d+)\s*[- ]?bolt|\b(\d+)\s*holes?\b', p)
    if m:
        nums = [g for g in m.groups() if g is not None]
        params['hole_count'] = int(nums[0]) if nums else 0
    else:
        m2 = re.search(r'(\d+)-bolt', p)
        params['hole_count'] = int(m2.group(1)) if m2 else 3
    m = re.search(r'(\d+)\s*mm\s*(?:hole|diameter|dia)', p)
    params['hole_diameter'] = int(m.group(1)) if m else 6
    if 'steel' in p:
        params['material'] = 'steel'
    elif 'aluminium' in p or 'aluminum' in p:
        params['material'] = 'aluminum'
    else:
        params['material'] = 'aluminum'
    m = re.search(r'supports\s*(?:up to\s*)?(\d+)\s*n', p)
    params['target_force_n'] = int(m.group(1)) if m else 2000
    if params['part_type'] == 'arm' or params['part_type'] == 'trapezoid':
        params['length'] = params.get('length', 150)
        params['width_left'] = params.get('width', 50)
        params['width_right'] = max(10, int(params['width_left'] * 0.5))
    elif params['part_type'] == 'rectangle':
        params['rect_width'] = params.get('width', 40)
        params['rect_height'] = params.get('thickness', 5)
    elif params['part_type'] == 'circle':
        params['circle_diameter'] = params.get('length', 80)
    for k,v in list(params.items()):
        if isinstance(v, str) and v.isdigit():
            params[k] = int(v)
    return params

def generate_2d_preview(params, out_png):
    fig, ax = plt.subplots(figsize=(6,3))
    ptype = params['part_type']
    if ptype in ('arm','trapezoid'):
        L = params['length']; wL = params['width_left']; wR = params['width_right']
        pts = [(0, 0), (L, 0), (L, wR), (0, wL)]
        ax.add_patch(Polygon(pts, closed=True, facecolor='lightgrey', edgecolor='black'))
        hc = params.get('hole_count', 0); hr = params.get('hole_diameter',6)/2.0
        for i in range(hc):
            x = (i+1) * L / (hc+1)
            y = (wL + wR) / 4.0
            ax.add_patch(Circle((x, y), radius=hr, facecolor='white', edgecolor='black'))
        ax.set_xlim(-10, L+10); ax.set_ylim(-10, max(wL,wR)+10)
    elif ptype == 'rectangle':
        L = params['length']; W = params['rect_width']
        ax.add_patch(Rectangle((0,0), L, W, facecolor='lightgrey', edgecolor='black'))
        hc = params.get('hole_count', 0); hr = params.get('hole_diameter',6)/2.0
        for i in range(hc):
            x = (i+1) * L / (hc+1); y = W/2.0
            ax.add_patch(Circle((x,y), radius=hr, facecolor='white', edgecolor='black'))
        ax.set_xlim(-10, L+10); ax.set_ylim(-10, W+10)
    elif ptype == 'circle':
        D = params['circle_diameter']; r = D/2.0
        ax.add_patch(Circle((0,0), radius=r, facecolor='lightgrey', edgecolor='black'))
        hc = params.get('hole_count', 0); hr = params.get('hole_diameter',6)/2.0
        if hc>0:
            ring_r = r*0.5
            for i in range(hc):
                ang = 2*math.pi*i/hc
                x = ring_r*math.cos(ang); y = ring_r*math.sin(ang)
                ax.add_patch(Circle((x,y), radius=hr, facecolor='white', edgecolor='black'))
        ax.set_xlim(-r-10, r+10); ax.set_ylim(-r-10, r+10)
    ax.set_aspect('equal'); ax.axis('off')
    plt.savefig(out_png, dpi=200, bbox_inches='tight'); plt.close(fig)

def build_scad(params):
    ptype = params['part_type']
    thickness = float(params.get('thickness', 5))
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
    else:
        raise ValueError("Unknown part type for SCAD")
    hole_count = int(params.get('hole_count',0))
    hole_r = float(params.get('hole_diameter',6))/2.0
    hole_objs = []
    if hole_count > 0:
        if ptype in ('arm','trapezoid'):
            for i in range(hole_count):
                x = (i+1) * L / (hole_count+1)
                y = (wL + wR) / 4.0
                hole_objs.append(translate([x, y, -1])(cylinder(r=hole_r, h=thickness + 2)))
        elif ptype == 'rectangle':
            for i in range(hole_count):
                x = (i+1) * L / (hole_count+1)
                y = W/2.0
                hole_objs.append(translate([x, y, -1])(cylinder(r=hole_r, h=thickness + 2)))
        elif ptype == 'circle':
            ring_r = r*0.5
            for i in range(hole_count):
                ang = 2*math.pi*i/hole_count
                x = ring_r*math.cos(ang); y = ring_r*math.sin(ang)
                hole_objs.append(translate([x, y, -1])(cylinder(r=hole_r, h=thickness + 2)))
    if hole_objs:
        holes_union = union()(*hole_objs)
        part = difference()(solid_base, holes_union)
    else:
        part = solid_base
    return part

CSV_FILE = "parts_generated.csv"
def append_csv(row_dict):
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
    if params['part_type'] in ('arm','trapezoid'):
        params.setdefault('length', 150)
        params.setdefault('width_left', 50)
        params.setdefault('width_right', int(params.get('width_left',50)*0.5))
    elif params['part_type'] == 'rectangle':
        params.setdefault('length', 120); params.setdefault('rect_width', 40)
    elif params['part_type'] == 'circle':
        params.setdefault('circle_diameter', 80)
    png_name = f"{save_prefix}_{timestamp}.png"
    scad_name = f"{save_prefix}_{timestamp}.scad"
    generate_2d_preview(params, png_name)
    part = build_scad(params)
    scad_render_to_file(part, scad_name, file_header='$fn = 96;')
    stl_name = scad_name.replace('.scad', '.stl')
    try:
        subprocess.run(["openscad", "-o", stl_name, scad_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stl_created = True
    except Exception:
        stl_created = False
    if params['part_type'] in ('arm','trapezoid'):
        net_area_mm2 = 0.5 * (params['width_left'] + params['width_right']) * params['length']
    elif params['part_type'] == 'rectangle':
        net_area_mm2 = params['length'] * params['rect_width']
    else:
        net_area_mm2 = math.pi * (params['circle_diameter']/2.0)**2
    thickness = params.get('thickness', 5)
    volume_cm3 = (net_area_mm2 * thickness) / 1000.0
    density = 2.7 if params.get('material','aluminum')=='aluminum' else 7.85
    weight_g = volume_cm3 * density
    yield_strength = 150 if params.get('material','aluminum')=='aluminum' else 250
    max_force = yield_strength * net_area_mm2
    status = "PASS" if max_force >= params.get('target_force_n', 2000) else "FAIL"
    row = {
        "timestamp": timestamp,
        "prompt": prompt,
        "part_type": params['part_type'],
        "png": os.path.abspath(png_name),
        "scad": os.path.abspath(scad_name),
        "stl": os.path.abspath(stl_name) if stl_created else "",
        "net_area_mm2": net_area_mm2,
        "thickness_mm": thickness,
        "volume_cm3": volume_cm3,
        "material": params.get('material'),
        "weight_g": weight_g,
        "max_force_n": max_force,
        "target_force_n": params.get('target_force_n'),
        "status": status
    }
    append_csv(row)
    print("=== Generated summary ===")
    for k,v in row.items():
        print(f"{k}: {v}")
    img = plt.imread(png_name)
    plt.figure(figsize=(8,4)); plt.imshow(img); plt.axis('off'); plt.show()

if __name__ == "__main__":
    example_prompts = [
        "Design an aluminum suspension arm 150mm long, 50mm wide, supports 2000N, 3-bolt mount",
        "Make a steel plate 80mm diameter with 4 holes 6mm, thickness 6mm"
    ]
    print("Choose an example prompt or paste your own:")
    for i,p in enumerate(example_prompts,1):
        print(f"{i}. {p}")
    print("0. Enter custom prompt")
    choice = input("Enter choice [0-2]: ").strip()
    if choice == '0':
        prompt = input("Paste prompt: ").strip()
    elif choice in ('1','2'):
        prompt = example_prompts[int(choice)-1]
    else:
        prompt = example_prompts[0]
    run_from_prompt(prompt, save_prefix="part")