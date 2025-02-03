#!/usr/bin/env python3
"""
4D Polytopes Visualization with Persistent Spandex Surface Coloring and Manual 3D Rotation
--------------------------------------------------------------------------------------------
This script visualizes 4D polytopes with a “spandex” effect: the exterior surfaces (flat faces)
of the projected 3D object are computed by grouping the convex-hull’s triangulated facets into
flat faces (based on their corner vertices). Each facet is assigned a persistent random color that
remains stable as long as that facet is visible. Each vertex is labeled with its index (if enabled).
The 3D view is manually rotatable by click-and-drag (the view is not automatically reset).
 
Hotkeys:
    1-6         : Select Polytope
    R           : Toggle Projection Mode (Perspective vs. Orthographic)
    + / -       : Increase/Decrease Rotation Speed
    T           : Cycle Through Visual Themes
    M           : Toggle on‑screen Menu
    P           : Toggle 4D Plane Overlays
    [ / ]       : Decrease/Increase Rotation‑Plane Opacity
    H           : Toggle Filled Exterior (Hull) Surfaces On/Off
    O           : Increase Surface Opacity
    L           : Decrease Surface Opacity
    V           : Toggle Vertex Labels
    Scroll      : Zoom In/Out
 
Requirements:
    Python 3.10+
    matplotlib, scipy
    (CuPy optional; otherwise falls back to NumPy)
"""

import itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm  # for colormap

# For drawing filled surfaces.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# For convex hull computations.
from scipy.spatial import ConvexHull

# Try to import CuPy.
try:
    import cupy as cp
    xp = cp
    gpu_enabled = True
except ImportError:
    xp = np
    gpu_enabled = False

# -------------------------------
# Global Variables and Toggles
# -------------------------------
current_polytope = None
polytope_data = {}  # maps polytope name -> {'vertices': ..., 'edges': ...}
rotation_speed = 0.02  # radians per frame
projection_perspective = True  # True: perspective; False: orthographic
show_menu = True

# Toggle for drawing 4D rotation-plane overlays.
show_planes = True
plane_opacity = 0.5

# Toggle for filled exterior surfaces ("spandex").
show_surfaces = False   # OFF by default
surface_opacity = 0.5

# Toggle for vertex labels.
show_labels = True      # ON by default

# Thread pool.
executor = ThreadPoolExecutor(max_workers=4)

zoom_scale = 1.0
frame_count = 0

# Dictionary to store persistent facet colors.
persistent_facet_colors = {}

# -------------------------------
# Theme Definitions
# -------------------------------
themes = [
    {
        "name": "Classic",
        "bg_color": "whitesmoke",
        "line_color": "blue",
        "point_color": "red",
        "menu_color": "black",
        "surface_color": "lightblue",  # placeholder
    },
    {
        "name": "Dark",
        "bg_color": "black",
        "line_color": "cyan",
        "point_color": "yellow",
        "menu_color": "white",
        "surface_color": "gray",
    },
    {
        "name": "Forest",
        "bg_color": "#1b3a1b",
        "line_color": "#a3e635",
        "point_color": "#74c365",
        "menu_color": "white",
        "surface_color": "seagreen",
    },
    {
        "name": "Neon",
        "bg_color": "#000",
        "line_color": "#39ff14",
        "point_color": "#ff073a",
        "menu_color": "#f0f",
        "surface_color": "hotpink",
    },
    {
        "name": "Midnight",
        "bg_color": "#0d0d2b",
        "line_color": "#66ccff",
        "point_color": "#ffcc00",
        "menu_color": "#fff",
        "surface_color": "navy",
    }
]
current_theme_index = 0

def cycle_theme():
    global current_theme_index
    current_theme_index = (current_theme_index + 1) % len(themes)
    theme = themes[current_theme_index]
    print(f"Switched to theme: {theme['name']}")

# -------------------------------
# Helper Functions for Facet Grouping
# -------------------------------
def compute_face_key(tri, pts, tol_normal=0.01, tol_d=0.01):
    A, B, C = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    v1 = B - A
    v2 = C - A
    normal = np.cross(v1, v2)
    norm_val = np.linalg.norm(normal)
    if norm_val == 0:
        normal = np.array([0, 0, 1])
    else:
        normal = normal / norm_val
    for comp in normal:
        if abs(comp) > 1e-6:
            if comp < 0:
                normal = -normal
            break
    d = np.dot(normal, A)
    key = (tuple(np.round(normal, 2)), round(d, 2))
    return key

def order_polygon_vertices(verts, normal):
    centroid = np.mean(verts, axis=0)
    if abs(normal[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])
    v = np.cross(normal, ref)
    v = v / np.linalg.norm(v)
    w = np.cross(normal, v)
    coords = []
    for p in verts:
        vec = p - centroid
        coords.append([np.dot(vec, v), np.dot(vec, w)])
    coords = np.array(coords)
    angles = np.arctan2(coords[:,1], coords[:,0])
    order = np.argsort(angles)
    return verts[order]

def match_facet_key(new_key, persistent_keys, tol=0.01):
    new_normal, new_d = new_key
    for key in persistent_keys:
        old_normal, old_d = key
        if np.allclose(new_normal, old_normal, atol=tol) and abs(new_d - old_d) < tol:
            return key
    return None

def get_facets(projected):
    hull = ConvexHull(projected, qhull_options='QJ')
    groups = {}
    for tri in hull.simplices:
        key = compute_face_key(tri, projected)
        matched = match_facet_key(key, groups.keys(), tol=0.01)
        if matched is not None:
            key = matched
        groups.setdefault(key, set()).update(tri)
    facets = []
    for key, idx_set in groups.items():
        idx_list = list(idx_set)
        pts = projected[idx_list]
        ordered_pts = order_polygon_vertices(pts, np.array(key[0]))
        ordered_indices = []
        for p in ordered_pts:
            for idx in idx_list:
                if np.allclose(projected[idx], p, atol=1e-3):
                    ordered_indices.append(idx)
                    break
        facets.append((key, ordered_indices, ordered_pts))
    return facets

# -------------------------------
# Plane Overlay Definitions
# -------------------------------
plane_coord = {
    "xy": (0, 1),
    "xz": (0, 2),
    "xw": (0, 3),
    "yz": (1, 2),
    "yw": (1, 3),
    "zw": (2, 3),
}
plane_colors = {
    "xy": "magenta",
    "xz": "cyan",
    "xw": "yellow",
    "yz": "lime",
    "yw": "orange",
    "zw": "pink",
}

def draw_rotation_planes(R_cpu):
    if not show_planes:
        return
    npoints = 7
    limit = 2.5
    for plane, coords in plane_coord.items():
        grid = np.linspace(-limit, limit, npoints)
        U, V = np.meshgrid(grid, grid)
        points_4d = np.zeros((npoints, npoints, 4), dtype=np.float32)
        idx1, idx2 = coords
        points_4d[:, :, idx1] = U
        points_4d[:, :, idx2] = V
        points_flat = points_4d.reshape(-1, 4)
        rotated = points_flat @ R_cpu.T
        projected = project_4d_to_3d(rotated)
        projected = np.array(projected)
        projected_mesh = projected.reshape(npoints, npoints, 3)
        X = projected_mesh[:, :, 0]
        Y = projected_mesh[:, :, 1]
        Z = projected_mesh[:, :, 2]
        ax.plot_surface(X, Y, Z, color=plane_colors[plane], alpha=plane_opacity,
                        rstride=1, cstride=1, linewidth=0, antialiased=True)

# -------------------------------
# Polytope Generation Functions
# -------------------------------
def generate_tesseract():
    vertices = xp.array(list(itertools.product([-1,1], repeat=4)), dtype=xp.float32)
    edges = []
    n = vertices.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if xp.sum(xp.abs(vertices[i]-vertices[j])) == 2:
                edges.append((i,j))
    return {'vertices': vertices, 'edges': edges}

def generate_pentachoron():
    a = 1 / math.sqrt(10)
    vertex_list = [
        [2*a,  2*a,  2*a, -3*a],
        [2*a, -2*a, -2*a, -3*a],
        [-2*a,  2*a, -2*a, -3*a],
        [-2*a, -2*a,  2*a, -3*a],
        [0.0,   0.0,  0.0,  4*a],
    ]
    vertex_list = [[float(val) for val in row] for row in vertex_list]
    vertices = xp.array(vertex_list, dtype=xp.float32)
    edges = []
    n = vertices.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i,j))
    return {'vertices': vertices, 'edges': edges}

def generate_16cell():
    vertices = []
    for i in range(4):
        for sign in [-1,1]:
            v = [0,0,0,0]
            v[i] = sign
            vertices.append(v)
    vertices = xp.array(vertices, dtype=xp.float32)
    edges = []
    n = vertices.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if xp.abs(xp.dot(vertices[i],vertices[j])) < 1e-5:
                edges.append((i,j))
    return {'vertices': vertices, 'edges': edges}

def generate_24cell():
    vertices = []
    for perm in set(itertools.permutations([1,1,0,0])):
        for signs in itertools.product(*[(-1,1) if x!=0 else (0,) for x in perm]):
            vertex = [s for s in signs]
            vertices.append(vertex)
    vertices = xp.array(list({tuple(v) for v in vertices}), dtype=xp.float32)
    edges = []
    n = vertices.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if xp.isclose(xp.linalg.norm(vertices[i]-vertices[j]), 2.0, atol=1e-3):
                edges.append((i,j))
    return {'vertices': vertices, 'edges': edges}

def generate_120cell():
    base = generate_tesseract()['vertices']
    theta = math.pi/4
    rot = xp.array([
        [math.cos(theta),0,0,-math.sin(theta)],
        [0,1,0,0],
        [0,0,1,0],
        [math.sin(theta),0,0,math.cos(theta)]
    ], dtype=xp.float32)
    rotated = xp.dot(base, rot.T)
    vertices = xp.concatenate([base, rotated], axis=0)
    edges = []
    n = vertices.shape[0]
    def compute_edges(start, end):
        local_edges = []
        for i in range(start, end):
            for j in range(i+1, n):
                if xp.linalg.norm(vertices[i]-vertices[j]) < 2.1:
                    local_edges.append((i,j))
        return local_edges
    chunk = n//4
    futures = []
    for k in range(4):
        start = k*chunk
        end = n if k==3 else (k+1)*chunk
        futures.append(executor.submit(compute_edges, start, end))
    for fut in futures:
        edges.extend(fut.result())
    return {'vertices': vertices, 'edges': edges}

def generate_600cell():
    base = generate_16cell()['vertices']
    extra = []
    n = base.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            mid = (base[i] + base[j]) / 2
            extra.append(mid.tolist())
    vertices = xp.concatenate([base, xp.array(extra, dtype=xp.float32)], axis=0)
    vertices = xp.array(list({tuple(v.tolist()) for v in vertices}), dtype=xp.float32)
    edges = []
    n = vertices.shape[0]
    def compute_edges(start, end):
        local_edges = []
        for i in range(start, end):
            for j in range(i+1, n):
                if xp.linalg.norm(vertices[i]-vertices[j]) < 1.1:
                    local_edges.append((i,j))
        return local_edges
    chunk = n//4 if n>=4 else n
    futures = []
    for k in range(4):
        start = k*chunk
        end = n if k==3 else (k+1)*chunk
        futures.append(executor.submit(compute_edges, start, end))
    for fut in futures:
        edges.extend(fut.result())
    return {'vertices': vertices, 'edges': edges}

POLYTOPE_GENERATORS = {
    '1': ("Tesseract", generate_tesseract),
    '2': ("Pentachoron", generate_pentachoron),
    '3': ("16-Cell", generate_16cell),
    '4': ("24-Cell", generate_24cell),
    '5': ("120-Cell", generate_120cell),
    '6': ("600-Cell", generate_600cell)
}

# -------------------------------
# 4D Rotation and Projection Functions
# -------------------------------
def rotation_matrix_4d(angles):
    R = xp.eye(4, dtype=xp.float32)
    def plane_rotation(i, j, theta):
        M = xp.eye(4, dtype=xp.float32)
        M[i,i] = math.cos(theta)
        M[j,j] = math.cos(theta)
        M[i,j] = -math.sin(theta)
        M[j,i] = math.sin(theta)
        return M
    for (plane, (i,j)) in zip(['xy','xz','xw','yz','yw','zw'],
                               [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        theta = angles.get(plane, 0)
        R = xp.dot(plane_rotation(i,j,theta), R)
    return R

def project_4d_to_3d(vertices):
    d = 4.0
    if projection_perspective:
        w = vertices[:,3:4]
        factor = d / (d - w + 1e-5)
        projected = vertices[:,:3] * factor
    else:
        projected = vertices[:,:3]
    return projected

# -------------------------------
# Matplotlib Animation Setup
# -------------------------------
fig = plt.figure("4D Polytopes - Spandex Surfaces")
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])

def set_axis_limits():
    lim = 4 * zoom_scale
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

set_axis_limits()
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

lines_artists = []
points_artist = None
rotation_angles = {'xy':0, 'xz':0, 'xw':0, 'yz':0, 'yw':0, 'zw':0}

# -------------------------------
# Update Function (Animation)
# -------------------------------
def update(frame):
    global current_polytope, lines_artists, points_artist, rotation_angles, frame_count, persistent_facet_colors
    frame_count += 1

    # Instead of clearing the entire axes, remove only the drawn artists.
    for coll in ax.collections[:]:
        coll.remove()
    for line in ax.lines[:]:
        line.remove()
    for txt in ax.texts[:]:
        txt.remove()

    set_axis_limits()

    current_theme = themes[current_theme_index]
    # Do not reset the view; allow manual rotation.
    
    ax.xaxis.pane.set_facecolor(current_theme["bg_color"])
    ax.yaxis.pane.set_facecolor(current_theme["bg_color"])
    ax.zaxis.pane.set_facecolor(current_theme["bg_color"])
    ax.set_facecolor(current_theme["bg_color"])

    for key in rotation_angles:
        rotation_angles[key] += rotation_speed
    R = rotation_matrix_4d(rotation_angles)
    R_cpu = cp.asnumpy(R) if gpu_enabled else R
    draw_rotation_planes(R_cpu)

    if current_polytope is None:
        return []

    vertices = current_polytope['vertices']
    rotated = xp.dot(vertices, R.T)
    projected = project_4d_to_3d(rotated)
    if gpu_enabled:
        projected = cp.asnumpy(projected)
    else:
        projected = np.array(projected)

    xs, ys, zs = projected[:,0], projected[:,1], projected[:,2]
    points_artist = ax.scatter(xs, ys, zs, c=current_theme["point_color"], s=20)
    artists = [points_artist]
    # Label vertices if enabled.
    if show_labels:
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), color=current_theme["line_color"], fontsize=8)
    for (i, j) in current_polytope['edges']:
        x_line = [projected[i,0], projected[j,0]]
        y_line = [projected[i,1], projected[j,1]]
        z_line = [projected[i,2], projected[j,2]]
        line, = ax.plot(x_line, y_line, z_line, c=current_theme["line_color"], lw=0.5)
        artists.append(line)

    if show_surfaces:
        try:
            facets = get_facets(projected)  # list of (key, ordered_indices, polygon)
            current_keys = set([key for key, idxs, poly in facets])
            for k in list(persistent_facet_colors.keys()):
                if k not in current_keys:
                    del persistent_facet_colors[k]
            for key, idxs, poly in facets:
                if key not in persistent_facet_colors:
                    persistent_facet_colors[key] = cm.hsv(np.random.rand())
            facet_colors = [persistent_facet_colors[key] for key, idxs, poly in facets]
            polys = [poly for key, idxs, poly in facets]
            poly3d = Poly3DCollection(polys, facecolors=facet_colors,
                                      edgecolors='none', linewidths=0, alpha=surface_opacity)
            ax.add_collection3d(poly3d)
        except Exception as e:
            print("Error computing facets for surfaces:", e)

    menu_str = (
        f"Theme: {current_theme['name']}\n"
        "Hotkeys:\n"
        "1-6 : Select Polytope\n"
        "R   : Toggle Projection Mode\n"
        "+/- : Increase/Decrease Rotation Speed\n"
        "T   : Cycle Themes\n"
        "M   : Toggle Menu\n"
        "P   : Toggle Plane Overlays\n"
        "[/] : Decrease/Increase Plane Opacity\n"
        "H   : Toggle Filled Surfaces\n"
        "O   : Increase Surface Opacity\n"
        "L   : Decrease Surface Opacity\n"
        "V   : Toggle Vertex Labels\n"
        "Scroll : Zoom In/Out"
    )
    if show_menu:
        ax.text2D(0.02, 0.90, menu_str, transform=ax.transAxes,
                  color=current_theme["menu_color"],
                  bbox=dict(facecolor='w', alpha=0.7), fontsize=9)
    return artists

# -------------------------------
# Event Handlers
# -------------------------------
def on_key(event):
    global current_polytope, polytope_data, rotation_speed, projection_perspective, show_menu, plane_opacity, show_surfaces, surface_opacity, show_planes, show_labels
    key = event.key
    if key in POLYTOPE_GENERATORS:
        name, gen_func = POLYTOPE_GENERATORS[key]
        if name not in polytope_data:
            print(f"Generating polytope: {name} ...")
            polytope_data[name] = gen_func()
        current_polytope = polytope_data[name]
        plt.title(f"4D Polytope: {name}")
    elif key.lower() == 'r':
        projection_perspective = not projection_perspective
        mode = "Perspective" if projection_perspective else "Orthographic"
        print(f"Projection mode toggled: {mode}")
    elif key == '+':
        rotation_speed += 0.005
        print(f"Rotation speed increased: {rotation_speed:.3f}")
    elif key in ('-', '_'):
        rotation_speed = max(0.001, rotation_speed - 0.005)
        print(f"Rotation speed decreased: {rotation_speed:.3f}")
    elif key.lower() == 'm':
        global show_menu
        show_menu = not show_menu
    elif key.lower() == 't':
        cycle_theme()
    elif key.lower() == 'p':
        show_planes = not show_planes
        print(f"Plane overlays toggled: {'On' if show_planes else 'Off'}")
    elif key == ']':
        plane_opacity = min(1.0, plane_opacity + 0.1)
        print(f"Plane opacity increased: {plane_opacity:.2f}")
    elif key == '[':
        plane_opacity = max(0.0, plane_opacity - 0.1)
        print(f"Plane opacity decreased: {plane_opacity:.2f}")
    elif key.lower() == 'h':
        show_surfaces = not show_surfaces
        print(f"Filled surfaces toggled: {'On' if show_surfaces else 'Off'}")
    elif key.lower() == 'o':
        surface_opacity = min(1.0, surface_opacity + 0.1)
        print(f"Surface opacity increased: {surface_opacity:.2f}")
    elif key.lower() == 'l':
        surface_opacity = max(0.0, surface_opacity - 0.1)
        print(f"Surface opacity decreased: {surface_opacity:.2f}")
    elif key.lower() == 'v':
        show_labels = not show_labels
        print(f"Vertex labels toggled: {'On' if show_labels else 'Off'}")
    else:
        return

def on_scroll(event):
    global zoom_scale
    if hasattr(event, 'step'):
        step = event.step
    else:
        step = 1 if event.button=='up' else -1
    factor = 0.9 if step>0 else 1.1
    zoom_scale *= factor
    print(f"Zoom scale updated: {zoom_scale:.3f}")
    set_axis_limits()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('scroll_event', on_scroll)

# -------------------------------
# Initialization and Start
# -------------------------------
default_name, default_gen = POLYTOPE_GENERATORS['1']
polytope_data[default_name] = default_gen()
current_polytope = polytope_data[default_name]
plt.title(f"4D Polytope: {default_name}")

ani = FuncAnimation(fig, update, frames=None, interval=30, blit=False, cache_frame_data=False)
plt.show()

executor.shutdown(wait=False)
