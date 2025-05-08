import numpy as np
import networkx as nx
from scipy.interpolate import splev, splrep
from config import PROJECT_ROOT  # Access project-wide configs if needed

def create_coarse_grid(image_shape, grid_size=100):
    h, w = image_shape[:2]
    x_grid = np.linspace(0, w-1, grid_size, dtype=int)
    y_grid = np.linspace(0, h-1, grid_size, dtype=int)
    return x_grid, y_grid

def mark_crater_cells(graph, crater_boxes, x_grid, y_grid, image_shape, buffer_pixels=15):
    buffered_boxes = []
    for (x1, y1, x2, y2) in crater_boxes:
        buffered_boxes.append((
            max(0, x1 - buffer_pixels),
            max(0, y1 - buffer_pixels),
            min(image_shape[1]-1, x2 + buffer_pixels),
            min(image_shape[0]-1, y2 + buffer_pixels)
        ))
    
    removed_nodes = 0
    for box in buffered_boxes:
        x1, y1, x2, y2 = box
        xi_min = np.searchsorted(x_grid, x1)
        xi_max = np.searchsorted(x_grid, x2)
        yi_min = np.searchsorted(y_grid, y1)
        yi_max = np.searchsorted(y_grid, y2)
        
        for xi in range(max(0, xi_min), min(len(x_grid), xi_max)):
            for yi in range(max(0, yi_min), min(len(y_grid), yi_max)):
                if (xi, yi) in graph:
                    graph.remove_node((xi, yi))
                    removed_nodes += 1
    
    print(f"Removed {removed_nodes} nodes from graph")

def point_to_line_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return numerator / denominator if denominator != 0 else 0

def douglas_peucker(points, epsilon=5):
    if len(points) < 3:
        return points
    
    dmax = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_to_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    
    if dmax > epsilon:
        rec_left = douglas_peucker(points[:index+1], epsilon)
        rec_right = douglas_peucker(points[index:], epsilon)
        return rec_left[:-1] + rec_right
    else:
        return [points[0], points[-1]]

def smooth_path_with_bspline(path, num_points=200, smoothness=0.5):
    if len(path) < 4:
        print("Not enough points to apply B-spline. Returning original path.")
        return path

    x = [p[0] for p in path]
    y = [p[1] for p in path]

    # Ensure x and y are in increasing order
    if not (np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0)):
        print("Input points are not in increasing order. Sorting them.")
        sorted_path = sorted(path, key=lambda p: (p[0], p[1]))
        x = [p[0] for p in sorted_path]
        y = [p[1] for p in sorted_path]

    try:
        tck, _ = splrep(x, y, s=smoothness)
        u = np.linspace(0, 1, num_points)
        xs, ys = splev(u, tck)
        return list(zip(xs.astype(int), ys.astype(int)))
    except Exception as e:
        print(f"Error during B-spline smoothing: {e}")
        return path

def find_safe_path(start, end, crater_boxes, image_shape, buffer_pixels=15, grid_size=100):
    x_grid, y_grid = create_coarse_grid(image_shape, grid_size)
    G = nx.grid_2d_graph(len(x_grid), len(y_grid))
    
    mark_crater_cells(G, crater_boxes, x_grid, y_grid, image_shape, buffer_pixels)
    
    start_xi = np.clip(np.searchsorted(x_grid, start[0]), 0, len(x_grid)-1)
    start_yi = np.clip(np.searchsorted(y_grid, start[1]), 0, len(y_grid)-1)
    end_xi = np.clip(np.searchsorted(x_grid, end[0]), 0, len(x_grid)-1)
    end_yi = np.clip(np.searchsorted(y_grid, end[1]), 0, len(y_grid)-1)

    start_node = (start_xi, start_yi)
    end_node = (end_xi, end_yi)
    
    def find_nearest_valid_node(node):
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        for dx, dy in directions:
            candidate = (node[0]+dx, node[1]+dy)
            if candidate in G:
                return candidate
        return None
    
    if start_node not in G:
        print(f"Start node {start_node} is in a crater zone!")
        start_node = find_nearest_valid_node(start_node)
        if start_node is None:
            print("No valid start node found!")
            return None
    
    if end_node not in G:
        print(f"End node {end_node} is in a crater zone!")
        end_node = find_nearest_valid_node(end_node)
        if end_node is None:
            print("No valid end node found!")
            return None

    try:
        def euclidean_heuristic(node, goal):
            return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
        
        path_grid = nx.astar_path(G, start_node, end_node, heuristic=euclidean_heuristic)
        path = [(x_grid[x], y_grid[y]) for x, y in path_grid]
        
        simplified_path = douglas_peucker(path, epsilon=5)
        print(f"Simplified path length: {len(simplified_path)}")
        smoothed_path = smooth_path_with_bspline(simplified_path, num_points=100, smoothness=0.5)
        
        return smoothed_path
    except nx.NetworkXNoPath:
        print("No valid path exists with current buffer settings")
        return None

__all__ = ['find_safe_path']