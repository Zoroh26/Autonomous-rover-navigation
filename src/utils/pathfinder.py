# Ensure these imports work from any location
import numpy as np
import networkx as nx
from config import PROJECT_ROOT  # Access project-wide configs if needed

def create_coarse_grid(image_shape, grid_size=100):
    """Create a coarse grid for pathfinding"""
    h, w = image_shape[:2]
    x_grid = np.linspace(0, w-1, grid_size, dtype=int)
    y_grid = np.linspace(0, h-1, grid_size, dtype=int)
    return x_grid, y_grid

def mark_crater_cells(graph, crater_boxes, x_grid, y_grid):
    """Mark grid cells containing craters as blocked"""
    for x1, y1, x2, y2 in crater_boxes:
        xi_min = np.searchsorted(x_grid, x1) - 1
        xi_max = np.searchsorted(x_grid, x2)
        yi_min = np.searchsorted(y_grid, y1) - 1
        yi_max = np.searchsorted(y_grid, y2)
        
        for xi in range(max(0, xi_min), min(len(x_grid), xi_max)):
            for yi in range(max(0, yi_min), min(len(y_grid), yi_max)):
                if (xi, yi) in graph:
                    graph.remove_node((xi, yi))

def find_safe_path(start, end, crater_boxes, image_shape, buffer_pixels=15):
    """Improved pathfinding with node validation"""
    x_grid, y_grid = create_coarse_grid(image_shape)
    G = nx.grid_2d_graph(len(x_grid), len(y_grid))
    
    # Apply buffer with boundary checks
    buffered_boxes = []
    for (x1, y1, x2, y2) in crater_boxes:
        buffered_boxes.append((
            max(0, x1 - buffer_pixels),
            max(0, y1 - buffer_pixels),
            min(image_shape[1]-1, x2 + buffer_pixels),
            min(image_shape[0]-1, y2 + buffer_pixels)
        ))
    
    # Mark craters with debug info
    removed_nodes = 0
    for box in buffered_boxes:
        x1, y1, x2, y2 = box
        xi_min = np.searchsorted(x_grid, x1)
        xi_max = np.searchsorted(x_grid, x2)
        yi_min = np.searchsorted(y_grid, y1)
        yi_max = np.searchsorted(y_grid, y2)
        
        for xi in range(max(0, xi_min), min(len(x_grid), xi_max)):
            for yi in range(max(0, yi_min), min(len(y_grid), yi_max)):
                if (xi, yi) in G:
                    G.remove_node((xi, yi))
                    removed_nodes += 1
    
    print(f"Removed {removed_nodes} nodes from graph")  # Debug info

    # Validate start/end nodes
    start_xi = np.clip(np.searchsorted(x_grid, start[0]), 0, len(x_grid)-1)
    start_yi = np.clip(np.searchsorted(y_grid, start[1]), 0, len(y_grid)-1)
    end_xi = np.clip(np.searchsorted(x_grid, end[0]), 0, len(x_grid)-1)
    end_yi = np.clip(np.searchsorted(y_grid, end[1]), 0, len(y_grid)-1)

    start_node = (start_xi, start_yi)
    end_node = (end_xi, end_yi)
    
    # Check node existence
    if start_node not in G:
        print(f"Start node {start_node} is in a crater zone!")
        # Find nearest valid node
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            candidate = (start_xi+dx, start_yi+dy)
            if candidate in G:
                start_node = candidate
                break
    
    if end_node not in G:
        print(f"End node {end_node} is in a crater zone!")
        # Find nearest valid node
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            candidate = (end_xi+dx, end_yi+dy)
            if candidate in G:
                end_node = candidate
                break

    try:
        path_grid = nx.astar_path(G, start_node, end_node)
        path = [(x_grid[x], y_grid[y]) for x, y in path_grid]
        
        # Simplify path
        if len(path) > 4:
            path = [path[i] for i in range(0, len(path), 2)] + [path[-1]]
            
        return path
    except nx.NetworkXNoPath:
        print("No valid path exists with current buffer settings")
        return None

__all__ = ['find_safe_path']  # Keep this export