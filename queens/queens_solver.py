from typing import List, Tuple

import sys
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw


def grid_bounding_box(image: np.ndarray) -> Tuple[int, int, int, int]:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary (black and white)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assume it's the grid)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def find_boundaries(image: np.ndarray) -> tuple:
    x, y, w, h = grid_bounding_box(image)
    cropped_grid = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    # Threshold the cropped grid area again
    _, grid_thresh = cv2.threshold(cropped_grid, 128, 255, cv2.THRESH_BINARY_INV)

    # Use horizontal and vertical projections to detect grid lines
    horizontal_projection = np.sum(grid_thresh, axis=1)
    vertical_projection = np.sum(grid_thresh, axis=0)

    # Detect peaks (grid lines) in the projections
    # A peak corresponds to a grid line
    horizontal_lines = np.where(horizontal_projection > np.max(horizontal_projection) * 0.5)[0]
    vertical_lines = np.where(vertical_projection > np.max(vertical_projection) * 0.5)[0]

    # Group nearby lines to count them as one
    def group_lines(lines, threshold=5):
        grouped_lines = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - grouped_lines[-1] > threshold:
                grouped_lines.append(lines[i])
        return grouped_lines

    horizontal_lines = group_lines(horizontal_lines)
    vertical_lines = group_lines(vertical_lines)

    return horizontal_lines, vertical_lines


def get_cropped_image(image: np.ndarray) -> Image:
    x, y, w, h = grid_bounding_box(image)

    cropped_image = image[y:y + h, x:x + w]
    return Image.fromarray(cropped_image)


def convert_to_num_grid(image: np.ndarray) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
    horizontal_lines, vertical_lines = find_boundaries(image)
    x, y, w, h = grid_bounding_box(image)

    cropped_image = image[y:y + h, x:x + w]
    color_map = defaultdict(list)

    h, v = len(horizontal_lines), len(vertical_lines)
    center_coords = [[(0, 0) for j in range(v - 1)] for i in range(h - 1)]

    for i in range(h - 1):
        for j in range(v - 1):
            x = (horizontal_lines[i] + horizontal_lines[i + 1]) // 2
            y = (vertical_lines[j] + vertical_lines[j + 1]) // 2
            center_coords[i][j] = (x, y)
            color = cropped_image[x, y]
            color_map[tuple(color)].append((i, j))

    num_grid = [[-1 for _ in range(h - 1)] for _ in range(v - 1)]
    for idx, (_, locs) in enumerate(color_map.items()):
        for i, j in locs:
            num_grid[i][j] = idx

    return num_grid, center_coords


def solve(grid: List[List[int]]) -> List[Tuple[int, int]]:
    m, n = len(grid), len(grid[0])
    seed = {}

    used_cols = set(x[0] for x in seed.values())
    used_colors = set(grid[i][j] for i, j in seed.items())

    def backtrack(row=0, used_cols=used_cols, used_regions=used_colors, ret=[]):
        if row == len(grid):
            return ret
        if row in seed:
            col = seed[row]
            new_cols = used_cols | {col}
            new_regions = used_regions | {grid[row][col]}
            new_ret = ret + [(row, col)]
            return backtrack(row + 1, new_cols, new_regions, new_ret)
        for col in range(n):
            diag = abs(ret[-1][1] - col) == 1 if ret else False
            if col not in used_cols and grid[row][col] not in used_regions and not diag:
                new_cols = used_cols | {col}
                new_regions = used_regions | {grid[row][col]}
                new_ret = ret + [(row, col)]
                sol = backtrack(row + 1, new_cols, new_regions, new_ret)
                if sol:
                    return sol

    return backtrack()


def plot(image: Image, solution: List[Tuple[int, int]], r = 10) -> Image:
    final_grid = image.copy()
    draw = ImageDraw.Draw(final_grid)

    for y, x in solution:
        bounding_box = (x - r, y - r, x + r, y + r)
        draw.ellipse(bounding_box, fill="black", outline="black", width=2)

    return final_grid


# Test the function
if __name__ == "__main__":
    image_path = sys.argv[1]
    if not image_path.endswith(".png"):
        raise Exception("Image must be a .png file.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Make sure the path is correct.")

    grid, center_coords = convert_to_num_grid(image)
    solution = solve(grid)
    solution_centers = [center_coords[i][j] for i, j in solution]
    completed_grid = plot(get_cropped_image(image), solution_centers)
    completed_grid.save("./solutions/" + image_path[:-4] + "_solved.png")
