import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
import os

# Helper function for line segment intersection (using determinants)
# Returns intersection point or None
def _intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:  # Lines are parallel or collinear
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    # Check if intersection point is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return np.array([intersect_x, intersect_y])

    return None # Intersection point is not on segments

@dataclass
class Recipe:
    name: str
    points: np.ndarray  # Nx2 array of (x, y) points
    quantity: int = 1
    position: Tuple[float, float] = (0.0, 0.0)  # (x, y) translation
    rotation: float = 0.0  # Rotation in degrees
    margin: float = 50.0  # Margin in millimeters (used for border/wave generation)
    angle: float = 30.0  # Angle for wave pattern in degrees

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] != 2:
             raise ValueError("points must be a Nx2 array")
        if len(self.points) < 1:
            raise ValueError("points array cannot be empty")

    def to_dict(self) -> Dict:
        """Convert recipe to dictionary for saving."""
        return {
            'name': self.name,
            'points': self.points.tolist(),
            'quantity': self.quantity,
            'position': self.position,
            'rotation': self.rotation,
            'margin': self.margin,
            'angle': self.angle
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Recipe':
        """Create recipe from dictionary."""
        return cls(
            name=data['name'],
            points=np.array(data['points']),
            quantity=data.get('quantity', 1),
            position=data.get('position', (0, 0)),
            rotation=data.get('rotation', 0),
            margin=data.get('margin', 50.0),
            angle=data.get('angle', 30.0)
        )

class RecipeManager:
    def __init__(self, recipe_dir: str = "recipes"):
        self.recipe_dir = Path(recipe_dir)
        self.recipes: Dict[str, Recipe] = {}
        self.load_recipes()

    def load_recipes(self):
        """Load all recipe files from the recipe directory."""
        if not self.recipe_dir.exists():
            self.recipe_dir.mkdir(parents=True)
            return

        recipe_json = self.recipe_dir / 'recipes.json'
        if recipe_json.exists():
            try:
                with open(recipe_json, 'r') as f:
                    all_data = json.load(f)
                    for name, data in all_data.items():
                        data['name'] = name # Ensure name from key is used
                        recipe = Recipe.from_dict(data)
                    self.recipes[recipe.name] = recipe
            except Exception as e:
                print(f"Error loading recipes file {recipe_json}: {e}")

    def save_recipes(self):
        """Save all recipes to a single JSON file."""
        if not self.recipe_dir.exists():
            self.recipe_dir.mkdir(parents=True)
        
        data_to_save = {}
        for name, recipe in self.recipes.items():
            data_to_save[name] = recipe.to_dict()
            
        recipe_json = self.recipe_dir / 'recipes.json'
        try:
            with open(recipe_json, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
             print(f"Error saving recipes file {recipe_json}: {e}")

    def get_recipe(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self.recipes.get(name)

    def get_all_recipes(self) -> List[Recipe]:
        """Get all available recipes."""
        return list(self.recipes.values())

    def transform_recipe_points(self, recipe: Recipe) -> np.ndarray:
        """
        Transform recipe points based on its position and rotation.
        
        Args:
            recipe: Recipe to transform
            
        Returns:
            Transformed points as numpy array
        """
        # Create rotation matrix
        angle_rad = np.radians(recipe.rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Apply rotation and translation
        points = recipe.points.copy()
        # Ensure points are 2D for matmul
        if points.ndim == 1:
             points = points.reshape(1, -1)
        elif points.ndim != 2 or points.shape[1] != 2:
            print(f"Warning: Invalid shape for points in transform: {points.shape}")
            return np.array([]) # Return empty if shape is wrong
            
        points = points @ rotation_matrix.T
        points += np.array(recipe.position)

        return points

    def check_recipe_placement(self, recipe: Recipe, slab_contour: np.ndarray) -> bool:
        """
        Check if a recipe fits within the slab contour.
        Uses pointPolygonTest on the transformed recipe points.
        Args:
            recipe: Recipe to check
            slab_contour: Slab contour points (Nx1x2 or Nx2)
        Returns:
            True if recipe fits, False otherwise
        """
        if slab_contour is None or len(slab_contour) < 3:
            return False # Cannot fit in invalid contour
        
        # Ensure slab_contour is in the format needed by pointPolygonTest (Nx1x2)
        if slab_contour.ndim == 2:
            contour_for_test = slab_contour.reshape(-1, 1, 2).astype(np.float32)
        elif slab_contour.ndim == 3 and slab_contour.shape[1] == 1 and slab_contour.shape[2] == 2:
            contour_for_test = slab_contour.astype(np.float32)
        else:
            print(f"Warning: Invalid slab_contour shape for placement check: {slab_contour.shape}")
            return False
            
        # Transform recipe points
        transformed_points = self.transform_recipe_points(recipe)
        if len(transformed_points) == 0:
            return True # Empty recipe technically fits
        
        # Check if all points are inside or on the edge of the slab contour
        for point in transformed_points:
            point_tuple = (float(point[0]), float(point[1]))
            # Use measureDist=False, returns +1 (inside), 0 (on edge), -1 (outside)
            if cv2.pointPolygonTest(contour_for_test, point_tuple, False) < 0:
                return False
                
        return True

    def _offset_contour(self, contour: np.ndarray, offset: float) -> Optional[np.ndarray]:
        """ Internal helper to create an offset contour (inset). """
        if contour is None or len(contour) < 3:
            return None
        if abs(offset) < 1e-6:
            return contour.reshape(-1, 2) # No offset needed

        points = contour.reshape(-1, 2).astype(np.float64)
        num_points = len(points)
        new_points = []

        for i in range(num_points):
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p0 = points[(i - 1 + num_points) % num_points]

            v1 = p2 - p1
            v0 = p1 - p0
            norm_v1 = np.linalg.norm(v1)
            norm_v0 = np.linalg.norm(v0)

            if norm_v1 < 1e-6 or norm_v0 < 1e-6:
                # Skip potentially collinear points causing issues
                continue

            # Inward normal depends on contour orientation (assume CCW from OpenCV)
            # For inset (negative offset conceptually), normals point inward
            inward_norm1 = np.array([v1[1], -v1[0]]) / norm_v1
            inward_norm0 = np.array([v0[1], -v0[0]]) / norm_v0

            # Offset distance 'd' for the line equation p.n = d
            # For inset, add the offset value to the dot product
            d1 = np.dot(p1, inward_norm1) + offset
            d0 = np.dot(p1, inward_norm0) + offset

            # Calculate determinant for intersection
            determinant = inward_norm0[0] * inward_norm1[1] - inward_norm0[1] * inward_norm1[0]

            if abs(determinant) < 1e-6: # Parallel lines
                 print(f"Warning: Offset lines parallel for vertex {i}. Margin might be too large or contour has issues.")
                 # Attempt to handle by slightly adjusting one normal?
                 # Or simply fail for this contour.
                 # For simplicity, let's fail.
                 return None

            # Solve for intersection point (Cramer's rule)
            new_x = (d0 * inward_norm1[1] - d1 * inward_norm0[1]) / determinant
            new_y = (d1 * inward_norm0[0] - d0 * inward_norm1[0]) / determinant
            new_points.append([new_x, new_y])

        if len(new_points) < 3:
            return None
            
        final_points = np.array(new_points)
        
        # Final check: ensure the offset polygon is smaller and roughly convex if original was
        # This implementation assumes convexity and CCW order.
        # A robust solution might involve more complex geometry libraries (e.g., Shapely)
        
        return final_points

    def generate_robot_path(self, recipe: Recipe, 
                          processing_z: float,
                          safe_z: float,
                          speed: float) -> List[str]:
        """
        Generate KUKA KRL path commands for a recipe.
        
        Args:
            recipe: Recipe to generate path for
            processing_z: Z coordinate for processing
            safe_z: Z coordinate for safe movements
            speed: Processing speed in mm/s
            
        Returns:
            List of KRL commands
        """
        commands: List[str] = []
        transformed_points = self.transform_recipe_points(recipe)
        
        if len(transformed_points) == 0:
             return commands # No points to process
        
        # Move to safe height above start point
        start_x, start_y = transformed_points[0]
        commands.append(f"LIN {{X {start_x:.1f}, Y {start_y:.1f}, Z {safe_z:.1f}}} C_DIS")
        
        # Move to processing height
        commands.append(f"LIN {{X {start_x:.1f}, Y {start_y:.1f}, Z {processing_z:.1f}}} C_DIS")
        
        # Set processing speed
        commands.append(f"$VEL.CP = {speed}")
        
        # Generate path points
        # For Border recipe, ensure we loop back to the start
        points_to_process = transformed_points[1:]
        if recipe.name == "Border" and len(transformed_points) > 1:
            # Add start point to the end for closed loop
            points_to_process = np.vstack([points_to_process, transformed_points[0]]) 
            
        for x, y in points_to_process:
            commands.append(f"LIN {{X {x:.1f}, Y {y:.1f}, Z {processing_z:.1f}}} C_DIS")
            
        # Retract to safe height from the last point of the trajectory
        if len(points_to_process) > 0:
             last_x, last_y = points_to_process[-1]
        else: # Only one point in recipe
             last_x, last_y = start_x, start_y
        commands.append(f"LIN {{X {last_x:.1f}, Y {last_y:.1f}, Z {safe_z:.1f}}} C_DIS")
        
        return commands

    def generate_border_recipe(self, base_contour: np.ndarray, margin: float = 50) -> Optional[Recipe]:
        """
        Generate a border recipe as a smaller convex polygon inside the base convex contour.
        The distance between the base contour and the border contour is exactly 'margin'.
        
        Args:
            base_contour: The base convex contour points (Nx1x2 numpy array).
            margin: Distance in millimeters to offset inwards from the base contour.
            
        Returns:
            Recipe object containing the border contour (Nx2 numpy array), or None if generation fails.
        """
        inner_points = self._offset_contour(base_contour, margin)
        
        if inner_points is None or len(inner_points) < 3:
            print("Error: Failed to generate inner contour for border recipe.")
            return None
            
        # Sanity Check: Ensure all generated points are inside the original contour
        original_contour_for_test = base_contour.reshape(-1, 1, 2).astype(np.int32)
        for p in inner_points:
             point_tuple = (float(p[0]), float(p[1]))
             if cv2.pointPolygonTest(original_contour_for_test, point_tuple, False) < 0:
                 print(f"Warning: Generated border point {p} is outside the base contour. Margin might be too large or contour might not be convex.")
                 # Optional: Allow points slightly outside due to float precision?
                 # For now, strict check.
                 return None # Indicate failure

        return Recipe(
            name="Border",
            points=inner_points,
            quantity=1,
            margin=margin,
            angle=0 # Angle not relevant for border
        )

    def add_recipe(self, recipe: Recipe):
        """Add a new recipe."""
        self.recipes[recipe.name] = recipe
        self.save_recipes()

    def remove_recipe(self, name: str):
        """Remove a recipe by name."""
        if name in self.recipes:
            del self.recipes[name]
            self.save_recipes()

    def generate_wave_recipe(self, base_contour: np.ndarray, margin: float = 50.0, angle: float = 30.0) -> Optional[Recipe]:
        """
        Generate a wave pattern recipe using a zigzag reflection path.
        Starts at the top-middle of the inner contour and bounces off walls.

        Args:
            base_contour: The base convex contour points (Nx1x2 numpy array).
            margin: Distance in millimeters to offset inwards from the base contour.
            angle: Zigzag angle relative to the downward vertical axis (degrees).

        Returns:
            Recipe object containing the wave path (Nx2 numpy array), or None if generation fails.
        """
        if angle <= 0 or angle >= 90:
             print("Error: Wave angle must be between 0 and 90 degrees.")
             return None
             
        # 1. Get the inner contour
        inner_contour = self._offset_contour(base_contour, margin)
        if inner_contour is None or len(inner_contour) < 3:
            print("Error: Failed to generate inner contour for wave recipe.")
            return None
            
        # 2. Find the top starting point
        min_y = np.min(inner_contour[:, 1])
        top_indices = np.where(np.abs(inner_contour[:, 1] - min_y) < 1e-6)[0]
        
        if len(top_indices) == 0: # Should not happen if inner_contour is valid
            return None 
        elif len(top_indices) == 1:
            start_point = inner_contour[top_indices[0]]
        else: # Horizontal top edge - find midpoint
            # Find the two points forming the top edge
            x_coords = inner_contour[top_indices, 0]
            min_x_idx = top_indices[np.argmin(x_coords)]
            max_x_idx = top_indices[np.argmax(x_coords)]
            start_point = (inner_contour[min_x_idx] + inner_contour[max_x_idx]) / 2.0
            
        # 3. Simulate the zigzag path
        wave_points = [start_point]
        current_point = start_point
        go_right = True # Start by going right
        max_iterations = 100 # Safety break
        num_segments = len(inner_contour)
        angle_rad = np.radians(angle)

        for _ in range(max_iterations):
            current_angle_offset = angle_rad if go_right else -angle_rad
            # Vector angle relative to positive X-axis: 270 degrees is down.
            # Angle is relative to *vertical*. Downward vertical vector (0, -1).
            # Rotate (0,-1) by +/- angle_rad:
            # x = 0*cos - (-1)*sin = sin
            # y = 0*sin + (-1)*cos = -cos
            direction_vector = np.array([np.sin(current_angle_offset), -np.cos(current_angle_offset)])

            # Define a long ray starting from current_point
            ray_start = current_point
            # Ensure ray is long enough to cross the contour
            max_dim = np.max(np.ptp(inner_contour, axis=0)) # Max width/height
            ray_end = current_point + direction_vector * max_dim * 2 

            closest_intersect = None
            min_dist_sq = float('inf')

            # Check intersection with all contour segments
            for i in range(num_segments):
                p_seg_start = inner_contour[i]
                p_seg_end = inner_contour[(i + 1) % num_segments]
                
                intersect = _intersect(ray_start, ray_end, p_seg_start, p_seg_end)
                
                if intersect is not None:
                    dist_sq = np.sum((intersect - current_point)**2)
                    # Find the closest intersection that is not the current point
                    if dist_sq > 1e-6 and dist_sq < min_dist_sq:
                         min_dist_sq = dist_sq
                         closest_intersect = intersect
            
            # If no valid intersection found, stop
            if closest_intersect is None:
                break
            
            # Add the intersection point and continue
            wave_points.append(closest_intersect)
            current_point = closest_intersect
            go_right = not go_right # Change direction

        if len(wave_points) < 2:
            print("Error: Failed to generate any wave path segments.")
            return None

        # Create the recipe
        return Recipe(
            name="Wave Pattern",
            points=np.array(wave_points),
            quantity=1,
            margin=margin,
            angle=angle # Store the angle used
        ) 