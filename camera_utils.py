import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class CameraCalibration:
    intrinsic_matrix: np.ndarray  # 3x3 camera matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients
    camera_pose: np.ndarray  # 4x4 transformation matrix from camera to robot base

class CameraUtils:
    def __init__(self, calibration_file: str = "camera_calibration.json"):
        self.calibration_file = Path(calibration_file)
        self.calibration: Optional[CameraCalibration] = None
        self.load_calibration()

    def load_calibration(self):
        """Load camera calibration data from file."""
        if not self.calibration_file.exists():
            print(f"Warning: Calibration file {self.calibration_file} not found")
            return

        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                self.calibration = CameraCalibration(
                    intrinsic_matrix=np.array(data['intrinsic_matrix']),
                    distortion_coeffs=np.array(data['distortion_coeffs']),
                    camera_pose=np.array(data['camera_pose'])
                )
        except Exception as e:
            print(f"Error loading camera calibration: {e}")

    def save_calibration(self):
        """Save camera calibration data to file."""
        if self.calibration is None:
            print("No calibration data to save")
            return

        try:
            data = {
                'intrinsic_matrix': self.calibration.intrinsic_matrix.tolist(),
                'distortion_coeffs': self.calibration.distortion_coeffs.tolist(),
                'camera_pose': self.calibration.camera_pose.tolist()
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving camera calibration: {e}")

    def pixel_to_robot_coordinates(self, pixel_point: Tuple[float, float], 
                                 z_plane: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to robot base coordinates.
        
        Args:
            pixel_point: (x, y) pixel coordinates
            z_plane: Z coordinate of the plane in robot base frame
            
        Returns:
            (x, y) robot coordinates if successful, None otherwise
        """
        if self.calibration is None:
            print("No camera calibration available")
            return None

        # Convert pixel coordinates to normalized camera coordinates
        pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1.0])
        camera_matrix_inv = np.linalg.inv(self.calibration.intrinsic_matrix)
        camera_coords = camera_matrix_inv @ pixel_homogeneous

        # Get camera position and orientation from pose matrix
        camera_position = self.calibration.camera_pose[:3, 3]
        camera_rotation = self.calibration.camera_pose[:3, :3]

        # Calculate ray direction in robot base frame
        ray_direction = camera_rotation @ camera_coords
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Calculate intersection with Z plane
        t = (z_plane - camera_position[2]) / ray_direction[2]
        robot_point = camera_position + t * ray_direction

        return (robot_point[0], robot_point[1])

    def robot_to_pixel_coordinates(self, robot_point: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """
        Convert robot base coordinates to pixel coordinates.
        
        Args:
            robot_point: (x, y, z) robot coordinates
            
        Returns:
            (x, y) pixel coordinates if successful, None otherwise
        """
        if self.calibration is None:
            print("No camera calibration available")
            return None

        # Convert robot point to homogeneous coordinates
        robot_homogeneous = np.array([robot_point[0], robot_point[1], robot_point[2], 1.0])

        # Transform to camera coordinates
        camera_pose_inv = np.linalg.inv(self.calibration.camera_pose)
        camera_point = camera_pose_inv @ robot_homogeneous
        camera_point = camera_point[:3] / camera_point[3]

        # Project to image plane
        pixel_homogeneous = self.calibration.intrinsic_matrix @ camera_point
        pixel_point = pixel_homogeneous[:2] / pixel_homogeneous[2]

        return (pixel_point[0], pixel_point[1])

    def transform_contour(self, contour: np.ndarray, z_plane: float) -> np.ndarray:
        """
        Transform a contour from image coordinates to robot coordinates.
        
        Args:
            contour: Contour points as numpy array
            z_plane: Z coordinate in robot space
            
        Returns:
            Transformed contour points in robot coordinates
        """
        if contour is None:
            return None
            
        # Transform each point
        robot_points = []
        for point in contour:
            # Access the point coordinates correctly from the (1, 2) shape
            x, y = point[0]
            robot_point = self.pixel_to_robot_coordinates((x, y), z_plane)
            robot_points.append(robot_point)
            
        return np.array(robot_points)

    def transform_recipe(self, recipe_points: np.ndarray, 
                        z_plane: float) -> Optional[np.ndarray]:
        """
        Transform recipe points from pixel coordinates to robot coordinates.
        
        Args:
            recipe_points: Nx2 array of (x, y) pixel coordinates
            z_plane: Z coordinate of the plane in robot base frame
            
        Returns:
            Nx2 array of (x, y) robot coordinates if successful, None otherwise
        """
        return self.transform_contour(recipe_points, z_plane) 