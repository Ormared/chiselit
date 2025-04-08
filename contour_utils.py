import cv2
import numpy as np
from typing import List, Tuple, Optional

class ContourDetector:
    def __init__(self):
        self.min_contour_area = 1000  # Minimum area to consider a contour
        self.approximation_epsilon = 0.02  # Approximation accuracy for contour simplification

    def detect_slab_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the main slab contour in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Contour points as numpy array if found, None otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        # Approximate the contour to reduce points
        epsilon = self.approximation_epsilon * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return approx_contour

    def draw_contour(self, image: np.ndarray, contour: np.ndarray, 
                    color: Tuple[int, int, int] = (0, 255, 0), 
                    thickness: int = 2) -> np.ndarray:
        """
        Draw a contour on the image.
        
        Args:
            image: Input image in BGR format
            contour: Contour points as numpy array
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with contour drawn
        """
        result = image.copy()
        cv2.drawContours(result, [contour], -1, color, thickness)
        return result

    def find_nearest_point(self, contour: np.ndarray, 
                         point: Tuple[float, float]) -> Tuple[int, float]:
        """
        Find the nearest point on the contour to the given point.
        
        Args:
            contour: Contour points as numpy array
            point: (x, y) coordinates
            
        Returns:
            Tuple of (index, distance) of nearest point
        """
        distances = []
        for i, contour_point in enumerate(contour):
            x, y = contour_point[0]
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            distances.append((i, dist))
            
        return min(distances, key=lambda x: x[1])

    def add_point_to_contour(self, contour: np.ndarray, 
                           point: Tuple[float, float]) -> np.ndarray:
        """
        Add a new point to the contour at the nearest edge.
        
        Args:
            contour: Contour points as numpy array
            point: (x, y) coordinates of new point
            
        Returns:
            Updated contour with new point
        """
        if len(contour) < 2:
            # Create new point with correct shape (1, 1, 2)
            new_point = np.array([[[point[0], point[1]]]])
            return np.vstack([contour, new_point])
            
        # Find nearest point
        nearest_idx, _ = self.find_nearest_point(contour, point)
        
        # Create new point with correct shape (1, 1, 2)
        new_point = np.array([[[point[0], point[1]]]])
        
        # Insert new point after nearest point
        return np.vstack([contour[:nearest_idx+1], new_point, contour[nearest_idx+1:]])

    def remove_point_from_contour(self, contour: np.ndarray, 
                                point: Tuple[float, float]) -> np.ndarray:
        """
        Remove the nearest point from the contour.
        
        Args:
            contour: Contour points as numpy array
            point: (x, y) coordinates near point to remove
            
        Returns:
            Updated contour without the removed point
        """
        if len(contour) <= 3:  # Keep at least 3 points for a valid contour
            return contour
            
        nearest_idx, _ = self.find_nearest_point(contour, point)
        return np.vstack([contour[:nearest_idx], contour[nearest_idx+1:]])

    def move_point_in_contour(self, contour: np.ndarray, 
                            old_point: Tuple[float, float],
                            new_point: Tuple[float, float]) -> np.ndarray:
        """
        Move a point in the contour to a new position.
        
        Args:
            contour: Contour points as numpy array
            old_point: Original point coordinates
            new_point: New point coordinates
            
        Returns:
            Updated contour with moved point
        """
        nearest_idx, _ = self.find_nearest_point(contour, old_point)
        result = contour.copy()
        result[nearest_idx] = [[new_point[0], new_point[1]]]
        return result
