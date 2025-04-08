import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                           QGroupBox, QScrollArea, QMessageBox, QListWidget,
                           QListWidgetItem, QSlider, QButtonGroup)
from PyQt6.QtCore import Qt, QPoint, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QTransform
from contour_utils import ContourDetector
from recipe_manager import RecipeManager, Recipe
from program_generator import ProgramGenerator, ProgramConfig
from camera_utils import CameraUtils

class StoneProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stone Slab Processing Path Generator")
        self.setMinimumSize(1200, 800)
        
        # Initialize variables
        self.image = None
        self.display_image = None
        self.slab_contour = None
        self.robot_contour = None  # Store transformed contour in robot coordinates
        self.recipe_contours = {}
        self.selected_recipe = None
        self.camera_calibration = None
        self.contour_fixed = False  # Flag to track if contour is fixed
        self.center_marker = None  # Store center marker position
        
        # Initialize utilities
        self.contour_detector = ContourDetector()
        self.recipe_manager = RecipeManager()
        self.program_generator = ProgramGenerator()
        self.camera_utils = CameraUtils()
        
        # Mouse interaction variables
        self.dragging_point = None
        self.drag_start_pos = None
        self.drag_original_pos = None
        self.point_radius = 5
        self.nearest_point_threshold = 10
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create left panel (image and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.image_label)
        
        # Image controls
        image_controls = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.capture_image_btn = QPushButton("Capture Image")
        self.capture_image_btn.clicked.connect(self.capture_image)
        image_controls.addWidget(self.load_image_btn)
        image_controls.addWidget(self.capture_image_btn)
        
        # Point editing controls
        point_controls = QHBoxLayout()
        point_controls.addWidget(QLabel("Point Editing:"))
        
        # Create button group for point editing modes
        self.point_edit_group = QButtonGroup(self)
        self.point_edit_group.setExclusive(True)
        
        # Add point mode button
        self.add_point_btn = QPushButton("Add Points")
        self.add_point_btn.setCheckable(True)
        self.add_point_btn.clicked.connect(self.on_point_mode_changed)
        self.point_edit_group.addButton(self.add_point_btn)
        point_controls.addWidget(self.add_point_btn)
        
        # Remove point mode button
        self.remove_point_btn = QPushButton("Remove Points")
        self.remove_point_btn.setCheckable(True)
        self.remove_point_btn.clicked.connect(self.on_point_mode_changed)
        self.point_edit_group.addButton(self.remove_point_btn)
        point_controls.addWidget(self.remove_point_btn)
        
        # Move point mode button
        self.move_point_btn = QPushButton("Move Points")
        self.move_point_btn.setCheckable(True)
        self.move_point_btn.setChecked(True)  # Default mode
        self.move_point_btn.clicked.connect(self.on_point_mode_changed)
        self.point_edit_group.addButton(self.move_point_btn)
        point_controls.addWidget(self.move_point_btn)
        
        left_layout.addLayout(image_controls)
        left_layout.addLayout(point_controls)
        
        main_layout.addWidget(left_panel)
        
        # Create right panel (settings and controls)
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Height Parameters Group
        height_group = QGroupBox("Height Parameters")
        height_layout = QVBoxLayout()
        
        # T (Manipulator-to-Table distance)
        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("T (mm):"))
        self.t_input = QDoubleSpinBox()
        self.t_input.setRange(0, 1000)
        self.t_input.setValue(200)
        t_layout.addWidget(self.t_input)
        height_layout.addLayout(t_layout)
        
        # Slab Thickness
        slab_layout = QHBoxLayout()
        slab_layout.addWidget(QLabel("Slab Thickness (mm):"))
        self.slab_thickness_input = QDoubleSpinBox()
        self.slab_thickness_input.setRange(0, 500)
        self.slab_thickness_input.setValue(50)
        slab_layout.addWidget(self.slab_thickness_input)
        height_layout.addLayout(slab_layout)
        
        # Safe Height
        safe_layout = QHBoxLayout()
        safe_layout.addWidget(QLabel("Safe Height (mm):"))
        self.safe_height_input = QDoubleSpinBox()
        self.safe_height_input.setRange(0, 1000)
        self.safe_height_input.setValue(250)
        safe_layout.addWidget(self.safe_height_input)
        height_layout.addLayout(safe_layout)
        
        height_group.setLayout(height_layout)
        right_layout.addWidget(height_group)
        
        # Add recipe selection group after the height parameters group
        recipe_group = QGroupBox("Recipe Selection")
        recipe_layout = QVBoxLayout()
        
        # Recipe list
        self.recipe_list = QListWidget()
        self.recipe_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.recipe_list.itemSelectionChanged.connect(self.on_recipe_selection_changed)
        self.recipe_list.setMinimumHeight(200)  # Make the list widget bigger
        recipe_layout.addWidget(QLabel("Available Recipes:"))
        recipe_layout.addWidget(self.recipe_list)
        
        # Recipe position
        # position_group = QGroupBox("Position")
        # position_layout = QVBoxLayout()
        
        # # X position
        # x_pos_layout = QHBoxLayout()
        # x_pos_layout.addWidget(QLabel("X:"))
        # self.recipe_x_pos = QDoubleSpinBox()
        # self.recipe_x_pos.setRange(-1000, 1000)
        # x_pos_layout.addWidget(self.recipe_x_pos)
        # position_layout.addLayout(x_pos_layout)
        
        # # Y position
        # y_pos_layout = QHBoxLayout()
        # y_pos_layout.addWidget(QLabel("Y:"))
        # self.recipe_y_pos = QDoubleSpinBox()
        # self.recipe_y_pos.setRange(-1000, 1000)
        # y_pos_layout.addWidget(self.recipe_y_pos)
        # position_layout.addLayout(y_pos_layout)
        
        # position_group.setLayout(position_layout)
        # recipe_layout.addWidget(position_group)
        
        # Recipe rotation
        # rotation_layout = QHBoxLayout()
        # rotation_layout.addWidget(QLabel("Rotation:"))
        # self.recipe_rotation = QSlider(Qt.Orientation.Horizontal)
        # self.recipe_rotation.setRange(-180, 180)
        # self.recipe_rotation.setValue(0)
        # self.recipe_rotation.valueChanged.connect(self.on_recipe_rotation_changed)
        # rotation_layout.addWidget(self.recipe_rotation)
        # recipe_layout.addLayout(rotation_layout)
        
        # Recipe placement buttons
        button_layout = QHBoxLayout()
        self.place_recipe_btn = QPushButton("Place Recipe")
        self.place_recipe_btn.clicked.connect(self.place_recipe)
        self.place_recipe_btn.setEnabled(False)
        button_layout.addWidget(self.place_recipe_btn)
        
        self.remove_recipe_btn = QPushButton("Remove Recipe")
        self.remove_recipe_btn.clicked.connect(self.remove_recipe)
        self.remove_recipe_btn.setEnabled(False)
        button_layout.addWidget(self.remove_recipe_btn)
        
        recipe_layout.addLayout(button_layout)
        recipe_group.setLayout(recipe_layout)
        right_layout.addWidget(recipe_group)
        
        # End Effector Selection Group
        ee_group = QGroupBox("End Effectors")
        ee_layout = QHBoxLayout()
        self.ee_checkboxes = []
        for i in range(1, 13):
            checkbox = QCheckBox(f"EE{i}")
            self.ee_checkboxes.append(checkbox)
            ee_layout.addWidget(checkbox)
        ee_group.setLayout(ee_layout)
        right_layout.addWidget(ee_group)
        
        # Camera Offset Group
        camera_group = QGroupBox("Camera Offset")
        camera_layout = QVBoxLayout()
        
        # X offset
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X (mm):"))
        self.camera_x_input = QDoubleSpinBox()
        self.camera_x_input.setRange(-500, 500)
        x_layout.addWidget(self.camera_x_input)
        camera_layout.addLayout(x_layout)
        
        # Y offset
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y (mm):"))
        self.camera_y_input = QDoubleSpinBox()
        self.camera_y_input.setRange(-500, 500)
        y_layout.addWidget(self.camera_y_input)
        camera_layout.addLayout(y_layout)
        
        # Z offset
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z (mm):"))
        self.camera_z_input = QDoubleSpinBox()
        self.camera_z_input.setRange(-500, 500)
        z_layout.addWidget(self.camera_z_input)
        camera_layout.addLayout(z_layout)
        
        camera_group.setLayout(camera_layout)
        right_layout.addWidget(camera_group)
        
        # Processing Parameters Group
        processing_group = QGroupBox("Processing Parameters")
        processing_layout = QVBoxLayout()
        
        # Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed (mm/s):"))
        self.speed_input = QSpinBox()
        self.speed_input.setRange(1, 1000)
        self.speed_input.setValue(150)
        speed_layout.addWidget(self.speed_input)
        processing_layout.addLayout(speed_layout)
        
        # Spindle RPM
        rpm_layout = QHBoxLayout()
        rpm_layout.addWidget(QLabel("Spindle RPM:"))
        self.rpm_input = QSpinBox()
        self.rpm_input.setRange(0, 10000)
        self.rpm_input.setValue(550)
        rpm_layout.addWidget(self.rpm_input)
        processing_layout.addLayout(rpm_layout)
        
        # Force
        force_layout = QHBoxLayout()
        force_layout.addWidget(QLabel("Force (N):"))
        self.force_input = QSpinBox()
        self.force_input.setRange(0, 1000)
        self.force_input.setValue(220)
        force_layout.addWidget(self.force_input)
        processing_layout.addLayout(force_layout)
        
        processing_group.setLayout(processing_layout)
        right_layout.addWidget(processing_group)
        
        # Action Buttons
        self.detect_contour_btn = QPushButton("Detect Slab Contour")
        self.detect_contour_btn.clicked.connect(self.detect_slab_contour)
        right_layout.addWidget(self.detect_contour_btn)
        
        self.confirm_contour_btn = QPushButton("Confirm Slab Contour")
        self.confirm_contour_btn.clicked.connect(self.confirm_slab_contour)
        self.confirm_contour_btn.setEnabled(False)
        right_layout.addWidget(self.confirm_contour_btn)
        
        self.generate_program_btn = QPushButton("Generate Program")
        self.generate_program_btn.clicked.connect(self.generate_program)
        self.generate_program_btn.setEnabled(False)
        right_layout.addWidget(self.generate_program_btn)
        
        self.export_program_btn = QPushButton("Export Program(s)")
        self.export_program_btn.clicked.connect(self.export_programs)
        self.export_program_btn.setEnabled(False)
        right_layout.addWidget(self.export_program_btn)
        
        right_panel.setWidget(right_widget)
        main_layout.addWidget(right_panel)
        
        # Set up mouse tracking for image interaction
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release

        # Initialize recipe-related variables
        self.selected_recipes = {}
        self.recipe_instances = {}
        self.update_recipe_list()

        # Add point editing mode variable
        self.point_edit_mode = "move"  # "move", "add", or "remove"

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is not None:
                self.display_image = self.image.copy()
                # Calculate and store center marker position
                height, width = self.image.shape[:2]
                self.center_marker = (width // 2, height // 2)
                self.display_image_with_contour()
            else:
                QMessageBox.critical(self, "Error", "Failed to load image")

    def capture_image(self):
        # TODO: Implement camera capture
        QMessageBox.information(self, "Not Implemented", "Camera capture not yet implemented")

    def display_image_with_contour(self):
        if self.display_image is not None:
            # Create a copy of the display image
            display = self.display_image.copy()
            
            # Draw center marker if image is loaded
            if self.center_marker is not None:
                x, y = self.center_marker
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1)  # Yellow center marker
                cv2.circle(display, (x, y), 7, (0, 255, 255), 2)  # Yellow circle around marker
            
            # Draw slab contour if it exists
            if self.slab_contour is not None:
                # Draw contour lines
                cv2.drawContours(display, [self.slab_contour], -1, (0, 255, 0), 2)
                
                # Draw contour points
                for point in self.slab_contour:
                    x, y = point[0]
                    cv2.circle(display, (int(x), int(y)), self.point_radius, (0, 0, 255), -1)
            
            # Draw recipe instances
            for recipe_name, recipe in self.recipe_instances.items():
                # Transform recipe points
                transformed_points = self.recipe_manager.transform_recipe_points(recipe)
                
                # Draw recipe points and lines
                points = transformed_points.astype(np.int32)
                is_closed = recipe_name == "Border" # Close the loop only for border recipe
                cv2.polylines(display, [points], isClosed=is_closed, color=(255, 0, 0), thickness=2)
                for point in points:
                    cv2.circle(display, tuple(point), self.point_radius, (255, 0, 0), -1)
            
            # Convert to QImage and display
            height, width = display.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(display.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def detect_slab_contour(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        # Detect contour
        self.slab_contour = self.contour_detector.detect_slab_contour(self.image)
        
        if self.slab_contour is None:
            QMessageBox.warning(self, "Warning", "No suitable contour detected")
            return
            
        self.display_image_with_contour()
        self.confirm_contour_btn.setEnabled(True)

    def confirm_slab_contour(self):
        if self.slab_contour is None:
            QMessageBox.warning(self, "Warning", "No contour to confirm")
            return
            
        # Validate contour
        if len(self.slab_contour) < 3:
            QMessageBox.warning(self, "Warning", "Contour must have at least 3 points")
            return
            
        # Calculate contour area
        area = cv2.contourArea(self.slab_contour)
        if area < 1000:  # Minimum area threshold
            QMessageBox.warning(self, "Warning", "Contour area too small")
            return
            
        # Transform contour to robot coordinates
        processing_z = self.t_input.value() - self.slab_thickness_input.value()
        robot_contour = self.camera_utils.transform_contour(self.slab_contour, processing_z)
        
        if robot_contour is None:
            QMessageBox.warning(self, "Warning", "Failed to transform contour to robot coordinates")
            return
            
        # Store the robot coordinates
        self.robot_contour = robot_contour
        
        # Create mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.slab_contour], -1, 255, -1)
        
        # Apply mask to image
        self.display_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        
        # Fix the contour
        self.contour_fixed = True
        
        # Generate border recipe (use existing margin if available, else default)
        existing_border = self.recipe_manager.get_recipe("Border")
        border_margin = existing_border.margin if existing_border else 50.0 # Default to 50mm
        border_recipe = self.recipe_manager.generate_border_recipe(self.slab_contour, margin=border_margin)
        if border_recipe is not None:
            self.recipe_manager.add_recipe(border_recipe)
            
        # Generate wave pattern recipe (use existing angle if available, else default)
        existing_wave = self.recipe_manager.get_recipe("Wave Pattern")
        wave_angle = existing_wave.angle if existing_wave else 30.0 # Default to 30 deg
        wave_recipe = self.recipe_manager.generate_wave_recipe(self.slab_contour, angle=wave_angle)
        if wave_recipe is not None:
            self.recipe_manager.add_recipe(wave_recipe)
            
        self.update_recipe_list()
        
        # Update display to show confirmed state
        self.display_image_with_contour()
        
        # Enable next steps
        self.generate_program_btn.setEnabled(True)
        self.place_recipe_btn.setEnabled(True)
        
        # Disable contour editing
        self.add_point_btn.setEnabled(False)
        self.remove_point_btn.setEnabled(False)
        self.move_point_btn.setEnabled(False)
        
        QMessageBox.information(self, "Success", "Slab contour confirmed and transformed to robot coordinates")

    def generate_program(self):
        if self.slab_contour is None:
            QMessageBox.warning(self, "Warning", "No contour to generate program from")
            return
            
        # Get selected end effectors
        selected_ees = [i+1 for i, checkbox in enumerate(self.ee_checkboxes) if checkbox.isChecked()]
        if not selected_ees:
            QMessageBox.warning(self, "Warning", "Please select at least one end effector")
            return
            
        # Create program configuration
        config = ProgramConfig(
            base_header="base1",
            processing_z=self.t_input.value() - self.slab_thickness_input.value(),
            safe_z=self.safe_height_input.value(),
            speed=self.speed_input.value(),
            rpm=self.rpm_input.value(),
            force=self.force_input.value(),
            camera_offset=(
                self.camera_x_input.value(),
                self.camera_y_input.value(),
                self.camera_z_input.value()
            ),
            selected_ees=selected_ees
        )
        
        # Generate recipe paths
        recipe_paths = {}
        for recipe in self.recipe_instances.values():
            path_commands = self.recipe_manager.generate_robot_path(
                recipe,
                config.processing_z,
                config.safe_z,
                config.speed
            )
            recipe_paths[recipe.name] = path_commands
        
        # Generate program
        program = self.program_generator.generate_program(
            config,
            self.slab_contour.tolist(),
            recipe_paths
        )
        
        self.export_program_btn.setEnabled(True)

    def export_programs(self):
        # TODO: Implement program export
        QMessageBox.information(self, "Not Implemented", "Program export not yet implemented")

    def on_point_mode_changed(self):
        """Handle point editing mode changes."""
        if self.add_point_btn.isChecked():
            self.point_edit_mode = "add"
        elif self.remove_point_btn.isChecked():
            self.point_edit_mode = "remove"
        else:
            self.point_edit_mode = "move"

    def get_image_coordinates(self, pos):
        """Convert screen coordinates to image coordinates."""
        if self.image is None or self.image_label.pixmap() is None:
            return None
            
        # Get the label size and pixmap size
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        pixmap_size = pixmap.size()
        
        # Get the actual image dimensions
        img_height, img_width = self.image.shape[:2]
        
        # Calculate the actual displayed image size and position
        # This accounts for the aspect ratio preservation
        scale = min(label_size.width() / img_width,
                   label_size.height() / img_height)
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        
        # Calculate the offset to center the image
        x_offset = (label_size.width() - scaled_width) / 2
        y_offset = (label_size.height() - scaled_height) / 2
        
        # Adjust mouse position by subtracting the offset
        adjusted_x = pos.x() - x_offset
        adjusted_y = pos.y() - y_offset
        
        # Convert to image coordinates
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x > scaled_width or adjusted_y > scaled_height:
            return None
            
        x = int(adjusted_x / scale)
        y = int(adjusted_y / scale)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        return (x, y)

    def image_mouse_press(self, event):
        if self.slab_contour is None or self.contour_fixed:
            return
            
        # Get mouse position in image coordinates
        coords = self.get_image_coordinates(event.pos())
        if coords is None:
            return
            
        x, y = coords
        
        if self.point_edit_mode == "move":
            # Check if clicking near a contour point
            for i, point in enumerate(self.slab_contour):
                px, py = point[0]
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < self.nearest_point_threshold:
                    self.dragging_point = i
                    self.drag_start_pos = (x, y)
                    # Store the original point position
                    self.drag_original_pos = (px, py)
                    return
        elif self.point_edit_mode == "add" and event.button() == Qt.MouseButton.LeftButton:
            # Add new point
            self.slab_contour = self.contour_detector.add_point_to_contour(
                self.slab_contour, (x, y)
            )
            self.display_image_with_contour()
        elif self.point_edit_mode == "remove" and event.button() == Qt.MouseButton.LeftButton:
            # Remove nearest point
            self.slab_contour = self.contour_detector.remove_point_from_contour(
                self.slab_contour, (x, y)
            )
            self.display_image_with_contour()

    def image_mouse_move(self, event):
        if self.point_edit_mode != "move" or self.dragging_point is None or self.drag_start_pos is None or self.contour_fixed:
            return
            
        # Get current mouse position in image coordinates
        coords = self.get_image_coordinates(event.pos())
        if coords is None:
            return
            
        x, y = coords
        
        # Calculate the offset from the drag start position
        dx = x - self.drag_start_pos[0]
        dy = y - self.drag_start_pos[1]
        
        # Update the point position by adding the offset to the original position
        new_x = self.drag_original_pos[0] + dx
        new_y = self.drag_original_pos[1] + dy
        
        # Ensure the new position is within image bounds
        img_height, img_width = self.image.shape[:2]
        new_x = max(0, min(new_x, img_width - 1))
        new_y = max(0, min(new_y, img_height - 1))
        
        # Update contour point position
        self.slab_contour[self.dragging_point] = [[new_x, new_y]]
        
        self.display_image_with_contour()

    def image_mouse_release(self, event):
        if self.point_edit_mode == "move":
            self.dragging_point = None
            self.drag_start_pos = None
            self.drag_original_pos = None

    def update_recipe_list(self):
        """Update the recipe list display."""
        self.recipe_list.clear()
        for recipe in self.recipe_manager.recipes.values():
            item = QListWidgetItem()
            widget = QWidget()
            layout = QHBoxLayout()
            
            # Recipe name
            name_label = QLabel(recipe.name)
            layout.addWidget(name_label)
            
            # Margin/Angle input
            if recipe.name == "Wave Pattern":
                param_label = QLabel("Angle (deg):")
                param_input = QSpinBox()
                param_input.setRange(0, 90)
                param_input.setValue(int(recipe.angle))
                param_input.valueChanged.connect(lambda v, r=recipe: self.update_recipe_angle(r, v))
            else:
                param_label = QLabel("Margin (mm):")
                param_input = QSpinBox()
                param_input.setRange(0, 200)
                param_input.setValue(int(recipe.margin))
                param_input.valueChanged.connect(lambda v, r=recipe: self.update_recipe_margin(r, v))
            
            layout.addWidget(param_label)
            layout.addWidget(param_input)
            
            # Quantity input
            quantity_label = QLabel("Quantity:")
            quantity_input = QSpinBox()
            quantity_input.setRange(1, 100)
            quantity_input.setValue(recipe.quantity)
            quantity_input.valueChanged.connect(lambda v, r=recipe: self.on_recipe_quantity_changed(r, v))
            layout.addWidget(quantity_label)
            layout.addWidget(quantity_input)
            
            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda checked, r=recipe: self.delete_recipe(r))
            layout.addWidget(delete_btn)
            
            widget.setLayout(layout)
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, recipe)
            self.recipe_list.addItem(item)
            self.recipe_list.setItemWidget(item, widget)

    def update_recipe_margin(self, recipe: Recipe, margin: int):
        """Update the margin of a recipe."""
        recipe.margin = float(margin)
        self.recipe_manager.save_recipe(recipe)
        # self.update_preview()
        self.display_image_with_contour()

    def update_recipe_angle(self, recipe: Recipe, angle: int):
        """Update the angle of a wave pattern recipe."""
        recipe.angle = float(angle)
        self.recipe_manager.save_recipe(recipe)
        self.display_image_with_contour()

    def on_recipe_quantity_changed(self, recipe, value):
        """Handle recipe quantity changes."""
        recipe.quantity = value
        if recipe.name in self.recipe_instances:
            self.recipe_instances[recipe.name].quantity = value

    def on_recipe_selection_changed(self):
        """Handle recipe selection changes."""
        selected_items = self.recipe_list.selectedItems()
        self.place_recipe_btn.setEnabled(len(selected_items) > 0)
        
        if len(selected_items) == 1:
            recipe = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if recipe is not None:  # Add null check
                self.remove_recipe_btn.setEnabled(recipe.name in self.recipe_instances)
            else:
                self.remove_recipe_btn.setEnabled(False)
        else:
            self.remove_recipe_btn.setEnabled(False)

    def place_recipe(self):
        """Place the selected recipe(s) on the slab."""
        if self.slab_contour is None:
            QMessageBox.warning(self, "Warning", "Please detect and confirm slab contour first")
            return
            
        selected_items = self.recipe_list.selectedItems()
        for item in selected_items:
            recipe = item.data(Qt.ItemDataRole.UserRole)
            
            # Create a new instance of the recipe
            instance = Recipe(
                name=recipe.name,
                points=recipe.points.copy(),
                quantity=recipe.quantity,  # Use the recipe's own quantity
            )
            
            # Check if recipe fits within slab contour
            if not self.recipe_manager.check_recipe_placement(instance, self.slab_contour):
                QMessageBox.warning(self, "Warning", 
                                  f"Recipe '{recipe.name}' does not fit within slab contour")
                continue
                
            self.recipe_instances[recipe.name] = instance
            self.remove_recipe_btn.setEnabled(True)
            
        self.display_image_with_contour()

    def remove_recipe(self):
        """Remove the selected recipe from the slab."""
        selected_items = self.recipe_list.selectedItems()
        if len(selected_items) == 1:
            recipe = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if recipe.name in self.recipe_instances:
                del self.recipe_instances[recipe.name]
                self.remove_recipe_btn.setEnabled(False)
                self.display_image_with_contour()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StoneProcessingApp()
    window.show()
    sys.exit(app.exec()) 