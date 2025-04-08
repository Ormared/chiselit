import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class RobotArmControlSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Arm Control System")
        self.root.geometry("1200x800")
        
        self.image = None
        self.original_image = None
        self.slab_contour = None
        self.recipe_contours = []
        self.selected_recipes = []
        self.active_contour = None
        self.is_editing = False
        self.selected_point_index = -1
        self.camera_offset = {"x": 0, "y": 0, "z": 0}
        self.start_position_x = 0
        self.slab_depth_offset = 0
        self.safe_height = 0
        self.z_height = 0
        self.speed = 100
        self.spindle_rpm = 1000
        self.spindle_force = 50
        self.selected_base_header = "BASE 1"
        self.selected_end_effectors = []
        
        self.create_frames()
        
        self.create_menu()
        self.create_image_panel()
        self.create_control_panel()
        
        self.bind_events()
        
    def create_frames(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Image panel frame
        self.image_frame = ttk.LabelFrame(self.left_frame, text="Image View")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel frame
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls")
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_menu(self):
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Capture Image", command=self.capture_image)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Export Program", command=self.export_program)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
    def create_image_panel(self):
        # Create canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.image_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_control_panel(self):
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.contour_tab = ttk.Frame(self.notebook)
        self.recipe_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.program_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.contour_tab, text="Contour Detection")
        self.notebook.add(self.recipe_tab, text="Recipe Selection")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.program_tab, text="Program Generation")
        
        self.create_contour_tab()
        
        self.create_recipe_tab()

        self.create_settings_tab()
        
        self.create_program_tab()
        
    def create_contour_tab(self):
        contour_frame = ttk.LabelFrame(self.contour_tab, text="Slab Contour Detection")
        contour_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Contour detection button
        detect_btn = ttk.Button(contour_frame, text="Detect Contours", command=self.detect_contours)
        detect_btn.pack(pady=5)
        
        # Contour editing frame
        edit_frame = ttk.LabelFrame(contour_frame, text="Edit Contours")
        edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Contour editing buttons
        self.edit_mode_var = tk.BooleanVar(value=False)
        edit_check = ttk.Checkbutton(edit_frame, text="Edit Mode", variable=self.edit_mode_var, 
                                     command=self.toggle_edit_mode)
        edit_check.pack(pady=5)
        
        add_point_btn = ttk.Button(edit_frame, text="Add Point", command=self.add_contour_point)
        add_point_btn.pack(pady=5)
        
        remove_point_btn = ttk.Button(edit_frame, text="Remove Point", command=self.remove_contour_point)
        remove_point_btn.pack(pady=5)
        
        confirm_btn = ttk.Button(edit_frame, text="Confirm Contour", command=self.confirm_contour)
        confirm_btn.pack(pady=5)
        
    def create_recipe_tab(self):
        # Recipe selection frame
        recipe_frame = ttk.LabelFrame(self.recipe_tab, text="Recipe Selection")
        recipe_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recipe list
        self.recipe_list_frame = ttk.Frame(recipe_frame)
        self.recipe_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sample recipes (would be loaded from a database or file in a real application)
        recipes = ["Recipe 1", "Recipe 2", "Recipe 3", "Recipe 4", "Recipe 5"]
        
        self.recipe_vars = []
        self.recipe_counts = []
        
        for i, recipe in enumerate(recipes):
            frame = ttk.Frame(self.recipe_list_frame)
            frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar(value=False)
            self.recipe_vars.append(var)
            
            chk = ttk.Checkbutton(frame, text=recipe, variable=var, command=self.update_recipe_selection)
            chk.pack(side=tk.LEFT)
            
            count_var = tk.IntVar(value=0)
            self.recipe_counts.append(count_var)
            
            spinner = ttk.Spinbox(frame, from_=0, to=10, width=5, textvariable=count_var)
            spinner.pack(side=tk.LEFT, padx=5)
            
            confirm_btn = ttk.Button(frame, text="Confirm", command=lambda idx=i: self.confirm_recipe(idx))
            confirm_btn.pack(side=tk.RIGHT)
            
    def create_settings_tab(self):
        # Settings frame
        settings_frame = ttk.LabelFrame(self.settings_tab, text="System Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Base header selection
        header_frame = ttk.Frame(settings_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text="Base Header:").pack(side=tk.LEFT, padx=5)
        
        self.header_var = tk.StringVar(value="BASE 1")
        headers = ["BASE 1", "BASE 2", "BASE 3", "BASE 4"]
        header_combo = ttk.Combobox(header_frame, textvariable=self.header_var, values=headers)
        header_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # End effector selection
        ee_frame = ttk.LabelFrame(settings_frame, text="End Effector Selection")
        ee_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ee_vars = []
        
        for i in range(12):
            var = tk.BooleanVar(value=False)
            self.ee_vars.append(var)
            
            chk = ttk.Checkbutton(ee_frame, text=f"EE{i+1}", variable=var, 
                                 command=self.update_end_effectors)
            chk.grid(row=i//4, column=i%4, padx=5, pady=2, sticky=tk.W)
        
        # Camera position frame
        camera_frame = ttk.LabelFrame(settings_frame, text="Camera Position")
        camera_frame.pack(fill=tk.X, pady=5)
        
        # X offset
        x_frame = ttk.Frame(camera_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="X Offset (mm):").pack(side=tk.LEFT, padx=5)
        self.camera_x_var = tk.DoubleVar(value=0)
        ttk.Entry(x_frame, textvariable=self.camera_x_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Y offset
        y_frame = ttk.Frame(camera_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Y Offset (mm):").pack(side=tk.LEFT, padx=5)
        self.camera_y_var = tk.DoubleVar(value=0)
        ttk.Entry(y_frame, textvariable=self.camera_y_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Z offset
        z_frame = ttk.Frame(camera_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="Z Offset (mm):").pack(side=tk.LEFT, padx=5)
        self.camera_z_var = tk.DoubleVar(value=0)
        ttk.Entry(z_frame, textvariable=self.camera_z_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Robot position parameters
        robot_frame = ttk.LabelFrame(settings_frame, text="Robot Parameters")
        robot_frame.pack(fill=tk.X, pady=5)
        
        # Starting position
        start_frame = ttk.Frame(robot_frame)
        start_frame.pack(fill=tk.X, pady=2)
        ttk.Label(start_frame, text="Starting Position X (mm):").pack(side=tk.LEFT, padx=5)
        self.start_pos_var = tk.DoubleVar(value=0)
        ttk.Entry(start_frame, textvariable=self.start_pos_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Slab depth
        depth_frame = ttk.Frame(robot_frame)
        depth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(depth_frame, text="Slab Depth (mm):").pack(side=tk.LEFT, padx=5)
        self.depth_var = tk.DoubleVar(value=0)
        ttk.Entry(depth_frame, textvariable=self.depth_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Safe height
        height_frame = ttk.Frame(robot_frame)
        height_frame.pack(fill=tk.X, pady=2)
        ttk.Label(height_frame, text="Safe Height (mm):").pack(side=tk.LEFT, padx=5)
        self.safe_height_var = tk.DoubleVar(value=0)
        ttk.Entry(height_frame, textvariable=self.safe_height_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Calculate safe height button
        calc_btn = ttk.Button(height_frame, text="Calculate", command=self.calculate_safe_height)
        calc_btn.pack(side=tk.RIGHT, padx=5)
        
    def create_program_tab(self):
        # Program generation frame
        program_frame = ttk.LabelFrame(self.program_tab, text="Trajectory Parameters")
        program_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Z height
        z_frame = ttk.Frame(program_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="Z Height (mm):").pack(side=tk.LEFT, padx=5)
        self.z_height_var = tk.DoubleVar(value=0)
        ttk.Entry(z_frame, textvariable=self.z_height_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Speed
        speed_frame = ttk.Frame(program_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Speed ($VEL.CP):").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=100)
        ttk.Entry(speed_frame, textvariable=self.speed_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Spindle RPM
        rpm_frame = ttk.Frame(program_frame)
        rpm_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rpm_frame, text="Spindle RPM:").pack(side=tk.LEFT, padx=5)
        self.rpm_var = tk.IntVar(value=1000)
        ttk.Entry(rpm_frame, textvariable=self.rpm_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Spindle Force
        force_frame = ttk.Frame(program_frame)
        force_frame.pack(fill=tk.X, pady=2)
        ttk.Label(force_frame, text="Spindle Force:").pack(side=tk.LEFT, padx=5)
        self.force_var = tk.DoubleVar(value=50)
        ttk.Entry(force_frame, textvariable=self.force_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Program preview
        preview_frame = ttk.LabelFrame(program_frame, text="Program Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.program_text = tk.Text(preview_frame, wrap=tk.WORD, height=10)
        self.program_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Generate button
        gen_btn = ttk.Button(program_frame, text="Generate Program", command=self.generate_program)
        gen_btn.pack(pady=5)
        
    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def capture_image(self):
        # In a real application, this would access the camera
        # For demonstration, we'll just show a dialog to load an image
        self.load_image()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Load the image with OpenCV
                self.original_image = cv2.imread(file_path)
                
                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.image = rgb_image.copy()
                
                # Display the image
                self.display_image()
                
                # Update status bar
                self.status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")
                
                # Clear any previous contours
                self.slab_contour = None
                self.recipe_contours = []
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def display_image(self):
        if self.image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, schedule a redisplay
            self.root.after(100, self.display_image)
            return
            
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = self.image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(self.image, (new_w, new_h))
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        # Calculate position to center the image
        x_center = (canvas_width - new_w) // 2
        y_center = (canvas_height - new_h) // 2
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)
        
        # Store display parameters for coordinate conversions
        self.display_info = {
            "scale": scale,
            "offset_x": x_center,
            "offset_y": y_center,
            "width": new_w,
            "height": new_h
        }
        
        # Redraw contours if they exist
        self.draw_contours()
        
    def detect_contours(self):
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
            
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                messagebox.showinfo("Info", "No contours detected.")
                return
                
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to reduce points
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Store contour
            self.slab_contour = approx_contour
            
            # Create a copy of the original image
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
            
            # Draw the contour on the image
            self.draw_contours()
            
            # Display the image with contours
            self.display_image()
            
            # Update status
            self.status_bar.config(text=f"Contour detected with {len(approx_contour)} points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect contours: {str(e)}")
            
    def draw_contours(self):
        if self.image is None or self.display_info is None:
            return
            
        # Create a copy of the displayed image
        displayed_image = self.image.copy()
        
        # Draw slab contour
        if self.slab_contour is not None:
            # Draw the contour
            cv2.drawContours(displayed_image, [self.slab_contour], 0, (0, 255, 0), 2)
            
            # Draw the contour points
            for i, point in enumerate(self.slab_contour):
                x, y = point[0]
                cv2.circle(displayed_image, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(displayed_image, str(i), (x + 10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw recipe contours
        for i, contour in enumerate(self.recipe_contours):
            color = (0, 0, 255)  # Red color for recipes
            cv2.drawContours(displayed_image, [contour], 0, color, 2)
            
            # Draw recipe label
            if len(contour) > 0:
                x, y = contour[0][0]
                cv2.putText(displayed_image, f"Recipe {i+1}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Update the displayed image
        self.image = displayed_image
        
        # Convert to PhotoImage and update canvas
        h, w = self.image.shape[:2]
        scale = self.display_info["scale"]
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(self.image, (new_w, new_h))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        # Update the image on canvas
        self.canvas.delete("image")
        self.canvas.create_image(
            self.display_info["offset_x"], 
            self.display_info["offset_y"], 
            anchor=tk.NW, 
            image=self.photo,
            tags="image"
        )
        
    def toggle_edit_mode(self):
        self.is_editing = self.edit_mode_var.get()
        
        if self.is_editing:
            self.status_bar.config(text="Edit mode active. Click and drag points to move them.")
        else:
            self.status_bar.config(text="Edit mode deactivated.")
            
    def on_mouse_down(self, event):
        if not self.is_editing or self.slab_contour is None:
            return
            
        # Convert canvas coordinates to image coordinates
        img_x, img_y = self.canvas_to_image(event.x, event.y)
        
        # Find the closest point
        closest_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(self.slab_contour):
            x, y = point[0]
            dist = ((x - img_x) ** 2 + (y - img_y) ** 2) ** 0.5
            
            if dist < closest_dist and dist < 20:  # 20 pixel threshold
                closest_dist = dist
                closest_idx = i
                
        self.selected_point_index = closest_idx
        
        if closest_idx != -1:
            self.status_bar.config(text=f"Selected point {closest_idx}")
        
    def on_mouse_drag(self, event):
        if not self.is_editing or self.selected_point_index == -1:
            return

        img_x, img_y = self.canvas_to_image(event.x, event.y)
        
        self.slab_contour[self.selected_point_index][0] = [img_x, img_y]
        
        self.draw_contours()
        
    def on_mouse_up(self, event):
        if self.selected_point_index != -1:
            self.selected_point_index = -1
            self.status_bar.config(text="Edit mode active")
            
    def canvas_to_image(self, canvas_x, canvas_y):
        """Convert canvas coordinates to original image coordinates"""
        if self.display_info is None:
            return canvas_x, canvas_y
            
        # Adjust for image position on canvas
        rel_x = canvas_x - self.display_info["offset_x"]
        rel_y = canvas_y - self.display_info["offset_y"]
        
        # Scale back to original image dimensions
        img_x = int(rel_x / self.display_info["scale"])
        img_y = int(rel_y / self.display_info["scale"])
        
        return img_x, img_y
        
    def image_to_canvas(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        if self.display_info is None:
            return img_x, img_y
            
        # Scale to displayed size
        scaled_x = img_x * self.display_info["scale"]
        scaled_y = img_y * self.display_info["scale"]
        
        # Adjust for image position on canvas
        canvas_x = scaled_x + self.display_info["offset_x"]
        canvas_y = scaled_y + self.display_info["offset_y"]
        
        return int(canvas_x), int(canvas_y)
        
    def add_contour_point(self):
        if not self.is_editing or self.slab_contour is None:
            messagebox.showinfo("Info", "Please detect a contour and enable edit mode first.")
            return
            
        # Get current mouse position on canvas
        x = self.root.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.root.winfo_pointery() - self.canvas.winfo_rooty()
        
        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image(x, y)
        
        # Find the closest segment to add a point
        best_idx = 0
        best_dist = float('inf')
        
        for i in range(len(self.slab_contour)):
            p1 = self.slab_contour[i][0]
            p2 = self.slab_contour[(i + 1) % len(self.slab_contour)][0]
            
            # Calculate distance from point to line segment
            dist = point_to_line_dist(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
            
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                
        # Add a new point after the closest segment
        new_point = np.array([[[img_x, img_y]]], dtype=np.int32)
        self.slab_contour = np.insert(self.slab_contour, best_idx + 1, new_point, axis=0)
        
        # Redraw
        self.draw_contours()
        
        # Update status
        self.status_bar.config(text=f"Added point at index {best_idx + 1}")
        
    def remove_contour_point(self):
        if not self.is_editing or self.slab_contour is None:
            messagebox.showinfo("Info", "Please detect a contour and enable edit mode first.")
            return
            
        if len(self.slab_contour) <= 3:
            messagebox.showinfo("Info", "Cannot remove more points. Minimum 3 points required.")
            return
            
        # Get current mouse position on canvas
        x = self.root.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.root.winfo_pointery() - self.canvas.winfo_rooty()
        
        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image(x, y)
        
        # Find the closest point to remove
        closest_idx = -1
        closest_dist = float('inf')
        
        for i, point in enumerate(self.slab_contour):
            px, py = point[0]
            dist = ((px - img_x) ** 2 + (py - img_y) ** 2) ** 0.5
            
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
                
        if closest_dist > 20:  # 20 pixel threshold
            messagebox.showinfo("Info", "No point found close to cursor. Please click closer to a point.")
            return
            
        # Remove the point
        self.slab_contour = np.delete(self.slab_contour, closest_idx, axis=0)
        
        # Redraw
        self.draw_contours()
        
        # Update status
        self.status_bar.config(text=f"Removed point at index {closest_idx}")
        
    def confirm_contour(self):
        if self.slab_contour is None:
            messagebox.showinfo("Info", "No contour to confirm.")
            return
            
        # Deactivate edit mode
        self.edit_mode_var.set(False)
        self.is_editing = False
        
        # Store contour as the active contour
        self.active_contour = self.slab_contour.copy()
        
        # Enable recipe tab
        self.notebook.select(1)  # Switch to recipe tab
        
        # Update status
        self.status_bar.config(text="Contour confirmed. Please select recipes.")
        
    def update_recipe_selection(self):
        selected_recipes = []
        
        for i, var in enumerate(self.recipe_vars):
            if var.get():
                selected_recipes.append(i)
                
        self.selected_recipes = selected_recipes
        
        # Update status
        self.status_bar.config(text=f"Selected {len(selected_recipes)} recipes")
        
    def confirm_recipe(self, recipe_idx):
        if not self.recipe_vars[recipe_idx].get():
            messagebox.showinfo("Info", "Please select the recipe first.")
            return
            
        if self.active_contour is None:
            messagebox.showinfo("Info", "Please confirm a slab contour first.")
            return
            
        # Get recipe count
        count = self.recipe_counts[recipe_idx].get()
        
        if count <= 0:
            messagebox.showinfo("Info", "Please set a recipe count greater than 0.")
            return
            
        # Create a contour for the recipe
        # In a real application, this would use the recipe specifications
        # For demonstration, we'll create a smaller version of the active contour
        
        # Calculate centroid of the contour
        M = cv2.moments(self.active_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Find the center by averaging all points
            cx = sum(point[0][0] for point in self.active_contour) // len(self.active_contour)
            cy = sum(point[0][1] for point in self.active_contour) // len(self.active_contour)
        
        # Create a scaled version of the contour
        scale = 0.7  # 70% of original size
        recipe_contour = np.array([
            [[int((point[0][0] - cx) * scale + cx), int((point[0][1] - cy) * scale + cy)]]
            for point in self.active_contour
        ], dtype=np.int32)
        
        # Add to recipe contours
        self.recipe_contours.append(recipe_contour)
        
        # Redraw
        self.draw_contours()
        
        # Update status
        self.status_bar.config(text=f"Recipe {recipe_idx + 1} confirmed with count {count}")
        
    def update_end_effectors(self):
        self.selected_end_effectors = []
        
        for i, var in enumerate(self.ee_vars):
            if var.get():
                self.selected_end_effectors.append(f"EE{i+1}")
                
    def calculate_safe_height(self):
        # Get Z height from inputs
        z_height = self.z_height_var.get()
        slab_depth = self.depth_var.get()
        
        # Calculate safe height (Z height + slab depth + 100mm margin)
        safe_height = z_height + slab_depth + 100
        
        # Update safe height field
        self.safe_height_var.set(safe_height)
        
        # Update status
        self.status_bar.config(text=f"Safe height calculated: {safe_height:.2f} mm")
        
    def generate_program(self):
        if self.active_contour is None:
            messagebox.showinfo("Info", "Please confirm a slab contour first.")
            return
            
        # Get parameter values
        z_height = self.z_height_var.get()
        speed = self.speed_var.get()
        rpm = self.rpm_var.get()
        force = self.force_var.get()
        safe_height = self.safe_height_var.get()
        
        # Generate program
        program = []
        
        # Add header
        program.append(f"$APO.CVEL = {speed}")
        program.append("$APO.CDIS = 3")
        
        # Add selected base header
        program.append(f"BASE: {self.header_var.get()}")
        
        # Add selected end effectors
        if self.selected_end_effectors:
            program.append("TOOL: " + ", ".join(self.selected_end_effectors))
            
        # Add spindle settings
        program.append(f"M3_SPINDLE: {rpm} RPM, {force} FORCE")
        
        # Add movement commands
        program.append("")
        program.append("; Movement commands")
        
        # Add points from contour
        for point in self.active_contour:
            x, y = point[0]
            program.append(f"LIN {{X {x:.4f}, Y {y:.4f}}} C_DIS")
            
        # Add safe height move
        program.append(f"LIN {{Z {safe_height}}} C_DIS ;move to safe height")
        
        # Join program lines
        program_text = "\n".join(program)
        
        # Show in text area
        self.program_text.delete("1.0", tk.END)
        self.program_text.insert("1.0", program_text)
        
        # Update status
        self.status_bar.config(text="Program generated successfully")
        
    def export_program(self):
        if not self.program_text.get("1.0", tk.END).strip():
            messagebox.showinfo("Info", "Please generate a program first.")
            return
            
        # Ask for file save location
        file_path = filedialog.asksaveasfilename(
            title="Save Program",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Write program to file
                with open(file_path, "w") as f:
                    f.write(self.program_text.get("1.0", tk.END))
                    
                # Update status
                self.status_bar.config(text=f"Program exported to: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export program: {str(e)}")
                
def point_to_line_dist(x, y, x1, y1, x2, y2):
    """Calculate the shortest distance from point (x,y) to line segment (x1,y1)-(x2,y2)"""
    # Line segment length squared
    l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    if l2 == 0:
        # Line segment is a point
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        
    # Calculate projection of point onto line
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))
    
    # Calculate closest point on line segment
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    
    # Return distance to closest point
    return ((x - px) ** 2 + (y - py) ** 2) ** 0.5
    
def main():
    root = tk.Tk()
    app = RobotArmControlSystem(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()