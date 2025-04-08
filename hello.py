import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import pyrealsense2 as rs
import json
from dataclasses import dataclass


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



# Contour should have a list of points connected in a doubly linked list manner for easier manipulation
class Contour:
    def __init__(self, points=None):
        # Initialize with points or an empty list
        self.points = points if points else []
        self.visible = True
        self.color = "blue"
        self.width = 2
        self.line_ids = []
        self.point_ids = []
        
    def draw(self, canvas, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Draw the contour on the canvas"""
        if not self.visible or not self.points:
            return
        
        # Clear previous drawings
        self.clear_from_canvas(canvas)
        
        # Scale points to canvas coordinates
        scaled_points = []
        for point in self.points:
            x = int((point.x * scale_ratio) + (canvas_width - img_width * scale_ratio) / 2)
            y = int((point.y * scale_ratio) + (canvas_height - img_height * scale_ratio) / 2)
            scaled_points.append((x, y))
        
        # Draw lines between points
        for i in range(len(scaled_points) - 1):
            x1, y1 = scaled_points[i]
            x2, y2 = scaled_points[i + 1]
            line_id = canvas.create_line(x1, y1, x2, y2, fill=self.color, width=self.width, tags=f"contour_line_{i}")
            self.line_ids.append(line_id)
        
        # Close the contour if it has at least 3 points
        if len(scaled_points) >= 3:
            x1, y1 = scaled_points[-1]
            x2, y2 = scaled_points[0]
            line_id = canvas.create_line(x1, y1, x2, y2, fill=self.color, width=self.width, tags=f"contour_line_close")
            self.line_ids.append(line_id)
        
        # Draw handle points
        for i, (x, y) in enumerate(scaled_points):
            point_color = "orange"
            point_id = canvas.create_oval(x-5, y-5, x+5, y+5, fill=point_color, tags=f"contour_point_{i}")
            self.point_ids.append(point_id)
        
        return self.line_ids, self.point_ids
    
    def clear_from_canvas(self, canvas):
        """Remove all contour elements from the canvas"""
        for line_id in self.line_ids:
            canvas.delete(line_id)
        for point_id in self.point_ids:
            canvas.delete(point_id)
        self.line_ids = []
        self.point_ids = []
    
    def hide(self, canvas):
        """Hide (remove) the contour from the canvas"""
        self.clear_from_canvas(canvas)
        self.visible = False
    
    def show(self, canvas, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Make the contour visible and draw it"""
        self.visible = True
        self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)
    
    def add_point(self, point, canvas=None, index=None, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Add a point at a specified index or at the end"""
        if index is None:
            self.points.append(point)
        else:
            self.points.insert(index, point)
            
        # Redraw on canvas if provided
        if canvas and self.visible:
            self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)
            
        return len(self.points) - 1
    
    def remove_point(self, index, canvas=None, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Remove a point at the specified index"""
        if 0 <= index < len(self.points) and len(self.points) > 3:  # Maintain at least 3 points for a valid contour
            removed_point = self.points.pop(index)
            
            # Redraw on canvas if provided
            if canvas and self.visible:
                self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)
                
            return True, removed_point
        return False, None
    
    def move_point(self, index, new_x, new_y, canvas=None, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Move a point to a new position"""
        if 0 <= index < len(self.points):
            old_point = self.points[index]
            self.points[index] = Point(new_x, new_y)
            
            # If canvas provided, we can just update the specific point and lines rather than redrawing everything
            if canvas and self.visible:
                # Get canvas coordinates
                x = int((new_x * scale_ratio) + (canvas_width - img_width * scale_ratio) / 2)
                y = int((new_y * scale_ratio) + (canvas_height - img_height * scale_ratio) / 2)
                
                # Update point on canvas
                if 0 <= index < len(self.point_ids):
                    canvas.delete(self.point_ids[index])
                    self.point_ids[index] = canvas.create_oval(x-5, y-5, x+5, y+5, fill="orange", tags=f"contour_point_{index}")
                
                # Update lines connected to this point
                self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)
                
            return True, old_point
        return False, None
    
    def get_point(self, index):
        """Get a point at the specified index"""
        if 0 <= index < len(self.points):
            return self.points[index]
        return None
    
    def point_count(self):
        """Return the number of points in the contour"""
        return len(self.points)
    
    def get_bounding_box(self):
        """Get the bounding box of the contour (min_x, min_y, max_x, max_y)"""
        if not self.points:
            return 0, 0, 0, 0
        
        min_x = min(point.x for point in self.points)
        min_y = min(point.y for point in self.points)
        max_x = max(point.x for point in self.points)
        max_y = max(point.y for point in self.points)
        
        return min_x, min_y, max_x, max_y
    
    def set_color(self, color, canvas=None, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Set the color of the contour"""
        self.color = color
        # Update canvas if provided
        if canvas and self.visible:
            self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)
    
    def set_width(self, width, canvas=None, scale_ratio=1.0, canvas_width=0, canvas_height=0, img_width=0, img_height=0):
        """Set the line width of the contour"""
        self.width = width
        # Update canvas if provided
        if canvas and self.visible:
            self.draw(canvas, scale_ratio, canvas_width, canvas_height, img_width, img_height)



@dataclass
class Recipe:
    name: str
    points: list
    iterations: int = 1
    is_selected: bool = False
    is_confirmed: bool = False


class SlabProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stone Slab Processing Application")
        self.root.geometry("1200x800")
        self.root.iconbitmap(default="slab.ico")
        
        self.image = None
        self.original_image = None
        self.contours = []
        self.selected_contour = None
        self.dragging_point = None
        self.dragging_index = None
        self.recipes = []
        self.selected_recipes = []
        self.selected_header = None
        self.z_height = tk.StringVar(value="0")
        self.speed = tk.StringVar(value="100")
        self.rpm = tk.StringVar(value="12000")
        self.force = tk.StringVar(value="50")
        self.safe_height = tk.StringVar()
        
        self.delete_mode = False
        self.add_mode = False
        self.contour_confirmed = False
        
        self.load_recipes()
        
        self.setup_ui()
        
        self.cap = None
        self.realsense_pipeline = None
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
    
    def setup_ui(self):
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image and contour editing
        left_frame = ttk.LabelFrame(main_frame, text="Slab Image & Contour")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(left_frame, bg="gray", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
         
        ttk.Button(btn_frame, text="Capture Image", command=self.capture_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Capture RealSense", command=self.capture_realsense_image).pack(side=tk.LEFT, padx=5)
         
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Detect Contour", command=self.detect_contour).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Confirm Contour", command=self.confirm_contour).pack(side=tk.LEFT, padx=5)
        
        # Add delete points toggle button
        self.delete_btn = ttk.Button(btn_frame, text="Delete Points", command=self.toggle_delete_mode)
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Add 'Add Points' toggle button
        self.add_btn = ttk.Button(btn_frame, text="Add Points", command=self.toggle_add_mode)
        self.add_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Settings and Parameters
        right_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, ipadx=5, ipady=5)
        
        # Recipe selection
        recipe_frame = ttk.LabelFrame(right_frame, text="Recipe Selection")
        recipe_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        self.recipe_listbox = tk.Listbox(recipe_frame, selectmode=tk.MULTIPLE, height=5)
        self.recipe_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        for recipe in self.recipes:
            self.recipe_listbox.insert(tk.END, recipe["name"])
        
        ttk.Button(recipe_frame, text="Apply Recipes", command=self.apply_recipes).pack(padx=5, pady=5)
        
        # Header selection
        header_frame = ttk.LabelFrame(right_frame, text="Base Header")
        header_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        self.header_combo = ttk.Combobox(header_frame, values=["BASE 1", "BASE 2", "BASE 3"])
        self.header_combo.pack(fill=tk.X, padx=5, pady=5)
        self.header_combo.current(0)
        
        # Trajectory parameters
        params_frame = ttk.LabelFrame(right_frame, text="Trajectory Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        ttk.Label(params_frame, text="Z Height (mm):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.z_height).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Speed ($VEL.CP):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.speed).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Spindle RPM:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.rpm).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Spindle Force:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.force).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(params_frame, text="Calculate Safe Height", command=self.calculate_safe_height).grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Safe Height (mm):").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.safe_height).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Export section
        export_frame = ttk.LabelFrame(right_frame, text="Generate & Export")
        export_frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        ttk.Button(export_frame, text="Generate Program", command=self.generate_program).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(export_frame, text="Export Program", command=self.export_program).pack(fill=tk.X, padx=5, pady=5)
        
        # Program preview
        preview_frame = ttk.LabelFrame(right_frame, text="Program Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, ipadx=5, ipady=5)
        
        self.preview_text = tk.Text(preview_frame, height=10, width=30)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def load_recipes(self):
        # Load predefined snake recipes (in a real application, this should be loaded from file?? IDK I should ask)
        self.recipes = [
            {"name": "Square Snake", "points": [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]},
            {"name": "Zigzag Snake", "points": [[0, 0], [20, 20], [40, 0], [60, 20], [80, 0], [100, 20]]},
            {"name": "Spiral Snake", "points": [[50, 50], [60, 50], [60, 60], [50, 60], [50, 50], [70, 50], [70, 70], [40, 70], [40, 40], [80, 40], [80, 80]]}
        ]
    
    def initialize_realsense(self):
        """Initialize the Intel RealSense camera."""
        self.realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.realsense_pipeline.start(config)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start RealSense camera: {e}")
            return False

    def capture_realsense_image(self):
        """Capture an image from the RealSense camera."""
        if self.realsense_pipeline is None:
            if not self.initialize_realsense():
                return
        
        frames = self.realsense_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            messagebox.showerror("Error", "Failed to capture frame from RealSense camera")
            return
        
        # Convert frame to numpy array
        frame_data = np.asanyarray(color_frame.get_data())
        
        # Process and display image
        self.original_image = frame_data.copy()
        self.image = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        self.display_image()
    
    def capture_image(self):

        if self.pipeline is None:
            messagebox.showerror("Error", "RealSense camera is not initialized")
            return

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            messagebox.showerror("Error", "Failed to capture frame from RealSense camera")
            return

        image = np.asanyarray(color_frame.get_data())
        self.original_image = image.copy()
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.display_image()

        # # Initialize camera if not already done
        # if self.cap is None:
        #     self.cap = cv2.VideoCapture(0)
        #     if not self.cap.isOpened():
        #         messagebox.showerror("Error", "Could not open camera")
        #         return
        
        # # Capture frame
        # ret, frame = self.cap.read()
        # if ret:
        #     self.original_image = frame.copy()
        #     self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     self.display_image()
        # else:
        #     messagebox.showerror("Error", "Failed to capture image")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image()
    
    def display_image(self):
        # if self.image is not None:
        #     img = Image.fromarray(self.image)
        #     img = ImageTk.PhotoImage(img)
        #     self.canvas.create_image(400, 300, image=img, anchor=tk.CENTER)
        #     self.canvas.image = img
        if self.image is not None:
            # Resize image to fit canvas if needed
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Ensure we have positive dimensions (needed on some platforms)
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            img_height, img_width = self.image.shape[:2]
            
            # Calculate aspect ratios
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            
            # Use the smaller ratio to ensure the entire image fits
            scale_ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            # Resize the image
            resized_image = cv2.resize(self.image, (new_width, new_height))
            
            # Convert to PhotoImage format for tkinter
            self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            
            # Clear and update canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_image, anchor=tk.CENTER)
            
            # Draw contours if available
            self.draw_contours()
    
    def detect_contour(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded")
            return
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            self.contours = [approx_contour.reshape(-1, 2).tolist()]
            
            self.image = self.original_image.copy()
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image()
            
            messagebox.showinfo("Success", "Contour detected. You can now edit it by dragging the points.")
        else:
            messagebox.showerror("Error", "No contours found in the image")
    
    def draw_contours(self):
        if not self.contours:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_height, img_width = self.image.shape[:2]
        
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scale_ratio = min(width_ratio, height_ratio)
        
        for contour in self.contours:
            scaled_points = []
            
            for i, point in enumerate(contour):
                x = int((point[0] * scale_ratio) + (canvas_width - img_width * scale_ratio) / 2)
                y = int((point[1] * scale_ratio) + (canvas_height - img_height * scale_ratio) / 2)
                scaled_points.append((x, y))
                
                # Change point color based on mode and whether contour is confirmed
                if not self.contour_confirmed:
                    point_color = "orange"  # default color
                    if self.delete_mode:
                        point_color = "red"  # delete mode color
                    elif self.add_mode:
                        point_color = "green"  # add mode color
                    
                    self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=point_color, tags=f"handle_{i}")
                else:
                    # Just draw points without handlers if contour is confirmed
                    self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="gray", tags=f"handle_{i}")
            
            # Draw lines between points
            for i in range(len(scaled_points) - 1):
                x1, y1 = scaled_points[i]
                x2, y2 = scaled_points[i + 1]
                line_id = self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, tags=f"line_{i}")
                
                # For add mode, we need to identify which line is clicked
                if self.add_mode and not self.contour_confirmed:
                    self.canvas.itemconfig(line_id, tags=(f"line_{i}", "contour_line"))
            
            # Draw last line to close the contour if needed
            if len(scaled_points) > 2:
                x1, y1 = scaled_points[-1]
                x2, y2 = scaled_points[0]
                line_id = self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, tags=f"line_{len(scaled_points)-1}")
                if self.add_mode and not self.contour_confirmed:
                    self.canvas.itemconfig(line_id, tags=(f"line_{len(scaled_points)-1}", "contour_line"))
        
        # Draw selected recipes inside the contour
        for recipe_idx, recipe in enumerate(self.selected_recipes):
            # Scale and position recipe points inside the contour
            scaled_recipe_points = self.scale_recipe_to_contour(recipe["points"], self.contours[0], scale_ratio, canvas_width, canvas_height, img_width, img_height)
            
            # Draw the recipe lines
            for i in range(len(scaled_recipe_points) - 1):
                x1, y1 = scaled_recipe_points[i]
                x2, y2 = scaled_recipe_points[i + 1]
                line_tag = f"recipe_{recipe_idx}_line_{i}"
                line = self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2, tags=(line_tag, "recipe_line"))
                
                # For add mode, tag lines that can receive new points
                if self.add_mode and self.contour_confirmed:
                    self.canvas.itemconfig(line, tags=(line_tag, "recipe_line", "recipe_add_line"))
            
            # If contour is confirmed, draw handles for recipe points
            if self.contour_confirmed:
                for i, (x, y) in enumerate(scaled_recipe_points):
                    # Change point color based on mode
                    point_color = "purple"  # default color for recipe points
                    if self.delete_mode:
                        point_color = "red"  # delete mode color
                    elif self.add_mode:
                        point_color = "green"  # add mode color
                        
                    self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=point_color, 
                                            tags=(f"recipe_{recipe_idx}_handle_{i}", "recipe_handle"))
    
    def scale_recipe_to_contour(self, recipe_points, contour, scale_ratio, canvas_width, canvas_height, img_width, img_height):
        """Scale and position recipe points to fit inside the contour."""
        if not contour or not recipe_points:
            return []
        
        # Calculate the bounding box of the contour in image coordinates
        min_x = min([point[0] for point in contour])
        max_x = max([point[0] for point in contour])
        min_y = min([point[1] for point in contour])
        max_y = max([point[1] for point in contour])
        
        contour_width = max_x - min_x
        contour_height = max_y - min_y
        
        # Calculate the bounding box of the recipe
        recipe_min_x = min([point[0] for point in recipe_points])
        recipe_max_x = max([point[0] for point in recipe_points])
        recipe_min_y = min([point[1] for point in recipe_points])
        recipe_max_y = max([point[1] for point in recipe_points])
        
        recipe_width = recipe_max_x - recipe_min_x
        recipe_height = recipe_max_y - recipe_min_y
        
        # Calculate scaling factors to fit recipe inside the contour (with some margin)
        margin = 0.9  # Leave 10% margin
        x_scale = (contour_width * margin) / recipe_width if recipe_width > 0 else 1
        y_scale = (contour_height * margin) / recipe_height if recipe_height > 0 else 1
        
        # Use the smaller scale to maintain aspect ratio
        recipe_scale = min(x_scale, y_scale)
        
        # Calculate offset to center the recipe in the contour
        offset_x = min_x + (contour_width - recipe_width * recipe_scale) / 2
        offset_y = min_y + (contour_height - recipe_height * recipe_scale) / 2
        
        # Scale and position recipe points
        scaled_points = []
        for point in recipe_points:
            # Scale relative to recipe's min coordinates and offset to center in contour
            img_x = offset_x + (point[0] - recipe_min_x) * recipe_scale
            img_y = offset_y + (point[1] - recipe_min_y) * recipe_scale
            
            # Convert to canvas coordinates
            canvas_x = int((img_x * scale_ratio) + (canvas_width - img_width * scale_ratio) / 2)
            canvas_y = int((img_y * scale_ratio) + (canvas_height - img_height * scale_ratio) / 2)
            
            scaled_points.append((canvas_x, canvas_y))
        
        return scaled_points
    
    def on_click(self, event):
        if not self.contours:
            return
            
        if self.contour_confirmed:
            # In confirmed mode, only allow recipe point editing
            self.handle_recipe_click(event)
            return
        
        # Check if we're in add point mode
        if self.add_mode:
            # Find if click is near a contour line to add a point
            for item in self.canvas.find_withtag("contour_line"):
                coords = self.canvas.coords(item)
                if coords:  # Make sure we have valid coordinates
                    # Get line endpoints
                    x1, y1, x2, y2 = coords
                    
                    # Calculate distance from click to line
                    # Using the formula for distance from point to line segment
                    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    if line_len_sq == 0:  # Line is actually a point
                        dist = ((event.x - x1) ** 2 + (event.y - y1) ** 2) ** 0.5
                    else:
                        t = max(0, min(1, ((event.x - x1) * (x2 - x1) + (event.y - y1) * (y2 - y1)) / line_len_sq))
                        proj_x = x1 + t * (x2 - x1)
                        proj_y = y1 + t * (x2 - y1)
                        dist = ((event.x - proj_x) ** 2 + (event.y - proj_y) ** 0.5)
                    
                    # If close enough to the line, add point
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    img_height, img_width = self.image.shape[:2]
                    
                    width_ratio = canvas_width / img_width
                    height_ratio = canvas_height / img_height
                    scale_ratio = min(width_ratio, height_ratio)
                    
                    # Calculate position on the image
                    img_x = int((event.x - (canvas_width - img_width * scale_ratio) / 2) / scale_ratio)
                    img_y = int((event.y - (canvas_height - img_height * scale_ratio) / 2) / scale_ratio)
                    
                    # Get line index from tag to determine where to insert
                    line_tag = self.canvas.gettags(item)[0]
                    line_idx = int(line_tag.split('_')[1])
                    
                    # find the closest point to the click
                    closest_point_idx = 0
                    min_dist = float('inf')
                    for i, point in enumerate(self.contours[0]):
                        dist = ((point[0] - img_x) ** 2 + (point[1] - img_y) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_point_idx = i

                    # Insert the new point after the closest point
                    self.contours[0].insert(closest_point_idx + 1, [img_x, img_y])
                    # self.contours[0].insert(line_idx + 1, [img_x, img_y])
                    
                    # Redraw contour
                    self.display_image()
                    return
            return
            
        # Existing code for delete or drag mode
        handle_id = None
        for item in self.canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5):
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("handle_"):
                    handle_id = int(tag.split("_")[1])
                    break
            if handle_id is not None:
                break
        
        if handle_id is not None:
            if self.delete_mode:
                # Delete the point in delete mode
                if len(self.contours[0]) > 3:  # Keep minimum 3 points to form a polygon
                    self.contours[0].pop(handle_id)
                    # Redraw the contour with the point removed
                    self.display_image()
                else:
                    messagebox.showwarning("Warning", "Cannot delete point. Minimum 3 points required.")
            else:
                # Regular point dragging behavior
                self.dragging_point = (event.x, event.y)
                self.dragging_index = handle_id
    
    def handle_recipe_click(self, event):
        """Handle click events for recipe editing after contour is confirmed."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = self.image.shape[:2]
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scale_ratio = min(width_ratio, height_ratio)
        
        # If in delete mode, check if we need to delete a recipe point
        if self.delete_mode:
            for recipe_idx, recipe in enumerate(self.selected_recipes):
                scaled_points = self.scale_recipe_to_contour(
                    recipe["points"], 
                    self.contours[0], 
                    scale_ratio, 
                    canvas_width, 
                    canvas_height, 
                    img_width, 
                    img_height
                )
                
                for point_idx, (x, y) in enumerate(scaled_points):
                    # Check if click is near this point
                    if abs(event.x - x) <= 5 and abs(event.y - y) <= 5:
                        # Make sure we keep at least 2 points in the recipe
                        if len(recipe["points"]) > 2:
                            self.selected_recipes[recipe_idx]["points"].pop(point_idx)
                            self.display_image()  # Redraw everything
                        else:
                            messagebox.showwarning("Warning", "Cannot delete point. Minimum 2 points required for a recipe path.")
                        return
            return
            
        # If in add mode, check if we need to add a recipe point
        if self.add_mode:
            # Find if click is near a recipe line to add a point
            for recipe_idx, recipe in enumerate(self.selected_recipes):
                scaled_points = self.scale_recipe_to_contour(
                    recipe["points"], 
                    self.contours[0], 
                    scale_ratio, 
                    canvas_width, 
                    canvas_height, 
                    img_width, 
                    img_height
                )
                
                # Check each line segment
                for i in range(len(scaled_points) - 1):
                    x1, y1 = scaled_points[i]
                    x2, y2 = scaled_points[i+1]
                    
                    # Calculate distance from click to line segment
                    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    if line_len_sq == 0:  # Line is actually a point
                        continue
                    
                    t = max(0, min(1, ((event.x - x1) * (x2 - x1) + (event.y - y1) * (y2 - y1)) / line_len_sq))
                    proj_x = x1 + t * (x2 - x1)
                    proj_y = y1 + t * (x2 - y1)
                    dist = ((event.x - proj_x) ** 2 + (event.y - proj_y) ** 0.5)
                    
                    # If close enough to the line segment, add a point
                    if dist <= 5:
                        # Convert from canvas coordinates back to recipe coordinates
                        # Get the contour bounds to calculate inverse scaling
                        contour = self.contours[0]
                        min_x = min([point[0] for point in contour])
                        max_x = max([point[0] for point in contour])
                        min_y = min([point[1] for point in contour])
                        max_y = max([point[1] for point in contour])
                        contour_width = max_x - min_x
                        contour_height = max_y - min_y
                        
                        # Get the recipe bounds
                        recipe = self.selected_recipes[recipe_idx]
                        recipe_min_x = min([point[0] for point in recipe["points"]])
                        recipe_max_x = max([point[0] for point in recipe["points"]])
                        recipe_min_y = min([point[1] for point in recipe["points"]])
                        recipe_max_y = max([point[1] for point in recipe["points"]])
                        recipe_width = recipe_max_x - recipe_min_x
                        recipe_height = recipe_max_y - recipe_min_y
                        
                        # Calculate scaling factors used for the recipe
                        margin = 0.9
                        x_scale = (contour_width * margin) / recipe_width if recipe_width > 0 else 1
                        y_scale = (contour_height * margin) / recipe_height if recipe_height > 0 else 1
                        recipe_scale = min(x_scale, y_scale)
                        
                        # Calculate offset used for the recipe
                        offset_x = min_x + (contour_width - recipe_width * recipe_scale) / 2
                        offset_y = min_y + (contour_height - recipe_height * recipe_scale) / 2
                        
                        # Convert canvas coordinates to image coordinates
                        img_x = (event.x - (canvas_width - img_width * scale_ratio) / 2) / scale_ratio
                        img_y = (event.y - (canvas_height - img_height * scale_ratio) / 2) / scale_ratio
                        
                        # Convert image coordinates back to recipe coordinates
                        recipe_x = ((img_x - offset_x) / recipe_scale) + recipe_min_x
                        recipe_y = ((img_y - offset_y) / recipe_scale) + recipe_min_y
                        
                        # Insert new point
                        self.selected_recipes[recipe_idx]["points"].insert(i+1, [recipe_x, recipe_y])
                        self.display_image()
                        return
            return
                
        # For dragging recipe points when not in add/delete mode
        for recipe_idx, recipe in enumerate(self.selected_recipes):
            scaled_points = self.scale_recipe_to_contour(
                recipe["points"], 
                self.contours[0], 
                scale_ratio, 
                canvas_width, 
                canvas_height, 
                img_width, 
                img_height
            )
            
            for point_idx, (x, y) in enumerate(scaled_points):
                # Check if click is near this point
                if abs(event.x - x) <= 5 and abs(event.y - y) <= 5:
                    # Start dragging this point
                    self.dragging_point = (event.x, event.y)
                    self.dragging_recipe = recipe_idx
                    self.dragging_index = point_idx
                    return
    
    def on_drag(self, event):
        if self.contour_confirmed:
            # Handle recipe point dragging
            if hasattr(self, 'dragging_recipe') and self.dragging_point is not None:
                # Update dragging position
                self.dragging_point = (event.x, event.y)
                # Redraw to show the dragging update in real-time
                self.display_image()
            return
            
        if self.dragging_point is not None and self.dragging_index is not None:
            old_x, old_y = self.dragging_point
            self.canvas.move(f"handle_{self.dragging_index}", event.x - old_x, event.y - old_y)
            
            self.dragging_point = (event.x, event.y)
            
            self.canvas.delete("contour")
            
            handles = []
            for i in range(len(self.contours[0])):
                items = self.canvas.find_withtag(f"handle_{i}")
                if items:
                    coords = self.canvas.coords(items[0])
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2
                    handles.append((center_x, center_y))
            
            for i in range(len(handles) - 1):
                x1, y1 = handles[i]
                x2, y2 = handles[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="contour")

    
    def on_release(self, event):
        if self.contour_confirmed:
            # Handle recipe point release
            if hasattr(self, 'dragging_recipe') and self.dragging_point is not None:
                # Convert canvas coordinates back to recipe coordinates
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img_height, img_width = self.image.shape[:2]
                width_ratio = canvas_width / img_width
                height_ratio = canvas_height / img_height
                scale_ratio = min(width_ratio, height_ratio)
                
                # Get the contour bounds to calculate inverse scaling
                contour = self.contours[0]
                min_x = min([point[0] for point in contour])
                max_x = max([point[0] for point in contour])
                min_y = min([point[1] for point in contour])
                max_y = max([point[1] for point in contour])
                contour_width = max_x - min_x
                contour_height = max_y - min_y
                
                # Get the recipe bounds
                recipe = self.selected_recipes[self.dragging_recipe]
                recipe_min_x = min([point[0] for point in recipe["points"]])
                recipe_max_x = max([point[0] for point in recipe["points"]])
                recipe_min_y = min([point[1] for point in recipe["points"]])
                recipe_max_y = max([point[1] for point in recipe["points"]])
                recipe_width = recipe_max_x - recipe_min_x
                recipe_height = recipe_max_y - recipe_min_y
                
                # Calculate scaling factors used for the recipe
                margin = 0.9
                x_scale = (contour_width * margin) / recipe_width if recipe_width > 0 else 1
                y_scale = (contour_height * margin) / recipe_height if recipe_height > 0 else 1
                recipe_scale = min(x_scale, y_scale)
                
                # Calculate offset used for the recipe
                offset_x = min_x + (contour_width - recipe_width * recipe_scale) / 2
                offset_y = min_y + (contour_height - recipe_height * recipe_scale) / 2
                
                # Convert canvas coordinates to image coordinates
                img_x = (event.x - (canvas_width - img_width * scale_ratio) / 2) / scale_ratio
                img_y = (event.y - (canvas_height - img_height * scale_ratio) / 2) / scale_ratio
                
                # Convert image coordinates back to recipe coordinates
                recipe_x = ((img_x - offset_x) / recipe_scale) + recipe_min_x
                recipe_y = ((img_y - offset_y) / recipe_scale) + recipe_min_y
                
                # Update the recipe point
                self.selected_recipes[self.dragging_recipe]["points"][self.dragging_index] = [recipe_x, recipe_y]
                
                # Reset dragging state
                self.dragging_point = None
                self.dragging_recipe = None
                self.dragging_index = None
                
                # Redraw everything
                self.display_image()
            return
        
        if self.dragging_point is not None and self.dragging_index is not None:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_height, img_width = self.image.shape[:2]
            
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            items = self.canvas.find_withtag(f"handle_{self.dragging_index}")
            if items:
                coords = self.canvas.coords(items[0])
                center_x = (coords[0] + coords[2]) / 2
                center_y = (coords[1] + coords[3]) / 2
                
                img_x = int((center_x - (canvas_width - img_width * scale_ratio) / 2) / scale_ratio)
                img_y = int((center_y - (canvas_height - img_height * scale_ratio) / 2) / scale_ratio)
                
                self.contours[0][self.dragging_index] = [img_x, img_y]
            
            # Reset dragging state
            self.dragging_point = None
            self.dragging_index = None

            self.display_image()
    
    def confirm_contour(self):
        if not self.contours:
            messagebox.showerror("Error", "No contour to confirm")
            return
        
        # Set the confirmed flag
        self.contour_confirmed = True
        
        # Update button labels for recipe editing instead of disabling them
        self.delete_btn.configure(text="Delete Recipe Points")
        self.add_btn.configure(text="Add Recipe Points")
        
        # Reset the mode flags
        self.delete_mode = False
        self.add_mode = False
        
        # Redraw to update the visual appearance
        self.display_image()
        
        messagebox.showinfo("Success", "Contour confirmed. You can now select recipes and edit their points.")
    
    def apply_recipes(self):
        selected_indices = self.recipe_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No recipes selected")
            return
        
        if not self.contours:
            messagebox.showerror("Error", "No contour detected. Please detect a contour first.")
            return
        
        self.selected_recipes = [self.recipes[i] for i in selected_indices]
        
        self.display_image()
        
        messagebox.showinfo("Success", f"{len(self.selected_recipes)} recipes applied")
    
    def calculate_safe_height(self):
        try:
            z_value = float(self.z_height.get())
            safe_z = z_value + 100  # Safe height is Z + 100mm
            self.safe_height.set(str(safe_z))
        except ValueError:
            messagebox.showerror("Error", "Invalid Z height value")
    
    def generate_program(self):
        # Validate inputs
        if not self.selected_recipes:
            messagebox.showerror("Error", "No recipes selected")
            return
        
        # Make sure we have a contour to scale the recipes to
        if not self.contours:
            messagebox.showerror("Error", "No contour detected. Please detect a contour first.")
            return
            
        try:
            z_height = float(self.z_height.get())
            speed = float(self.speed.get())
            rpm = float(self.rpm.get())
            force = float(self.force.get())
            safe_height = float(self.safe_height.get()) if self.safe_height.get() else z_height + 100
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values")
            return
        
        # Get selected header
        header = self.header_combo.get()
        
        # Generate the program
        program = []
        
        # Add header based on selection
        if header == "BASE 1":
            program.append("; BASE 1 HEADER")
            program.append("$APO.CVEL = " + str(speed))
            program.append("$APO.CDIS = 3")
        elif header == "BASE 2":
            program.append("; BASE 2 HEADER")
            program.append("$APO.CVEL = " + str(speed))
            program.append("$APO.CDIS = 5")
        else:
            program.append("; BASE 3 HEADER")
            program.append("$APO.CVEL = " + str(speed))
            program.append("$APO.CDIS = 7")
        
        # Add spindle settings
        program.append(f"M3_SPINDLE({rpm}, {force})")
        
        # Add scaled recipe points to the program
        for recipe in self.selected_recipes:
            # Scale recipe to fit inside contour (but in image coordinates)
            scaled_recipe = self.get_scaled_recipe_for_program(recipe["points"], self.contours[0])
            
            for point in scaled_recipe:
                x, y = point
                program.append(f"LIN {{X {x:.4f}, Y {y:.4f}, Z {z_height:.4f}}} C_DIS")
        
        # Add safe height move at the end
        program.append(f"LIN {{Z {safe_height}}} C_DIS ;move to safe height")
        
        # Display the program in the preview
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, "\n".join(program))
        
        messagebox.showinfo("Success", "Program generated successfully")
    
    def get_scaled_recipe_for_program(self, recipe_points, contour):
        """Scale recipe points to fit inside the contour for the program output."""
        if not contour or not recipe_points:
            return []
        
        # Calculate the bounding box of the contour
        min_x = min([point[0] for point in contour])
        max_x = max([point[0] for point in contour])
        min_y = min([point[1] for point in contour])
        max_y = max([point[1] for point in contour])
        
        contour_width = max_x - min_x
        contour_height = max_y - min_y
        
        # Calculate the bounding box of the recipe
        recipe_min_x = min([point[0] for point in recipe_points])
        recipe_max_x = max([point[0] for point in recipe_points])
        recipe_min_y = min([point[1] for point in recipe_points])
        recipe_max_y = max([point[1] for point in recipe_points])
        
        recipe_width = recipe_max_x - recipe_min_x
        recipe_height = recipe_max_y - recipe_min_y
        
        # Calculate scaling factors to fit recipe inside the contour (with some margin)
        margin = 0.9  # Leave 10% margin
        x_scale = (contour_width * margin) / recipe_width if recipe_width > 0 else 1
        y_scale = (contour_height * margin) / recipe_height if recipe_height > 0 else 1
        
        # Use the smaller scale to maintain aspect ratio
        recipe_scale = min(x_scale, y_scale)
        
        # Calculate offset to center the recipe in the contour
        offset_x = min_x + (contour_width - recipe_width * recipe_scale) / 2
        offset_y = min_y + (contour_height - recipe_height * recipe_scale) / 2
        
        # Scale and position recipe points
        scaled_points = []
        for point in recipe_points:
            # Scale relative to recipe's min coordinates and offset to center in contour
            img_x = offset_x + (point[0] - recipe_min_x) * recipe_scale
            img_y = offset_y + (point[1] - recipe_min_y) * recipe_scale
            
            scaled_points.append([img_x, img_y])
        
        return scaled_points
    
    def export_program(self):
        program_text = self.preview_text.get(1.0, tk.END)
        if not program_text.strip():
            messagebox.showerror("Error", "No program to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".src",
            filetypes=[("Source files", "*.src"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, "w") as file:
                file.write(program_text)
            messagebox.showinfo("Success", f"Program exported to {file_path}")

    # Add delete mode toggle function
    def toggle_delete_mode(self):
        # Handle differently based on whether contour is confirmed
        if self.contour_confirmed:
            # For recipe points
            # Turn off add mode if it's on
            if self.add_mode:
                self.add_mode = False
                self.add_btn.configure(text="Add Recipe Points")
            
            self.delete_mode = not self.delete_mode
            if self.delete_mode:
                self.delete_btn.configure(text="Exit Delete Mode")
                self.canvas.config(cursor="no")
            else:
                self.delete_btn.configure(text="Delete Recipe Points")
                self.canvas.config(cursor="")
        else:
            # For contour points
            # Turn off add mode if it's on
            if self.add_mode:
                self.add_mode = False
                self.add_btn.configure(text="Add Points")
                
            self.delete_mode = not self.delete_mode
            if self.delete_mode:
                self.delete_btn.configure(text="Exit Delete Mode")
                # Change cursor to indicate delete mode
                self.canvas.config(cursor="no")
            else:
                self.delete_btn.configure(text="Delete Points")
                # Restore default cursor
                self.canvas.config(cursor="")
        
        # Redraw contours to update visual appearance
        self.display_image()
    
    def toggle_add_mode(self):
        # Handle differently based on whether contour is confirmed
        if self.contour_confirmed:
            # For recipe points
            # Turn off delete mode if it's on
            if self.delete_mode:
                self.delete_mode = False
                self.delete_btn.configure(text="Delete Recipe Points")
            
            self.add_mode = not self.add_mode
            if self.add_mode:
                self.add_btn.configure(text="Exit Add Mode")
                self.canvas.config(cursor="crosshair")
            else:
                self.add_btn.configure(text="Add Recipe Points")
                self.canvas.config(cursor="")
        else:
            # For contour points
            # Turn off delete mode if it's on
            if self.delete_mode:
                self.delete_mode = False
                self.delete_btn.configure(text="Delete Points")
            
            self.add_mode = not self.add_mode
            if self.add_mode:
                self.add_btn.configure(text="Exit Add Mode")
                # Change cursor to indicate add mode
                self.canvas.config(cursor="crosshair")
            else:
                self.add_btn.configure(text="Add Points")
                # Restore default cursor
                self.canvas.config(cursor="")
        
        # Redraw contours to update visual appearance
        self.display_image()

def main():
    root = tk.Tk()
    app = SlabProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()