

**Project Goal:** Develop a graphical user interface (GUI) application using Python and OpenCV 4 (cv2 library) to define processing paths on a stone slab image and generate corresponding KUKA KRL robot arm programs for multiple selected end effectors.

**System Setup & Coordinates:**

*   **Robot & Table:** A robot arm is positioned relative to a table with a uniform surface color.
*   **Camera:** A camera is mounted on the robot arm.
*   **Calibration:** The application **requires pre-existing camera calibration data** (intrinsic matrix, distortion coefficients, and the camera's pose relative to the robot base) to be configured or loaded. This data is essential for transforming 2D image coordinates (pixels) to 3D robot coordinates (mm).
*   **Coordinate System & Heights:**
    *   The robot's base coordinate system is used.
    *   `T`: **Operator Input:** The vertical distance (Z distance) from the robot manipulator's origin/reference point (when at its "starting" or image capture position) down to the table surface (in mm).
    *   `SlabThickness_Z`: **Operator Input:** The thickness of the slab, measured from the table surface to the top surface of the slab (in mm).
    *   `ProcessingZ`: **Calculated:** The absolute Z coordinate (in the robot's base frame) at which processing (e.g., cutting, engraving) will occur. This is calculated as: `ProcessingZ = T - SlabThickness_Z`.
    *   `SafeZ`: **Operator Input/Confirmed:** The absolute Z coordinate (in the robot's base frame) for safe rapid movements above the slab. The system will suggest a default (e.g., `ProcessingZ + 100mm`), but the operator can override it.

**Core Application Requirements:**

**1. User Interface (GUI):**
    *   Implement a user-friendly GUI (e.g., using PyQt, Tkinter, Kivy).
    *   Guide the user logically through the workflow steps.
    *   Include areas for image display, parameter input, selection lists/menus, control buttons, and user feedback/program preview.

**2. Image Acquisition:**
    *   Provide options:
        *   Capture live image from the connected camera via OpenCV.
        *   Upload an existing image file (e.g., JPG, PNG) of the slab on the table.
    *   Display the acquired image in the GUI.

**3. Slab Contour Definition:**
    *   **Automatic Detection:** Use OpenCV (e.g., color thresholding, `cv2.Canny`, `cv2.findContours`) to detect the outer contour of the slab against the uniform table background.
    *   **Visualization & Selection:** Overlay detected contours; allow the user to select the primary slab contour.
    *   **Manual Editing:** Enable interactive adjustment of the selected contour:
        *   Drag existing points (vertices) with the mouse.
        *   Add new points to the contour (e.g., click on an edge).
        *   Remove existing points (e.g., right-click a point).
    *   **Data:** Store contours as ordered lists of (x, y) pixel coordinates.
    *   **Confirmation:** "Confirm Slab Contour" button.

**4. Recipe Management & Placement:**
    *   **Recipe Definition:** Load recipes from external files. Each recipe is defined as an ordered list of XY points representing a path, with a defined start and end point. (Specify file format, e.g., simple CSV, JSON).
    *   **Recipe Selection UI:**
        *   Display a list of available recipes (by name).
        *   Use checkboxes for selection.
        *   For each selected recipe, provide a numeric input field (defaulting to 1) for the operator to specify the **Quantity** (number of times this recipe's program block should be generated).
    *   **Recipe Placement & Adjustment (Visual Instance):**
        *   For each *type* of recipe selected (regardless of quantity), overlay *one* visual instance of its contour onto the slab image.
        *   Allow the operator to interactively reposition (drag) and rotate this visual instance within the slab contour.
        *   Allow adjustment of the contour points of this visual instance (drag, add, remove points). This adjustment defines the geometry for *all* generated instances of this recipe type.
    *   **Confirmation:** "Confirm Recipe Placements" button.

**5. Configuration & Settings Input:**
    *   **Base Header:** Dropdown menu to select a predefined Base Header text block (loaded from configuration/files) to be prepended to each generated program.
    *   **End Effector Selection:**
        *   Display a list/grid of selectable End Effectors (e.g., "EE1" to "EE12", with icons/text). Allow **multiple selections**.
        *   **Effect:** For each End Effector selected, a complete, separate robot program file/section will be generated. The selection must map to a specific change within that program (e.g., setting a tool variable: `DECL TOOL MyTool = $TOOL_DATA[EE_Index]`, or calling a tool change macro `CHANGE_TOOL(EE_Name)` - **Specify the exact code modification required** based on the robot language).
    *   **Camera Offset:**
        *   Input fields for X, Y, Z offset (mm) of the camera's optical center relative to the robot's flange/TCP (specify which). Default to (0, 0, 0).
        *   **Note:** Clarify if this offset is used *directly* in the pixel-to-robot coordinate calculation (e.g., if the image capture pose differs from the processing TCP) or if it's purely informational/for external calibration setup. Assume informational unless specified otherwise.
    *   **Height Parameters (`T`, `SlabThickness_Z`):** Input fields for the operator to enter `T` (Manipulator-to-Table distance) and `SlabThickness_Z` (Slab Thickness) in mm.
    *   **Safe Height (`SafeZ`):** Input field for the absolute `SafeZ` coordinate. Suggest `(T - SlabThickness_Z) + 100mm` as default, allow override.
    *   **Trajectory Parameters:**
        *   **Speed (`$VEL.CP`):** Input field for processing speed (units as required by robot, e.g., mm/s).
        *   **Spindle Settings:** Input fields for Spindle RPM (integer) and Force (numeric, specify units, e.g., kg or N).

**6. Robot Program Generation:**
1. **Trigger:** "Generate Program" button (enabled after confirmations & inputs).
2. **Process:**
    1.  Retrieve confirmed slab contour (pixels).
    2.  Retrieve confirmed recipe contours (pixels) and their quantities.
    3.  Retrieve all configuration settings (`T`, `SlabThickness_Z`, `SafeZ`, Speed, Spindle, Base Header, Selected EEs, Camera Offset).
    4.  Calculate `ProcessingZ = T - SlabThickness_Z`.
    5.  **For each `EE_selected` in the list of selected End Effectors:**
        - Initialize program text for `EE_selected`.
        - Append the selected Base Header content.
        - Append the specific code line(s) determined by `EE_selected` (e.g., `DECL TOOL MyTool = ...` or `CHANGE_TOOL(...)`).
            *   Append speed setting (e.g., `$VEL.CP = [Speed_Value]`).
            *   **For each `Recipe_j` selected with `Quantity_j`:**
                *   Transform the adjusted visual contour points for `Recipe_j` (pixels) into robot base coordinates (X, Y in mm) using the pre-loaded camera calibration data. Let the transformed points be `RobotPath_j`.
                *   **Repeat `Quantity_j` times:**
                    *   Generate rapid move (`LIN` or `PTP`) to `SafeZ` above the start point of `RobotPath_j`.
                    *   Generate linear move (`LIN`) to the start point of `RobotPath_j` at `SafeZ`.
                    *   Append Spindle activation command: `M3_SPINDLE ([RPM_Value], [Force_Value])`
                    *   Generate linear move (`LIN`) to the start point of `RobotPath_j` at `ProcessingZ`.
                    *   Generate sequence of linear moves (`LIN {X, Y, Z=ProcessingZ} C_DIS`) following the points in `RobotPath_j`.
                    *   Generate linear move (`LIN`) retracting to `SafeZ` along the Z-axis.
            *   Append any required end-of-program commands (e.g., Spindle off, M30).
            *   Store or display the generated program text for `EE_selected`.

**7. Program Export:**
    *   Provide an "Export Program(s)" button.
    *   If multiple EEs were selected, save each generated program to a separate file, potentially named suggestively (e.g., `output_EE1.src`, `output_EE3.src`). Allow user to choose directory and base filename. Specify target file extension(s) (e.g., `.SRC`, `.PG`).

**Non-Functional Requirements:**

*   **Modularity:** Design contour/recipe handling and program generation logic flexibly.
*   **Technology:** Python 3.x, OpenCV 4.x (`opencv-python`), suitable GUI library.
*   **Error Handling:** Basic checks for file loading, valid numeric inputs, etc.
*   **Configuration:** Provide a way to load/save camera calibration data and recipe definitions.

**Example Output Snippet (for one recipe, one EE):**

```robotlanguage
; --- Selected Base Header Content ---
$CONFIG ...
; --- End of Base Header ---

; --- EE Specific Setup (Example for EE1) ---
DECL TOOL MyTool = $TOOL_DATA[1]
$TOOL = MyTool
; --- End EE Setup ---

; Settings from GUI
$VEL.CP = 150       ; User defined speed (e.g., 150 mm/s)

; --- Generated Path for Recipe 'CircleCut' (Instance 1 of N) ---

; Move to safe height above start
LIN {Z 125} C_DIS   ; SafeZ (e.g., T=200, SlabZ=50 -> ProcZ=150. SafeZ=ProcZ+100=250? OR SafeZ input directly e.g. 125) -> CLARIFY SafeZ logic slightly more, use user input directly. Assume SafeZ=125 input.
; Approach point at SafeZ
LIN {X 50.1, Y 20.3, Z 125} C_DIS ; Start X,Y of transformed recipe path

; Activate Spindle and Plunge
M3_SPINDLE (550, 220) ; User defined RPM/Force
LIN {X 50.1, Y 20.3, Z 75} C_DIS  ; Plunge to ProcessingZ (e.g., T=200, SlabZ=50 -> ProcZ = 150? No, T=200, SlabZ=50 -> T-SlabZ = 150. Example Z 40 was given before? Let's use T-Z. T=100, Z=60 -> T-Z=40) Let's use example T=100, Z=60 -> ProcessingZ = 40.

; Process contour points at ProcessingZ
LIN {X 55.8, Y 22.9, Z 40} C_DIS
LIN {X 60.2, Y 28.1, Z 40} C_DIS
; ... more points ...
LIN {X 48.5, Y 18.0, Z 40} C_DIS ; Last point of recipe path

; Retract to safe height
LIN {Z 125} C_DIS

; (If Quantity > 1 for 'CircleCut', repeat the 'Generated Path' block)
; (If other recipes selected, generate their blocks similarly)

; --- End of Program ---
M5_SPINDLE_OFF ; Example Spindle Off
M30 ; Example End Program
```
