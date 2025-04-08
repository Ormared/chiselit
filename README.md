KUKA arm 
CV
Fixed distance from arm's ee to the "table"
On "table" there is a slab
Arm is supposed to identify a slab and do a waved motion on a slab.
Camera can wherever is chosen, lightning can vary from bad to good.

Slab size: ~2.5x1.8
Ability to set offset
Camera setup and calibration is a part of the task.


You are  

Task:
You have a robot arm with a mounted camera on it and table(uniform color) with a stone slab on it. The starting position of a robot arm is `x` cm from the table, the depth of the slab(distance from the table to the slabs top side) is `offset` cm.

<task>
Your task is, using CV 3 or 4(whatever you are more familiar with), to:
1. Slab Contour Detection & Editing:
    AI detects slab contours and overlays vectors on the slab image.
    Operator can manually adjust contours by dragging points with the mouse.
    Operator confirms the adjusted contour.

2. Recipe Selection:
    Operator selects one or multiple "snake" recipes after confirming the contour.
    Recipes are visually overlaid on the slab image.

3. Base Header Selection:
    Operator selects "BASE 1" or other available headers from a dropdown menu.
    Selected header contents are copied into the final output file.

4. Trajectory Parameters Input:
    Operator inputs Z height value in a designated field.
    Operator sets speed ($VEL.CP) in a separate input field.
    Spindle settings (M3_SPINDLE RPM & force) are adjustable in the UI.

5. Smart Safe Height Calculation:
    System suggests a safe lift height (e.g., input Z height + 100mm).
    Operator can confirm or adjust the suggested value.

6. Program Generation & Export:
    After confirmation, the program file is generated with all parameters. An example is <export_program_example>
    Export option to save the generated program in a required format.
</task>

<export_program_example>
$APO.CVEL = 100
$APO.CDIS = 3 

LIN {X -87.7816, Y 60.0423} C_DIS
LIN {X -75.3547, Y 49.6149} C_DIS
LIN {X -60.111, Y 44.0666} C_DIS
LIN {X -43.8889, Y 44.0666} C_DIS
LIN {X -32.4619, Y 48.2257} C_DIS
LIN {X -26.4662, Y 37.8409} C_DIS
LIN {X -35.7816, Y 30.0244} C_DIS
LIN {X -43.8926, Y 15.9756} C_DIS
LIN {X -46.0042, Y 4} C_DIS

LIN {Z 140} C_DIS ;move to save height
</export_program_example>