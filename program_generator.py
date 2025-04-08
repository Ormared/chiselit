from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class ProgramConfig:
    base_header: str
    processing_z: float
    safe_z: float
    speed: float
    rpm: int
    force: float
    camera_offset: tuple[float, float, float]
    selected_ees: List[int]

class ProgramGenerator:
    def __init__(self, header_dir: str = "headers"):
        self.header_dir = Path(header_dir)
        self.headers: Dict[str, str] = {}
        self.load_headers()

    def load_headers(self):
        """Load all header files from the header directory."""
        if not self.header_dir.exists():
            self.header_dir.mkdir(parents=True)
            return

        for header_file in self.header_dir.glob("*.txt"):
            try:
                with open(header_file, 'r') as f:
                    self.headers[header_file.stem] = f.read()
            except Exception as e:
                print(f"Error loading header {header_file}: {e}")

    def get_header(self, name: str) -> Optional[str]:
        """Get a header by name."""
        return self.headers.get(name)

    def get_all_headers(self) -> List[str]:
        """Get all available header names."""
        return list(self.headers.keys())

    def generate_program(self, config: ProgramConfig, 
                        slab_contour: List[tuple[float, float]],
                        recipe_paths: Dict[str, List[str]]) -> str:
        """
        Generate a complete KUKA KRL program.
        
        Args:
            config: Program configuration
            slab_contour: List of (x, y) points defining the slab contour
            recipe_paths: Dictionary mapping recipe names to their path commands
            
        Returns:
            Complete KRL program as string
        """
        program_lines = []
        
        # Add base header
        if config.base_header in self.headers:
            program_lines.append(self.headers[config.base_header])
        program_lines.append("")  # Empty line for readability
        
        # Add camera offset information as comment
        program_lines.append(f"; Camera offset: X={config.camera_offset[0]:.1f}, "
                           f"Y={config.camera_offset[1]:.1f}, "
                           f"Z={config.camera_offset[2]:.1f}")
        program_lines.append("")
        
        # Add program variables
        program_lines.extend([
            "DECL GLOBAL REAL $PROCESSING_Z",
            "DECL GLOBAL REAL $SAFE_Z",
            "DECL GLOBAL REAL $SPEED",
            "DECL GLOBAL INT $RPM",
            "DECL GLOBAL REAL $FORCE",
            "",
            f"$PROCESSING_Z = {config.processing_z:.1f}",
            f"$SAFE_Z = {config.safe_z:.1f}",
            f"$SPEED = {config.speed:.1f}",
            f"$RPM = {config.rpm}",
            f"$FORCE = {config.force:.1f}",
            ""
        ])
        
        # Generate program for each selected end effector
        for ee_index in config.selected_ees:
            program_lines.extend([
                f"; --- Program for End Effector {ee_index} ---",
                f"DECL TOOL MyTool = $TOOL_DATA[{ee_index}]",
                "$TOOL = MyTool",
                ""
            ])
            
            # Add recipe paths
            for recipe_name, path_commands in recipe_paths.items():
                program_lines.extend([
                    f"; --- Path for Recipe: {recipe_name} ---",
                    *path_commands,
                    ""
                ])
            
            # Add end effector specific cleanup
            program_lines.extend([
                "M5_SPINDLE_OFF",
                "M30",
                ""
            ])
        
        return "\n".join(program_lines)

    def save_program(self, program: str, output_file: str):
        """
        Save the generated program to a file.
        
        Args:
            program: Program text to save
            output_file: Path to save the program to
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(program) 