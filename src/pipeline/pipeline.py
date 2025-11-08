import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class PipelineContext:
    """A data class to hold and pass data between pipeline components."""
    def __init__(self, run_name: str, args: Any):
        self.run_name = run_name
        self.args = args
        self.paths = self._setup_paths(run_name)
        self.data: Dict[str, Any] = {}

    def _setup_paths(self, run_name: str) -> Dict[str, str]:
        """Sets up all the necessary directories for a pipeline run."""
        paths = {}
        paths["main_output_dir"] = os.path.join("results", run_name)
        paths["video_output_folder"] = os.path.join(paths["main_output_dir"], "1_video")
        paths["generated_video_path"] = os.path.join(paths["video_output_folder"], "video.mp4")
        
        frames_base_dir = os.path.join(paths["main_output_dir"], "2_frames")
        scene_dir_name = f"{run_name}_scene"
        paths["frames_scene_dir"] = os.path.join(frames_base_dir, scene_dir_name)
        
        paths["masks_dir"] = os.path.join(paths["main_output_dir"], "3_masks")
        paths["overlaid_masks_dir"] = os.path.join(paths["main_output_dir"], "3_masks_overlaid")
        paths["trace_output_dir"] = os.path.join(paths["main_output_dir"], "4_trace_anything_output")
        paths["pointclouds_dir"] = os.path.join(paths["main_output_dir"], "5_pointclouds")
        paths["registration_output_dir"] = os.path.join(paths["main_output_dir"], "6_registration")
        
        # Ensure all directories are created
        for key, path in paths.items():
            # We only want to create directories, not parent paths of files
            if key.endswith("dir") or key.endswith("folder"):
                os.makedirs(path, exist_ok=True)
        
        # Special case for the trace anything scene sub-directory
        os.makedirs(os.path.join(paths["trace_output_dir"], scene_dir_name), exist_ok=True)

        return paths

class PipelineComponent(ABC):
    """Abstract base class for a component in the processing pipeline."""
    def __init__(self, context: PipelineContext):
        self.context = context

    @abstractmethod
    def run(self):
        """Executes the pipeline component."""
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self) -> str:
        """A short, lowercase name for the component, used for CLI matching."""
        return self.name.lower()

class Pipeline:
    """Manages the execution of a series of pipeline components."""
    def __init__(self, context: PipelineContext):
        self.context = context
        self.components: List[PipelineComponent] = []

    def add_component(self, component_class: type):
        """Adds a component to the pipeline."""
        self.components.append(component_class(self.context))

    def run(self, start_at: str, end_at: str):
        """Runs the pipeline from the specified start to end components."""
        all_component_names = [component.short_name for component in self.components]

        try:
            start_index = all_component_names.index(start_at)
            end_index = all_component_names.index(end_at)
        except ValueError as e:
            print(f"Error: Invalid component name provided. '{e.args[0]}'.")
            print(f"Available components are: {all_component_names}")
            return

        if start_index > end_index:
            print(f"Error: Start component '{start_at}' cannot be after end component '{end_at}'.")
            return
            
        components_to_run = self.components[start_index : end_index + 1]
        
        print(f"\nStarting pipeline for run: '{self.context.run_name}'.")
        print(f"Executing components: {[c.name for c in components_to_run]}")

        for component in components_to_run:
            print(f"\n--- Running Component: {component.name} ---")
            component.run()

        print("\nPipeline finished successfully!")
        final_output = self.context.paths.get(f"{end_at}_output_dir", self.context.paths["main_output_dir"])
        print(f"Results for this run can be found in: {final_output}")
