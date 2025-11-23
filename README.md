# CSE 291 Project

## 0. Environment Setup

Clone the repository:
```bash
git clone --recursive https://github.com/Abuynits/cse291_proj.git
```

We use UV for the development of this project. To set up the environment, run the following command in the project root directory:

```bash
source setup.sh
```

<b>Notes</b>
- Tested with Ubuntu 24.04 (setup.sh currently only supports linux)
- Tested with CUDA 12.4 and 12.8
- Tested with RTX 3090, 4090
- Requires ~23GB for Trace Anything running on 41 frames.
    - For longer video sequences (>41 frames), we chunk and run TraceAnything independently of other chunks.

>[!NOTE]
> Environment already working?
> Skip to [running the pipeline](#simplified-pipeline)

# Using the Pipeline
Once all models are setup, we can use this easy to use CLI for inference.

## Inference
There are two ways to run inference. One is to directly write a prompt into the CLI:
```bash
python run.py -n name_of_containing_directory --video_prompt 'A toy robot walking across a platform' --target_object 'toy robot'
```

Another is to link it to a JSON of the following format:
```json
{
    "hovering_drone": {
        "prompt": "Person throwing a baseball.",
        "target_object": "baseball"
    },
    "sliding_crate": {
        "prompt": "A dog walking",
        "target_object": "dog"
    }
}
```
This is already linked to prompts/video_generation_prompts/sample_prompts.json by default.
So this simple command will run all subjects in the pipeline.
```bash
python run.py
```

>[!TIP]
> Want to debug certain modules of the pipeline to avoid rerunning?
> The pipeline is made up of 5 parts, in order:
>
> ["videogeneration", "segmentation", "tracing", "pointclouds", "registration"]

A command that will skip the first two steps as well as the last step would then be:
```bash
python run.py --start-at tracing --end-at pointclouds
```

## Adding your own modules
The current pipeline works out-the-box with Wan2.1, SAM2, and TraceAnything.
(And hopefully DiffusionReg, but this can be swapped out arbitrarily.)

To replace a module (particulary a registration method), simply do the following:
- Define your class in src/pipeline/registration.py
    - Inherit off Registration class (an Abstract PipelineComponent; see src/pipeline.py)
    - Implement run() that saves to a '6_registration' folder
- Add module to src/pipeline/__init__.py list, and then import into run.py
- Replace the corresponding module in the create_pipeline function

This generalizes to the other PipelineComponents, but might only want to touch tracing and video_generation.

## Visualization
We can visualize any output in our results/ directory with the following command:
```bash
python visualize.py name_in_results_folder
```
This will launch a Viser application in which we are able to see the trajectories and the pointclouds at any timestep using the GUI.

>[!TIP]
> Trajectories not displaying? After marking the checkbox, press the rebuild button and then play the video.

Alternatively, we have provided a `visualize_pointcloud.ipynb` notebook that visualizes the comparison between the TraceAnything pointclouds and the Euclidean transformed pointclouds from an adjacent timestep using the learning-based registration method, DiffusionReg.