# CSE 291 Project

## 0. Environment Setup

We use UV for the development of this project. Install UV with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then source the environment:
```bash
uv sync
```

<b>Notes</b>
- Tested with CUDA 12.4 and 12.8
- Tested with RTX 3090, 4090
- Requires ~23GB for Trace Anything running on 41 frames (TODO: chunking)

>[!NOTE]
> Environment already working?
> Skip to [running the pipeline](#simplified-pipeline)

## 1. Running WAN 2.1:

Wan 2.1 is accessed through huggingface.

You should now be able to a test inference script for `Wan 2.1`:
```bash
uv run tests/test_wan.py
```

## 2. Running Trace Anything

Download Trace Anything model weights:
```bash
mkdir -p TraceAnything/checkpoints
wget -O TraceAnything/checkpoints/trace_anything.pt "https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt?download=true"
```

Then from the project directory run:
```bash
uv pip install -e ./TraceAnything
```

You should now be able to a test inference script for `Trace Anything`:
```bash
uv run tests/test_trace_anything.py
```


## 3. Generating Videos with Wan 2.1

The first part of the pipeline is for generating sample videos with WAN 2.1.

You can find the `hydra` config in the `config/video_generation` directory,
 to specify the generation parameters (and the location of prompts).

Prompts are located in the `prompts/video_generation` directory.

Wan2.1 uses a `negative_prompt.txt` file to specify what to not generate.

You can run the video generation script from the project directory with:
```bash
CUDA_VISIBLE_DEVICES=5 uv run src/video_generation/generator.py --config-path ../../config/video_generation --config-name config.yaml
```

TODO: make this multiprocessed if multiple GPUs are available.

## 4. Configuring Grounded-SAM

Grounded-SAM is available from huggingface. verify that it is working and that 
you have installed sam2 correctly through:
```bash
uv run tests/test_grounded_sam.py
```

## 5. Estimating Trajectories with Trace Anything

Similarly to video generation, the trajectory estimation script uses `hydra`
for configuration. The config files are located in the `config/trajectory_estimation` directory.

You can run the trajectory estimation script from the project directory with:

```bash
CUDA_VISIBLE_DEVICES=5 uv run src/trajectory_estimation/estimator.py --config-path ../../config/trajectory_estimation --config-name base
```

# Simplified Pipeline
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
