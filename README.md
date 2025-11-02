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

## 1. Running WAN 2.1:

Wan 2.1 is accessed through huggingface. The installation was tested on a
Nvidia 3090 GPU with CUDA 12.4.

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

## 5. Configuring Grounded-SAM

Grounded-SAM is available from huggingface. verify that it is working and that 
you have installed sam2 correctly through:
```bash
uv run tests/test_grounded_sam.py
```

## 6. Estimating Trajectories with Trace Anything

Similarly to video generation, the trajectory estimation script uses `hydra`
for configuration. The config files are located in the `config/trajectory_estimation` directory.

You can run the trajectory estimation script from the project directory with:

```bash
CUDA_VISIBLE_DEVICES=5 uv run src/trajectory_estimation/estimator.py --config-path ../../config/trajectory_estimation --config-name base
```
