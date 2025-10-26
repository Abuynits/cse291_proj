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


## 2. Running Trace Anything:

Clone / Download Trace Anything repo:

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
