### Environment Setup

This project uses Python and PyTorch for long-horizon time-series forecasting with the ETTh1 dataset.

#### 1. Activate the Conda environment

On the UiS Gorina4 server, activate the Miniconda environment:

```bash
uenv miniconda3-py312
conda activate ele680
```

#### 2. Install Python dependencies

Install the general project dependencies from `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

**Note:** The CUDA-enabled PyTorch package requires the official PyTorch wheel repository. For a new environment, install the PyTorch build using Step 3 below. If `requirements.txt` contains `torch==2.3.1+cu118`, run Step 3 first, or configure pip to use the PyTorch wheel repository when installing requirements.

#### 3. Install PyTorch with CUDA support

The project uses PyTorch 2.3.1 with CUDA 11.8 to support the NVIDIA Tesla P100 GPU.

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Verify dependencies and CUDA support

Check for Python dependency conflicts:

```bash
python -m pip check
```

Verify the PyTorch installation:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA version:', torch.version.cuda); print('Supported GPU architectures:', torch.cuda.get_arch_list())"
```

Verified results on Gorina4:

```text
PyTorch version: 2.3.1+cu118
CUDA available: True
PyTorch CUDA version: 11.8
Supported GPU architectures: ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_37', 'sm_90']
```

The `sm_60` architecture indicates support for the Tesla P100 GPU architecture. An actual GPU computation test is still required before training.

#### 5. GPU configuration

The project uses the following configuration:

| Component | Configuration |
|---|---|
| Server | UiS Gorina4 |
| Python | 3.12.15 |
| Conda environment | `ele680` |
| PyTorch | 2.3.1+cu118 |
| CUDA runtime | 11.8 |
| Assigned GPU | NVIDIA Tesla P100 (GPU 0, 12 GB) |
| GPU compute capability | 6.0 (`sm_60`) |

Group 2 shares GPU 0 with Group 1. GPU-intensive experiments should be coordinated between the groups.

Check GPU usage:

```bash
nvidia-smi -i 0
```

To select GPU 0 before initializing CUDA in a Python process:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

The PyTorch installation includes the required CUDA runtime dependencies. No separate system CUDA environment was activated for this installation.

#### 6. Run dataset exploration

Run the exploration script:

```bash
python explore_etth1.py
```

The project also includes Jupyter notebooks for dataset exploration, preprocessing, and forecasting experiments.