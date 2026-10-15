### Environment Setup

This project uses Python for long-horizon time-series forecasting with the ETTh1 dataset.

**1. Activate the Conda environment**

```bash
uenv miniconda3-py312
conda activate ele680
```

**2. Install Python dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Verify dependencies**

```bash
python -m pip check
```

**4. Run the dataset exploration**

```bash
python explore_etth1.py
```

GPU-specific dependencies and installation instructions will be documented separately once the forecasting framework is configured.