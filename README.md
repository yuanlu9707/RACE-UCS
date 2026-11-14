# RACE-UCS

RACE-UCS for **"RACE-UCS: Reconstruction-Guided Individual–Team Representation Learning and Calibrated Exploration for UAV Crowdsourcing"**.

## 📄 Description

Unmanned aerial vehicle (UAV) crowdsourcing (UCS) has emerged as a promising paradigm for spatio-temporal data collection in applications such as disaster response. However, in time-sensitive data collection involving mobile users, partial observability and communication constraints make it challenging for existing methods to effectively model teammate states and team-level task information and facilitate efficient multi-UAV collaborative exploration, limiting collaborative performance and data freshness. Therefore, we propose a multi-agent deep reinforcement learning framework, termed RACE-UCS, to improve the composite sensing efficiency of UCS tasks. Specifically, we first propose a reconstruction-guided individual–team representation learning (RIRL) mechanism, which reconstructs other UAVs' local observations from historical observations and enhances the semantic consistency and agent discriminability of reconstructed representations through individual-level latent representation alignment. RIRL further incorporates team-level representation consistency learning to capture shared task semantics, thereby enhancing multi-UAV collaborative decision-making without explicit inter-UAV communication. Second, we propose a collaborative representation-calibrated exploration (CRE) mechanism, which preserves future-state predictive information while regulating information compression through collaborative representation calibration and distributionally robust constraints. It further constructs intrinsic rewards based on latent distribution discrepancies to promote multi-UAV collaborative exploration. Experimental results on the NCSU and Purdue datasets demonstrate that RACE-UCS outperforms seven baseline methods.

```text
RACE-UCS/
├── README.md
├── requirements.txt
├── source_code/
│   ├── main_DPPO.py              # Training and testing entry point
│   ├── get_args.py               # Environment, RIRL, and CRE parameters
│   ├── algorithms/
│   │   ├── ritrl.py              # Reconstruction-guided individual–team representation learning
│   │   ├── cre.py                # Collaborative representation-calibrated exploration
│   │   ├── algo/                 # PPO agents, runner, and buffers
│   │   └── config/Mobile_IPPO.py  # PPO optimization settings
│   ├── envs/                     # Environment and NCSU/Purdue data
│   ├── env_configs/              # Maps and parallel environment wrappers
│   ├── tools/post/vis.py         # Automatically invoked trajectory renderer
│   └── vis.py                    # Standalone trajectory visualization
└── runs/                         # Generated experiment outputs
```

## 🔧 Dependencies

The dependency file provides a suggested compatibility baseline for Python 3.8 and PyTorch 1.13.1, based on the current imports and APIs.

```bash
git clone https://github.com/yuanlu9707/RACE-UCS.git
cd RACE-UCS
conda create -n race-ucs python=3.8 -y
conda activate race-ucs
python -m pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch build first. For example, the CUDA 11.7 build for Linux/Windows is:

```bash
python -m pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1131) for other builds. GPU execution requires a compatible NVIDIA driver. Use `--device cpu` for CPU execution. The geospatial dependencies are needed for trajectory rendering, which is also called when training saves trajectories. `wandb` is imported by the code even when online logging is muted.

## 💻 Training

Run commands from **`source_code`**; the code uses this working directory for imports and visualization.

```bash
cd source_code
python main_DPPO.py --dataset NCSU --device cuda:0 --group NCSU
```

Train on Purdue:

```bash
python main_DPPO.py --dataset purdue --device cuda:0 --group purdue
```



RIRL (`use_ritrl` in the code) and CRE (`use_cre`) are both enabled by default in the current configuration. Important settings in [`source_code/get_args.py`](source_code/get_args.py) include:

| Parameter | Default | Meaning                                                                |
| --- |---------|------------------------------------------------------------------------|
| `dataset` | `NCSU`  | `NCSU` or `purdue`                                                     |
| `poi_num` | `48`    | Parser default; Purdue automatically selects 59 users                  |
| `uav_num` | `5`     | Number of UAVs                                                         |
| `--aoith n` | `30`    | AoI threshold, measured in time slots                                  |
| `--txth n` | `1`     | Transmission-rate threshold, measured in Mbps                          |
| `--update_num n` | `10`    | Number of antennas; a UAV can serve at most `n` users in one time slot |




Outputs are written to **`runs/<group>/<timestamp-and-settings>/`** at the repository root. Depending on training progress, an experiment directory contains:

```text
EXPERIMENT_DIRECTORY/
├── Models/            # Best network model
├── params.json        # Parameters used for this experiment
├── train_saved_trajs/ # Best UAV trajectories from training episodes
├── train_output.txt   # Metrics associated with the best training model
├── test_output.txt    # Evaluation metrics
└── vis.html           # HTML visualization of the saved UAV trajectories
```

HTML filenames start with `vis_` in the current implementation. Standalone testing saves its outputs under the experiment's `test/` subdirectory.

## 🏁 Testing

From `source_code`, load the best actor using the experiment directory:

```bash
python main_DPPO.py --test --dataset NCSU --device cuda:0 --checkpoint ../runs/NCSU/EXPERIMENT_DIRECTORY
```

Replace `EXPERIMENT_DIRECTORY` with your saved run folder and use the same dataset and model settings as training. Results are saved to `<checkpoint>/test/`.

## 🗺️ Visualization

Generate an HTML visualization of the saved UAV trajectories from `source_code`:

```bash
python tools/post/vis.py --output_dir ../runs/NCSU/EXPERIMENT_DIRECTORY
```

## 👏 Reference

Thanks for:  MetaCS: https://github.com/yuxiaooye/DRL_dyna_AoI.

## 📜 Acknowledgement

This work is supported by the National Key Research and Development Program of China (2021YFF0901205).

Corresponding author: Lei Yang.

## 📧 Contact

For questions, please contact yuanlu9707@163.com.
Thank you.

## Citation

"RACE-UCS: Reconstruction-Guided Individual–Team Representation Learning and Calibrated Exploration for UAV Crowdsourcing"
