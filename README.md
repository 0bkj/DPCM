# DPCM: Dual-domain Prototype and Collaborative Memory Network for Multivariate Time Series Anomaly Detection

**Official PyTorch implementation of the paper: DPCM: Dual-domain Prototype and Collaborative Memory Network for Multivariate Time Series Anomaly Detection.**

If **DPCM** helps your research, please consider giving us a ⭐ star!

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)

## 💡 Overview
**DPCM**, a Dual-domain Prototype and Collaborative Memory Network for multivariate time series anomaly detection, which detects anomalies by jointly modeling reconstruction deviation, temporal state-transition deviation, and cross-variable collaboration deviation.

## 🚀 Getting Started

### Installation

Ensure you have a Python 3.10+ environment ready. Install the necessary dependencies via:

```
pip install -r requirements.txt
```

### Data preparation
Regarding the datasets, we follow the TAB data processing pipeline. Download the dataset from TAB (https://github.com/decisionintelligence/TAB) and store it under the data folder, for example, data/tab/multi_ts/MSL.csv.

### Train & Evaluation

- **Model Definition**: Explore the core logic in [here](./models).
- **Reproduction**: Run the provided scripts to replicate our results. For instance, to test on the PSM dataset:

```shell
python main.py --config config/psm/psm.yaml 
```

## 🙏 Acknowledgements
We acknowledge the following open-source projects for their outstanding contributions to the field:

- TAB: Unified Benchmarking of Time Series Anomaly Detection Methods（https://github.com/decisionintelligence/TAB）

## ✉️ Contact

If you have any questions or suggestions, feel free to contact:
- [Shang Wang]  (wangshang@bupt.edu.cn)
- [PengYu Chen]  (penychen@bupt.edu.cn)
- Or describe it in Issues.

