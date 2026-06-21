# SurgVLA-Bench: Towards Evaluating Vision-Language-Action Models for Laparoscopic Surgical Robotics

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

This repository contains the official code and dataset links for the paper **"SurgVLA-Bench: Towards Evaluating Vision-Language-Action Models for Laparoscopic Surgical Robotics"**.

SurgVLA-Bench is a comprehensive benchmark specifically designed for evaluating Vision-Language-Action (VLA) models in laparoscopic surgical contexts. Built upon the SurRoL simulation platform, we construct a hierarchical task taxonomy ranging from atomic actions to complete surgical procedures. 

## 📢 News
* **[2026-06]** Our paper has been provisionally accepted for presentation at **MICCAI 2026**!
* **[2026-06]** Initial release of the codebase and the SurgVLA dataset.

## 📊 Dataset

Unlike general robotics, the surgical domain previously lacked standardized datasets suitable for VLA model training and evaluation. We provide a comprehensive standardized dataset supporting multiple mainstream formats including RLDS and LeRobot formats.

The dataset contains about 800 complete trajectories comprising approximately 40,000 action frames across eight surgical tasks. 

You can access and download our full dataset from the `main` branch of our repository:
🔗 **[Kanden1112/surg-vla-dataset](https://huggingface.co/datasets/Kanden1112/surg-vla-dataset)**

## 🛠️ Installation & Environment Setup

To systematically evaluate different VLA paradigms, we benchmarked autoregressive models (OpenVLA) and flow matching models ($\pi_0$, $\pi_{0.5}$, and SmolVLA). Because these architectures have conflicting dependencies, **we provide three separate Conda environments** for testing different models.

First, clone this repository:
```bash
git clone [https://github.com/exploding4994/SurgVLA.git](https://github.com/exploding4994/SurgVLA.git)
cd SurgVLA# Surg_VLA
