# ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition [ICLR 2025]
[Joseph Fioresi](https://joefioresi718.github.io/), [Ishan Dave](https://daveishan.github.io/), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://joefioresi718.github.io/ALBAR_webpage/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.00156)

Official PyTorch implementation for ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition.

> **Abstract:**
> Bias in machine learning models can lead to unfair decision making, and while it has been well-studied in the image and text domains, it remains underexplored in action recognition. Action recognition models often suffer from background bias (i.e., inferring actions based on background cues) and foreground bias (i.e., relying on subject appearance), which can be detrimental to real-life applications such as autonomous vehicles or assisted living monitoring. While prior approaches have mainly focused on mitigating background bias using specialized augmentations, we thoroughly study both foreground and background bias.
> In this paper, we propose ALBAR, a novel adversarial training method that mitigates foreground and background biases without requiring specialized knowledge of the bias attributes. Our framework applies an adversarial cross-entropy loss to the sampled static clip (where all the frames are the same) and aims to make its class probabilities uniform using a proposed <em>entropy maximization</em> loss. Additionally, we introduce a <em>gradient penalty</em> loss for regularization against the debiasing process. We evaluate our method on established background and foreground bias protocols, setting a new state-of-the-art and strongly improving combined debiasing performance by over <strong>12%</strong> absolute on HMDB51. Furthermore, we identify an issue of background leakage in the existing UCF101 protocol for bias evaluation which provides a shortcut to predict actions and does not provide an accurate measure of the debiasing capability of a model. We address this issue by proposing more fine-grained segmentation boundaries for the actor, where our method also outperforms existing approaches.
## Usage

1. Setup datasets to train/evaluate on.
    - Current dataloader can handle Kinetics400, UCF101, and HMDB51
2. Edit config.py to point paths to correct directory.
3. Edit params_debias.py with appropriate training parameters.
4. Setup and activate environment.

#### Environment creation: 
```bash
conda create -n albar -y python=3.10
conda activate albar
pip install -r requirements.txt
```
#### Run in activated environment:
```bash
accelerate launch train.py
```

## Citation
If you find our work useful for your research, please consider citing our paper using the following BibTeX:
```bibtex
@inproceedings{fioresi2025albar,
  title={ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition},
  author={Fioresi, Joseph and Dave, Ishan Rajendrakumar and Shah, Mubarak},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2025}
}
```
