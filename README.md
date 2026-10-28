# Toward Zero-Forgetting: A Training-Free Multimodal Framework for Remote Sensing Class-Incremental Learning

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/ZFCIL-v1.0-darkcyan"></a>
  <a href='https://ieeexplore.ieee.org/document/11662113'><img src='https://img.shields.io/badge/IEEE-11662113-blue.svg'></a>
</p>

🎉The code repository for "Toward Zero-Forgetting: A Training-Free Multimodal Framework for Remote Sensing Class-Incremental Learning" [[paper]](https://ieeexplore.ieee.org/document/11662113) in PyTorch.

## Abstract

Existing class-incremental learning (CIL) methods for remote sensing (RS) scene classification often tend to be training-intensive or rely on static visual features that may inadequately capture the complex interclass similarity and intraclass diversity inherent in RS imagery. Moreover, directly reusing features from models pretrained on natural images (e.g., CLIP) often suffers from a domain gap and limited semantic richness when transferred to the RS domain. In this article, we propose a simple yet effective training-free CIL framework for RS scene classification that leverages multimodal semantic information to build more discriminative category representations. Our framework treats pretrained models as frozen feature extractors to guarantee zero forgetting of the representation space. To enhance semantic discriminability, we employ a large language model (LLM) to generate rich candidate textual descriptions for each class and introduce an image-guided description selection (IGDS) strategy to align semantic information with visual characteristics. Classification is performed using a distance-based metric without any additional training. Extensive experimental results demonstrate that our framework achieves leading performance and superior stability across different session sequences, surpassing both training-based and training-free baseline methods.

## Requirements

### 🗂️ Environment

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. Other dependencies are listed in `requirements.txt`.

```bash
conda create -n zfcil python=3.10
conda activate zfcil
pip install -r requirements.txt
```

`torch`/`torchvision` should be installed with the CUDA build; see the note at the top of `requirements.txt`.

### 🔎 Dataset

Download the datasets from the links below and place them under `data/` (the `ROOT` in each config is `./data/`). Each `train/` and `test/` folder must contain one subfolder per class:

```
data/
├── RSI-CB/
│   ├── train/
│   └── test/
├── UCMerced_LandUse/
│   ├── train/
│   └── test/
├── NWPU-RESISC45/
│   ├── train/
│   └── test/
└── AID/
    ├── train/
    └── test/
```

- **NWPU-RESISC45**: [official page](https://gcheng-nwpu.github.io/#Datasets) · [Hugging Face mirror](https://huggingface.co/datasets/jonathan-roberts1/NWPU-RESISC45)
- **AID**: [official page](https://captain-whu.github.io/AID/) · [Hugging Face mirror](https://huggingface.co/datasets/blanchon/AID)
- **UC Merced Land Use**: [official page](http://weegee.vision.ucmerced.edu/datasets/landuse.html) · [TorchGeo zip](https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip)
- **RSI-CB (RSI-CB256)**: [GitHub (OneDrive / BaiduYun links)](https://github.com/lehaifeng/RSI-CB)

### 🧠 CLIP weights

The CLIP `ViT-B/16` backbone weights are downloaded automatically on the first run and cached at `~/.cache/clip/ViT-B-16.pt`.

If you need to download them manually, the file is served at:

```
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

Place it at `~/.cache/clip/ViT-B-16.pt` (see also the official [openai/CLIP](https://github.com/openai/CLIP) repository).

## 🔑 Running scripts

Each dataset has its own config under `configs/`. Run one dataset with:

```bash
cd /root/IL/ZFCIL
python main.py --config configs/umerced.yaml   # UC Merced
python main.py --config configs/aid.yaml       # AID
python main.py --config configs/rsi.yaml       # RSI-CB
python main.py --config configs/nwpu.yaml      # NWPU-RESISC45
```

Each config runs three seeds (`SEED`) sequentially. To run all four datasets in a row:

```bash
bash run_all.sh
```

Per-task accuracy is printed to stdout and, with `run_all.sh`, also saved to `logs/<dataset>.log`.
