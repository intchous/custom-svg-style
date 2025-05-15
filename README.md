# Style Customization of Text-to-Vector Generation with Image Diffusion Priors

[![arXiv](https://img.shields.io/badge/)]()
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://customsvg.github.io/)

## Setup

Create a new conda environment:

```shell
conda create --name svg_diffusion python=3.10
conda activate svg_diffusion
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Install [diffvg](https://github.com/BachiLi/diffvg):

```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
```

### TODO

- [X] Release Stage1 model weights.
- [ ] Release Stage2 model weights.
- [X] Release the inference code.
- [ ] Release training scripts.


### Model Weights

| Model name | Weight |
| ------------------------ | ------------------------ | 
| **Stage1**          | [link](https://drive.google.com/drive/folders/1RTV_lG-xg5_vWTKwqW4zCujgprqJbCr0?usp=drive_link)          |  |
| **Stage2**            |   TBA                  


## Quickstart

1. **Download the pretrained weights**  
   Put the files in the `pretrained` directory.

2. **Launch the Gradio demo**:
    ~~~bash
    CUDA_VISIBLE_DEVICES=0 python -m svg_ldm.gradio_t2svg
    ~~~

    Use the `Generation Count` slider to sample multiple SVGs in one click.

3. **or run the inference script**:
    ~~~bash
    CUDA_VISIBLE_DEVICES=0 python -m svg_ldm.test_ddpm_tr_svgs --test_num 4
    ~~~

    Adjust `test_num` to control how many samples are produced per prompt.

---

> **Tips**  
> - Diffusion sampling is stochastic. Vary the random seed or adjust the sampling settings to explore different outputs.  
> - We recommend DDPM sampling for higher-quality results, which takes about 30 seconds to generate one SVG on an NVIDIA RTX 4090.
> - The model was trained only on simple class labels (see `dataset/label.csv`), so it doesn’t understand complex text prompts. Fine-tune the model on richer SVG datasets can support more detailed prompts.


## Contact

If you have any question, contact us through email at zhangpeiying17@gmail.com.
