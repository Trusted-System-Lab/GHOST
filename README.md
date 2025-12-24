# GHOST

Official repo for the paper "Mitigating Gradient Inversion Risks in Language Models via Token Obfuscation" in Asia CCS'26.

## Usage

### Install Dependencies

Create a new conda environment and install the required packages. The following command will create a new conda
environment named `GHOST` with Python 3.12.3 and install the necessary packages.

First, make sure you have conda installed. You can download and 
install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) from their
official websites.

Then, run the following commands to create a new conda environment and activate it.

```bash
conda create -n GHOST python=3.12.3
conda activate GHOST
```

Then, you need to install `torch` for most of the deep learning operations. Our expected version is `2.3.0` with cuda 
`12.1`. However, the compatibility depends on your specific GPU hardware. You need to check
the [official PyTorch website](https://pytorch.org/get-started/previous-versions/) and see if you can install these versions. 
If not, try a different version and it may still work. But this is out of our testing scope.

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

Then, you need to `pip` install the following packages.

```bash
pip install transformers==4.44.2 datasets==2.19.1 evaluate==0.4.3 accelerate==0.30.1 nltk==3.8.1 spacy==3.8.2 absl-py==2.1.0 rouge_score==0.1.2 scikit-learn==1.6.0 bitsandbytes==0.42.0 peft==0.14.0 vec2text==0.0.13 flair==0.15.1
```

After that, you need to download the `en_core_web_sm` model for `spacy`. You can do this by running the following
command:

```bash
python -m spacy download en_core_web_sm
```

You also need to download the `punkt` tokenizer for `nltk`. You can do this by running the following command:

```bash
python -c "import nltk; nltk.download('punkt')"
```

### Experiments