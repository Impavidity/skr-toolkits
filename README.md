# Toolkits for Structured Knowledge Research (skr!)

## Contents
- [How to obtain Wikipedia HTML?](docs/wikipedia-toolkit.md)
- How to extract tables from Wikipedia? 
- How to align sentences with tables in Wikipedia?
- How to extract triples from text?
- How to link entities to Wikipedia?

## Environment Setup

```
conda create --name skr2 python=3.7
source activate skr2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
cd third_party
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
pushd fairseq
pip install --editable ./
popd
git clone https://github.com/facebookresearch/GENRE.git
pushd GENRE
pip install --editable ./
popd
pip install beautifulsoup4
pip install requests
pip install transformers
```
