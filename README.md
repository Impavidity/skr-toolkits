# Toolkits for Structured Knowledge Research (skr!)

## Contents
- [How to obtain Wikipedia HTML?](docs/wikipedia-toolkit.md#how-to-crawl-wikipedia-html-pages)
- How to extract tables from Wikipedia? 
- How to align sentences with tables in Wikipedia?
- How to extract triples from text?
- How to link entities to Wikipedia?
- How to write `run.py` for retrieval models?
- [How to dump all Wikidata Properties?](docs/wikidata-predicate.md)

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
pip install -U spacy
python -m spacy download en_core_web_sm
pip install pudb
pip install wandb
pushd third_party
git clone https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
popd
conda install -c pytorch faiss-gpu
pip install nltk
pip install pandas
pip install tabulate
pip install atpbar
pip isntall allennlp=0.9.0
pip uninstall overrides
pip install overrides==3.1.0
```
