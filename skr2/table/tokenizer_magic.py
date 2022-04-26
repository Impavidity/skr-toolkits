class Token:
  def __init__(self, text, start_index, end_index):
    self.text = text
    self.start_index = start_index
    self.end_index = end_index

  def __str__(self):
    return self.text

class SpacyTokenizerWrapper:
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def tokenize(self, text):
    _text = self.tokenizer(text)
    return [Token(token.text, start_index=token.idx, end_index=token.idx + len(token.text)) for token in _text]

class SpacyTokenizerMagic:
  tokenizer = None
  @classmethod
  def get(cls):
    if cls.tokenizer is None:
      import spacy
      cls.tokenizer = SpacyTokenizerWrapper(
        tokenizer=spacy.load('en_core_web_sm'))
    return cls.tokenizer


class TabartTokenizerMagic(object):
  tabart_tokenizer = None

  @classmethod
  def get(cls, update_answer_coordinates=True):
    if cls.tabart_tokenizer is None:
      from relogic.pretrainkit.tokenizers.tabart_tokenizer import TaBARTTokenizer
      cls.tabart_tokenizer = TaBARTTokenizer(
        tokenizer_name="facebook/bart-large",
        max_model_input_sizes=500,
        update_answer_coordinates=update_answer_coordinates)
    return cls.tabart_tokenizer

class TapasTokenizerMagic(object):
  tapas_tokenizer = None

  @classmethod
  def get(cls):
    if cls.tapas_tokenizer is None:
      from relogic.tablekit.models.tokenization_tapas import TapasTokenizer
      cls.tapas_tokenizer = TapasTokenizer.from_pretrained(
        'google/tapas-base-finetuned-wtq',
        update_answer_coordinates=True)
    return cls.tapas_tokenizer

class BartTokenizerMagic(object):
  bart_tokenizer = None

  @classmethod
  def get(cls):
    if cls.bart_tokenizer is None:
      from transformers import BartTokenizer
      cls.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    return cls.bart_tokenizer