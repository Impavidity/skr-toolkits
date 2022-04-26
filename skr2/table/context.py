import dataclasses
from skr2.table.table import table_linearization
from skr2.table.tokenizer_magic import SpacyTokenizerMagic


def has_exact_match_alignment(alignments):
  for alignment in alignments:
    if alignment["match_type"] == "EXACT":
      return True

class FocusEntity:
  def __init__(self, text, coordinate, mentions):
    self.text = text
    self.coordinate = coordinate
    self.mentions = mentions

class Mention:
  def __init__(self, text, span):
    self.text = text
    self.span = span

@dataclasses.dataclass(frozen=True)
class ContextConfig:
  filter_with_exact_match: bool = True


class Context:
  def __init__(self,
               config=None,
               text=None,
               page_url=None,
               page_table_index=None,
               annotations=None):
    self.config = config
    self.text = text
    self.page_url = page_url
    self.page_title_index = page_table_index
    self._tokens = None
    self.tokenizer = SpacyTokenizerMagic.get()

  def __repr__(self):
    return self.text

  def set_tokenizer(self, tokenizer):
    self.tokenizer = tokenizer

  @property
  def tokens(self):
    if self._tokens is None:
      self._tokens = self.tokenizer.tokenize(self.text)
    return self._tokens



  def _create_focus_spans(self, annotations):
    focus_spans = []
    highlighted_cells = self.annotations
    row_id_to_cell = {}
    for highlighted_cell in highlighted_cells:
      row_idx = highlighted_cell["row_idx"]
      col_idx = highlighted_cell["col_idx"]
      if has_exact_match_alignment(highlighted_cell["alignment"]):
        if row_idx not in row_id_to_cell:
          row_id_to_cell[row_idx] = []
        if col_idx not in row_id_to_cell[row_idx]:
          row_id_to_cell[row_idx].append(col_idx)
          mentions = []
          # Rematch the cell in the text to create mentions
          focus_spans.append(FocusEntity(
            text=highlighted_cell["text"],
            coordinate=[row_idx, col_idx],
            mentions=mentions
          ))
    return focus_spans


  @classmethod
  def from_extractor_annotation(cls, d, config):
    """
    Construct class from the output of HTML table extractor
    :param d: json object
    :return: Context class
    """
    # self.annotations = d["highlighted_cells"]
    # self.focus_spans = cls._create_focus_spans(annotations)
    # return cls(
    #   config=config,
    #   text=d["sentence"]["text"],
    #   page_url=d["url"],
    #   page_table_index=d["idx"],
    #   focus_spans=focus_spans
    # )
    pass

  @classmethod
  def from_totto_annotation(cls, d, config):
    """
    Construct class from the ToTTo dataset
    :param d:
    :param config:
    :return:
    """
    return cls(
      config=config,
      text=d["sentence_annotations"][0]["final_sentence"],
      page_url=None,
      page_table_index=None,
      annotations=d["highlighted_cells"]
    )


  def entity_masking(self):
    pass

  def entity_replacement(self, table):
    # We only replace unique spans
    corrupted_text = None
    correction_actions = None

  def to_training_example(self, table):

    return {
      "text": self.text,
      "entities": [entity.text for entity in self.focus_spans],
      "table": table_linearization(table.linearize_table()),
      "page_title": table.page_title,
      "section_titles": table.section_titles,
      "page_url": table.page_url,
      "page_table_index": table.page_table_index,
    }

  def is_valid(self):
    """
    More than two tuple alignment (size of two)
    More than one tuple alignment (size of three)
    """
    row_id_to_cell = {}
    for focus_span in self.focus_spans:
      if focus_span.coordinate[0] not in row_id_to_cell:
        row_id_to_cell[focus_span.coordinate[0]] = []
      if focus_span.coordinate[1] not in row_id_to_cell[focus_span.coordinate[0]]:
        row_id_to_cell[focus_span.coordinate[0]].append(focus_span.coordinate[1])
    size_of_two = 0
    for row_idx in row_id_to_cell:
      if len(row_id_to_cell[row_idx]) >= 3:
        return True
      if len(row_id_to_cell[row_idx]) == 2:
        size_of_two += 1
        if size_of_two >= 2:
          return True
    return False

class Rewriter:
  def __init__(self):
    pass

  def get_contrastive_statement(
          self):
    pass




if __name__ == "__main__":
  from malaina.utils import crash_on_ipy
  import json

  config = ContextConfig()

  # valid_contexts = []
  # invalid_contexts = []
  # with open("/mnt/efs/fs1/relogic-sql-2021/data/examples/wikipedia/en/extraction/4/sentence.jsonl_") as fin:
  #   for idx, line in enumerate(fin):
  #     d = json.loads(line)
  #     context = Context.from_extractor_annotation(d, config)
  #     if context is not None:
  #       if context.is_valid():
  #         valid_contexts.append(context)
  #         if len(valid_contexts) > 10:
  #           break
  #       else:
  #         invalid_contexts.append(context)
  # raise NotImplementedError()

  with open("/home/p8shi/relogic-sql/data/examples/tablekit/ToTTo/totto_data/totto_train_data.jsonl") as fin:
    for idx, line in enumerate(fin):
      d = json.loads(line)
      context = Context.from_totto_annotation(d, config)