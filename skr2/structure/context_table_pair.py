"""
Concept:
Context & Table
Context: annotations is one argument
annotations has "highlighted_cells" key, which is tuple with row_index and column index
Table has Row and Cell: cell = table.get_cell(row_index, column_index)
FocusEntity is like a joint, it has three arguments: text, coordinate, mentions.
Mention has two argument: text and span
There is no help function on how to obtain this alignment.
"""

from skr2.structure.table import Table, TableConfig, Cell
from skr2.structure.context import Context, ContextConfig
from skr2.table.tokenizer_magic import SpacyTokenizerMagic


class Span:
  def __init__(self, start_index, end_index):
    self.start_index = start_index
    self.end_index = end_index

  def __eq__(self, other):
    return (self.start_index == other.start_index) and \
           (self.end_index == other.end_index)

class FocusEntity:
  def __init__(self, text, coordinate, mentions):
    self.text = text
    self.coordinate = coordinate
    self.mentions = mentions

class Mention:
  def __init__(self, text, span):
    self.text = text
    self.span = span

  def __key(self):
    return self.span.start_index, self.span.end_index

  def __hash__(self):
    return hash(self.__key())

  def __eq__(self, other):
    return self.span == other.span

class ContextTablePair:
  tokenizer = SpacyTokenizerMagic.get()
  def __init__(self,
               config,
               table,
               context,
               focus_entities):
    self.config = config
    self.table = table
    self.context = context
    self.focus_entities = focus_entities
    self._context_mentions = None
    self._focus_entity_group = None

  @property
  def context_spans(self):
    if self._context_mentions is None:
      mentions = set()
      for focus_entity in self.focus_entities:
        for mention in focus_entity.mentions:
          if mention not in mentions:
            mentions.add(mention)
      self._context_mentions = mentions
    return self._context_mentions

  @property
  def focus_entity_group(self):
    # Row ID -> List of Focus Entity
    if self._focus_entity_group is None:
      group = {}
      for focus_entity in self.focus_entities:
        row_index = focus_entity.coordinate[0]
        if row_index not in group:
          group[row_index] = []
        group[row_index].append(focus_entity)
      self._focus_entity_group = group
    return self._focus_entity_group



  @classmethod
  def from_totto_annotation(cls, d, table_config=None, context_config=None):
    def exact_match(x_list, y_list):
      x_str = " ".join([str(t) for t in x_list])
      y_str = " ".join([str(t) for t in y_list])
      if x_str == y_str:
        return True
      else:
        return False

    def has_unique_exact_match_alignment(context: Context, cell: Cell, tokenizer):
      # This unique means unique in text
      if cell.row_index is None or cell.column_index is None:
        return None
      cell_tokens = tokenizer.tokenize(cell.text)
      context_tokens = context.tokens
      n = len(cell_tokens)
      mentions = []
      for i in range(len(context_tokens) - n + 1):
        n_gram_list = context_tokens[i: i+n]
        if exact_match(n_gram_list, cell_tokens):
          # Char Level Index
          span = Span(start_index=n_gram_list[0].start_index, end_index=n_gram_list[-1].end_index)
          mention = Mention(text=" ".join([str(t) for t in n_gram_list]), span=span)
          mentions.append(mention)
      if len(mentions) > 0:
        return FocusEntity(
          text=cell.text,
          coordinate=[cell.row_index + 1, cell.column_index],
          mentions=mentions)
      else:
        return None

    focus_entities = []
    if table_config is None:
      table_config = TableConfig()
    table = Table.from_totto_annotation(d, table_config)
    if context_config is None:
      context_config = ContextConfig()
    context = Context.from_totto_annotation(d, context_config)
    if table_config.no_process is False:
      for highlighted_cell in d["highlighted_cells"]:
        row_index, column_index = highlighted_cell
        cell = table.get_cell(row_index, column_index)
        focus_entity = has_unique_exact_match_alignment(context, cell, cls.tokenizer)
        if focus_entity is not None:
          focus_entities.append(focus_entity)
    return cls(
      config=None,
      table=table,
      context=context,
      focus_entities=focus_entities)

if __name__ == "__main__":
  import json
  from tqdm import tqdm
  import pickle
  from malaina.utils import crash_on_ipy
  highquality = []
  with open("/home/p8shi/relogic-sql/data/examples/tablekit/ToTTo/totto_data/totto_train_data.jsonl") as fin:
    count, total = 0, 0
    for idx, line in tqdm(enumerate(fin)):
      ex = json.loads(line)
      if len(ex["table"]) >= 2:
        context_table_pair = ContextTablePair.from_totto_annotation(ex)
        if len(ex["highlighted_cells"]) == len(context_table_pair.focus_entities):
          count += 1
          highquality.append(context_table_pair)
        total += 1
      if total % 1000 == 0:
        print(count, total)
    print(count, total)
    pickle.dump(highquality, open("highquality", "wb"))


