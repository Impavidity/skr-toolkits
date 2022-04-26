from skr2.table.context_table_pair import ContextTablePair, Span
# from transformers import T5TokenizerFast
from typing import Text, List, Tuple
import dataclasses
from enum import Enum

@dataclasses.dataclass(frozen=True)
class Replacement:
  original_text: Text
  replaced_text: Text
  original_span: Span
  replaced_span: Span


class ContrastiveType(Enum):
  NEGATIVE = 0
  POSITIVE = 1

@dataclasses.dataclass(frozen=True)
class RewriteResult:
  contrastive_type: ContrastiveType
  contrastive_text: Text
  original_text: Text
  annotation: List[Replacement]

class Rewriter:
  def __init__(self):
    self._tokenizer = None # T5TokenizerFast.from_pretrained("t5-base")
    self._detokenizer = None

  def get_random_contrastive_statement(
        self, candidate: ContextTablePair, is_negative_example=True):
    # For each mention in the context,
    # find the span, replace it with another entity to make it negative
    # Find entity tuple (that belong to one row of the table)
    # 1. Replace one of them with other entity in the same column
    # 2. Use language model to do so
    # The rewriting is based on the size of focus entities.
    # If len == 1:
    # If len == 2:
    # If len >= 3:
    # if len(candidate.context_spans) >= 3:
    rng = candidate.table.rng
    replacements = []
    for group_index in candidate.focus_entity_group:
      if len(candidate.focus_entity_group[group_index]) >= 3 and len(candidate.table.contents) > 2:
        replacement_entity = rng.choice(candidate.focus_entity_group[group_index])
        # Replace to what? Same column but in different row.
        column_index = replacement_entity.coordinate[1]
        row_index = replacement_entity.coordinate[0]
        other_row_candidate = list(range(1, len(candidate.table.contents) + 1))
        other_row_candidate.remove(row_index)
        other_row_index = rng.choice(other_row_candidate)
        cell = candidate.table.get_cell(other_row_index, column_index)
        if cell.row_index is None or cell.column_index is None:
          return None
        for mention in replacement_entity.mentions:
          replacements.append((mention.span, cell))

    if len(replacements) > 0:
      # Do replacement and generate annotations
      # This is tokenizer dependent
      # Let's use T5-base by default

      # sort replacement and check overlap
      replacements = sorted(replacements, key=lambda x: x[0].start_index)
      end_index = -1
      for replacement in replacements:
        if replacement[0].start_index > end_index:
          end_index = replacement[0].end_index
          continue
        else:
         return None

      kept_segments = []
      annotations = [] # List of Replacement
      start_index = 0
      text = candidate.context.text
      p = 0
      for replacement in replacements:
        span, cell = replacement
        kept_segments.append(text[start_index:span.start_index])
        p += len(text[start_index:span.start_index])
        start_index = span.end_index
        kept_segments.append(cell.text)
        annotations.append(Replacement(
          original_text=text[span.start_index: span.end_index],
          replaced_text=cell.text,
          original_span=span,
          replaced_span=Span(start_index=p, end_index=p+len(cell.text))))
        p += len(cell.text)
      kept_segments.append(text[span.end_index:])
      assert 2 * len(replacements) + 1 == len(kept_segments)
      contrastive_context = "".join(kept_segments)
      return RewriteResult(
        contrastive_type=ContrastiveType.NEGATIVE,
        contrastive_text=contrastive_context,
        original_text=text,
        annotation=annotations)
    else:
      return None