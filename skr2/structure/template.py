import dataclasses
import abc
from typing import List, Dict, Text, Optional, Tuple, Set, Iterable, Union, Mapping, Callable
import enum
import itertools
import six
import copy
import numpy as np
import functools

def _split_thousands(delimiter, value):
  split = value.split(delimiter)
  return len(split) > 1 and any(map(lambda x: len(x) == 3, split))

def convert_to_float(value):
  """Converts value to a float using a series of increasingly complex heuristics.

  Args:
    value: object that needs to be converted. Allowed types include
      float/int/strings.

  Returns:
    A float interpretation of value.

  Raises:
    ValueError if the float conversion of value fails.
  """
  if isinstance(value, float):
    return value
  if isinstance(value, int):
    return float(value)
  if not isinstance(value, six.string_types):
    raise ValueError("Argument value is not a string. Can't parse it as float")
  sanitized = value

  try:
    # Example: 1,000.7
    if "." in sanitized and "," in sanitized:
      return float(sanitized.replace(",", ""))
    # 1,000
    if "," in sanitized and _split_thousands(",", sanitized):
      return float(sanitized.replace(",", ""))
    # 5,5556
    if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        ",", sanitized):
      return float(sanitized.replace(",", "."))
    # 0.0.0.1
    if sanitized.count(".") > 1:
      return float(sanitized.replace(".", ""))
    # 0,0,0,1
    if sanitized.count(",") > 1:
      return float(sanitized.replace(",", ""))
    return float(sanitized)
  except ValueError:
    # Avoid adding the sanitized value in the error message.
    raise ValueError("Unable to convert value to float")

class EvaluationResult(enum.Enum):
  TRUE = 0
  FALSE = 1
  ERROR = 2

def _bool_to_result(b):
  if b:
    return EvaluationResult.TRUE
  return EvaluationResult.FALSE


def _to_floats(values):
  return [convert_to_float(v) for v in values]


class SynthesizationError(Exception):
  pass

def _format_number(number):
  number = float(number)
  if number.is_integer():
    return str(int(number))
  return f'{number:.2f}'


class SelectClause(abc.ABC):
  """An SQL-like select clause that maps a list of rows to a set of values."""

  @abc.abstractmethod
  def evaluate(self, table, rows):
    Ellipsis

  @abc.abstractmethod
  def verbalize(self):
    Ellipsis



class CountSelectClause(SelectClause):

  def evaluate(self, table, rows):
    return {_format_number(len(rows))}

  def verbalize(self):
    return 'the count'

def _have_rank(values):
  return any([value.rank is not None for value in values])

class Aggregation(enum.Enum):
  """Mostly numeric value aggregations."""
  NONE = 0
  LOWEST = 1
  GREATEST = 2
  FIRST = 3
  LAST = 4
  SUM = 5
  RANGE = 6
  AVERAGE = 7
  FIRST_DIFF = 8
  LAST_DIFF = 9

  def evaluate(self, values):
    if self == Aggregation.FIRST_DIFF:
      if len(values) < 2 or not _have_rank(values):
        return None
      return {_format_number(abs(convert_to_float(values[0].text) - convert_to_float(values[1].text)))}
    if self == Aggregation.LAST_DIFF:
      if len(values) < 2 or not _have_rank(values):
        return None
      return {_format_number(abs(convert_to_float(values[-2].text) - convert_to_float(values[-1].text)))}
    if self == Aggregation.NONE:
      answer = set()
      for v in values:
        if len(v.text) > 0:
          answer.add(v.text)
      if len(answer) > 0:
        return answer
      else:
        return None
    else:
      if len(values) < 2:
        # We are strict and require at leat 2 values for aggregation.
        return None
      if self == Aggregation.FIRST:
        return {values[0].text}
      if self == Aggregation.LAST:
        return {values[-1].text}
      else:
        if not _have_rank(values):
          # Rank is none but numeric operation expected.
          return None
        if self == Aggregation.LOWEST:
          return {min(values, key=lambda value: value.rank).text}
        if self == Aggregation.GREATEST:
          return {max(values, key=lambda value: value.rank).text}
        float_values = _to_floats(value.text for value in values)
        if self == Aggregation.SUM:
          return {_format_number(sum(float_values))}
        if self == Aggregation.AVERAGE:
          return {_format_number(np.mean(float_values))}
        if self == Aggregation.RANGE:
          return {_format_number(max(float_values) - min(float_values))}

        else:
          raise ValueError(f'Unknown aggregation: {self}')

  def verbalize(self, column_name):
    if self == Aggregation.NONE:
      return column_name
    elif self == Aggregation.FIRST:
      return f'the first {column_name}'
    elif self == Aggregation.LAST:
      return f'the last {column_name}'
    elif self == Aggregation.LOWEST:
      return f'the lowest {column_name}'
    elif self == Aggregation.GREATEST:
      return f'the greatest {column_name}'
    elif self == Aggregation.SUM:
      return f'the total {column_name}'
    elif self == Aggregation.AVERAGE:
      return f'the average {column_name}'
    elif self == Aggregation.RANGE:
      return f'the range of {column_name}'
    elif self == Aggregation.LAST_DIFF:
      return f'the difference of {column_name} for last two'
    elif self == Aggregation.FIRST_DIFF:
      return f'the difference of {column_name} for first two'
    else:
      raise ValueError(f'Unknown aggregation: {self}')

@dataclasses.dataclass(frozen=True)
class ValueAggregationClause(SelectClause):
  """An SQL like select clause à la 'select SUM(COLUMN_NAME)'."""
  aggregation: Aggregation
  column_name: Text
  column: Optional

  def evaluate(self, table, rows):
    values = [row[self.column.column_id] for row in rows]
    return self.aggregation.evaluate(values)

  def verbalize(self):
    return self.aggregation.verbalize(self.column_name)


@dataclasses.dataclass(frozen=True)
class TemplateConfig:
  prob_count_aggregation: float = 0.2


class Comparator(enum.Enum):
  """A comparator that can be used in a condition."""
  EQUALS = 0
  GREATER = 2
  LESSER = 3

  def verbalize(self):
    if self == Comparator.EQUALS:
      return 'is'
    elif self == Comparator.LESSER:
      return 'is less than'
    elif self == Comparator.GREATER:
      return 'is greater than'
    else:
      raise ValueError(f'Unknown comparator: {self}')

  def compare(self, left, right):
    if self == Comparator.EQUALS:
      return _bool_to_result(right == left)
    try:
      left = set(_to_floats(left))
      right = set(_to_floats(right))
    except ValueError:
      return EvaluationResult.ERROR
    if not left or not right:
      return EvaluationResult.ERROR
    if self == Comparator.LESSER:
      return _bool_to_result(max(left) < min(right))
    if self == Comparator.GREATER:
      return _bool_to_result(min(left) > max(right))
    raise ValueError(f'Unknown comparator: {self}')

@dataclasses.dataclass(frozen=True)
class WhereClause:
  column_name: str
  cell_value: str
  column: Optional
  comparator: Comparator = Comparator.EQUALS



  def verbalize(self):
    return f'{self.column_name} {self.comparator.verbalize()} {self.cell_value}'

  def filter(self, table, rows):
    return [
        row for row in rows if self.matches(
            row[self.column.column_id].text,
            self.cell_value,
        )
    ]

  def matches(self, value, other_value):
    result = self.comparator.compare(
        {value},
        {other_value},
    )
    if result == EvaluationResult.TRUE:
      return True
    if result == EvaluationResult.FALSE:
      return False
    raise ValueError(f'Error comparing values {result}')


class Expression(abc.ABC):
  """An expression that evaluates to a set of values."""

  @abc.abstractmethod
  def evaluate(self, table):
    Ellipsis

  @abc.abstractmethod
  def verbalize(self):
    Ellipsis


@dataclasses.dataclass(frozen=True)
class ComplexExpression(Expression):
  """A complex expression evaluated against a table.

    All aggregations except COUNT require a 'value'.
  """
  where_clauses: List[WhereClause]
  select_clause: SelectClause

  def evaluate(self, table):
    # TODO Consider combining into a single statement.
    rows = table.contents
    for clause in self.where_clauses:
      try:
        rows = clause.filter(table, rows)
      except ValueError:
        return None

    if not rows:
      return None

    return self.select_clause.evaluate(table, rows)

  def verbalize(self):
    qualifiers = []
    for row in self.where_clauses:
      qualifiers.append(row.verbalize())

    qualification = ''
    if qualifiers:
      qualification = f' when {" and ".join(qualifiers)}'

    message = self.select_clause.verbalize()
    return f'{message}{qualification}'

@dataclasses.dataclass(frozen=True)
class QuestionAnswerPair:
  expression: ComplexExpression
  answer: Optional
  def question(self):
    return f'What is {self.expression.verbalize()}'


def _add_where_clause(expression_fn, table, column, where_clauses):
  column_values = list(row[column.column_id].text for row in table.contents)
  pairs = list(itertools.product(column_values, Comparator))
  table.rng.shuffle(pairs)

  for column_value, comparator in pairs:
    if len(column_value) == 0:
      continue
    new_expr = expression_fn(
      where_clauses + [WhereClause(
        column_name=column.text,
        cell_value=column_value,
        comparator=comparator,
        column=column)]
    )
    new_values = new_expr.evaluate(table)

    if new_values is None:
      continue
    else:
      return new_expr, new_values

  return None


def _synthesize_where_clause(
      table,
      expression_fn
):
  expr = expression_fn([])
  values = expr.evaluate(table)

  if values is None:
    raise SynthesizationError("Cannot create where clause in initial checking")

  columns = copy.deepcopy(table.columns)
  table.rng.shuffle(columns)
  for column in columns:
    if isinstance(expr.select_clause, ValueAggregationClause) and column.text == expr.select_clause.column.text:
      continue
    result = _add_where_clause(
      expression_fn,
      table, column, expr.where_clauses)
    if result is None:
      continue
    # if table.rng.random() > table.config.prob_stop_value_row_expansion:
    return result # only one where condition
    # expr, values = result
  return expr, values

def _synthesize_count_condition(table):
  def _create_expression(where_clauses):
    return ComplexExpression(
        where_clauses=where_clauses,
        select_clause=CountSelectClause(),
    )
  expr, values = _synthesize_where_clause(
    table, expression_fn=_create_expression)
  return QuestionAnswerPair(
    expression=expr, answer=values)

def _synthesize_expression(table, column):

  def _create_expression(
      where_clauses,
      aggregation,
  ):
    return ComplexExpression(
        where_clauses=where_clauses,
        select_clause=ValueAggregationClause(
          aggregation=aggregation,
          column_name=column.text,
          column=column,
        ),
    )

  aggregations = list(Aggregation)
  table.rng.shuffle(aggregations)

  for aggregation in aggregations:
    expression_fn = functools.partial(
      _create_expression,
      aggregation=aggregation,
    )
    if aggregation == Aggregation.FIRST_DIFF or aggregation == Aggregation.LAST_DIFF:
      expr = expression_fn([])
      values = expr.evaluate(table)
      if values is None:
        continue
      else:
        return QuestionAnswerPair(
            expression=expr, answer=values)
    else:
      try:
        expr, values = _synthesize_where_clause(
          table=table,
          expression_fn=expression_fn,
        )
        return QuestionAnswerPair(
          expression=expr, answer=values)
      except SynthesizationError:
        continue
  raise SynthesizationError('Cannot create expression')

def _synthesize_aggregation_expression(table):
  columns = copy.deepcopy(table.columns)
  table.rng.shuffle(columns)
  for column in columns:
    try:
      return _synthesize_expression(table, column)
    except SynthesizationError:
      continue
  raise SynthesizationError("Couldn't synthesize aggregation")