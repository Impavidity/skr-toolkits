from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark import SparkContext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from genre.fairseq_model import GENRE
from genre.utils import get_entity_spans_fairseq as get_entity_spans
from pyspark.sql.types import (StructField, StructType, StringType)
import torch

output_file_path = "/tmp/predictions"

def local_extractor(partition_index=0):
  extractor_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
  num_gpus = torch.cuda.device_count()
  has_gpu = torch.cuda.is_available()

  if has_gpu and num_gpus != 0:
    gpu_id = partition_index
    extractor_model.to("cuda:{}".format(gpu_id))
    extractor_model.eval()
  else:
    raise NotImplementedError("You must run on GPU")

  return extractor_model

def local_linker(partition_index=0):
  linker_model = GENRE.from_pretrained("third_party/GENRE/models/fairseq_e2e_entity_linking_aidayago")
  num_gpus = torch.cuda.device_count()
  has_gpu = torch.cuda.is_available()

  if has_gpu and num_gpus != 0:
    gpu_id = partition_index + num_gpus // 2
    linker_model.to("cuda:{}".format(gpu_id))
    linker_model.eval()
  else:
    raise NotImplementedError("You must run on GPU")

  return linker_model


def extract_triplets(text):
  triplets = []
  relation, subject, relation, object_ = '', '', '', ''
  text = text.strip()
  current = 'x'
  for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
    if token == "<triplet>":
      current = 't'
      if relation != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        relation = ''
      subject = ''
    elif token == "<subj>":
      current = 's'
      if relation != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
      object_ = ''
    elif token == "<obj>":
      current = 'o'
      relation = ''
    else:
      if current == 't':
        subject += ' ' + token
      elif current == 's':
        object_ += ' ' + token
      elif current == 'o':
        relation += ' ' + token
  if subject != '' and relation != '' and object_ != '':
    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
  return triplets

def extraction_map(index, partition):
  extractor = local_extractor(partition_index=index)
  linker = local_linker(partition_index=index)
  tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
  batch_size = 1
  batch = []
  batch_meta = []
  count = 0
  gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
  }
  for row in partition:
    row_dict = row.asDict()
    batch.append(row_dict["text"])
    batch_meta.append({
      "doc_id": row_dict["doc_id"],
      "sent_id": row_dict["sent_id"]
    })
    count += 1
    if count == batch_size:
      model_inputs = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors='pt')
      generated_tokens = extractor.generate(
        model_inputs["input_ids"].to(extractor.device),
        attention_mask=model_inputs["attention_mask"].to(extractor.device),
        **gen_kwargs,
      )
      decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

      entity_spans = get_entity_spans(linker, batch)

      for sentence, triplets, spans, meta in zip(batch, decoded_preds, entity_spans, batch_meta):
        ret_rows = extract_triplets(triplets)
        for ret_row in ret_rows:
          ret_row["doc_id"] = meta["doc_id"]
          ret_row["sent_id"] = meta["sent_id"]
          yield Row(**ret_row)
        for span in spans:
          ret_row = {
            "head": sentence[span[0]: span[0]+span[1]],
            "type": "LINK_TO",
            "tail": span[2],
            "doc_id": meta["doc_id"],
            "sent_id": meta["sent_id"],
          }
          yield Row(**ret_row)

      batch = []
      batch_meta = []
      count = 0

  if count != 0:
    model_inputs = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors='pt')
    generated_tokens = extractor.generate(
      model_inputs["input_ids"].to(extractor.device),
      attention_mask=model_inputs["attention_mask"].to(extractor.device),
      **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    entity_spans = get_entity_spans(linker, batch)

    for sentence, triplets, spans, meta in zip(batch, decoded_preds, entity_spans, batch_meta):
      ret_rows = extract_triplets(triplets)
      for ret_row in ret_rows:
        ret_row["doc_id"] = meta["doc_id"]
        ret_row["sent_id"] = meta["sent_id"]
        yield Row(**ret_row)
      for span in spans:
        ret_row = {
          "head": sentence[span[0]: span[0] + span[1]],
          "type": "LINK_TO",
          "tail": span[2],
          "doc_id": meta["doc_id"],
          "sent_id": meta["sent_id"],
        }
        yield Row(**ret_row)


sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([
    Row(doc_id="0", sent_id="0", text='Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'),
    Row(doc_id="0", sent_id="1", text='Phloem is a conductive (or vascular) tissue found in plants.'),
])

partitions = df.coalesce(2)

rdd = partitions.rdd.mapPartitionsWithIndex(lambda index, partition:
                                        extraction_map(index,
                                                    partition))
all_cols = []
all_cols.append('head')
all_cols.append('type')
all_cols.append('tail')
all_cols.append('doc_id')
all_cols.append('sent_id')
new_fields = []
for field_name in all_cols:
  new_fields.append(StructField(field_name, StringType(), True))
string_schema = StructType(new_fields)
spark.createDataFrame(rdd, string_schema).write.mode("overwrite").parquet(output_file_path)

