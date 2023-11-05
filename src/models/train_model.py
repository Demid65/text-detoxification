from datasets import load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import os
import warnings

warnings.filterwarnings("ignore")

TRAINING_DATA_FOLDER = 'data/training/'
TRAINING_FOLDER = 'training/'
MODELS_FOLDER = 'models/'

PREFIX = 'detoxify text: '

INPUT_MODEL = 't5-small'
OUTPUT_MODEL = 't5-small-detoxification'
OUTPUT_PATH = os.path.join(MODELS_FOLDER, OUTPUT_MODEL)

TEMP_STAGE_LIMIT = 3

PRETRAIN_LR = 1e-2
TRAIN_LR = 2e-5

PRETRAIN_EPOCH = 4
TRAIN_EPOCH = 3

BATCH_SIZE = 16
FP16 = True

# load split datasets
print(f'loading datasets from {TRAINING_DATA_FOLDER}')
pretrain_dataset = Dataset.load_from_disk(os.path.join(TRAINING_DATA_FOLDER, 'pretrain'))
pretrain_val_dataset = Dataset.load_from_disk(os.path.join(TRAINING_DATA_FOLDER, 'pretrain_val'))
train_dataset = Dataset.load_from_disk(os.path.join(TRAINING_DATA_FOLDER, 'train'))
train_val_dataset = Dataset.load_from_disk(os.path.join(TRAINING_DATA_FOLDER, 'train_val'))
test_dataset = Dataset.load_from_disk(os.path.join(TRAINING_DATA_FOLDER, 'test'))

max_input_length = 128
max_target_length = 128
metric = load_metric("sacrebleu")

# set up the intermediate functions
def preprocess_function(examples):
    inputs = [PREFIX + ex for ex in examples['input']]
    targets = examples['target']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# load model and tokeinzer
print(f'loading original model {INPUT_MODEL}')
tokenizer = T5Tokenizer.from_pretrained(INPUT_MODEL)
model = T5ForConditionalGeneration.from_pretrained(INPUT_MODEL)

# set up the data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# tokenize the datasets
print('tokenizing datasets')
pretrain_processed = pretrain_dataset.map(preprocess_function, batched=True)
pretrain_val_processed = pretrain_val_dataset.map(preprocess_function, batched=True)
train_processed = train_dataset.map(preprocess_function, batched=True)
train_val_processed = train_val_dataset.map(preprocess_function, batched=True)

# set up training arguments
args = Seq2SeqTrainingArguments(
    TRAINING_FOLDER,
    evaluation_strategy = "epoch",
    learning_rate=PRETRAIN_LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=TEMP_STAGE_LIMIT,
    num_train_epochs=PRETRAIN_EPOCH,
    predict_with_generate=True,
    fp16=FP16,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=pretrain_processed,
    eval_dataset=pretrain_val_processed,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# run the first stage of the training
print('training stage one')
trainer.train()

# save the model
print(f'saving model to {OUTPUT_PATH}')
trainer.save_model(OUTPUT_PATH)

tokenizer = T5Tokenizer.from_pretrained(OUTPUT_PATH)
model = T5ForConditionalGeneration.from_pretrained(OUTPUT_PATH)

# set up the data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# set up training arguments
args = Seq2SeqTrainingArguments(
    TRAINING_FOLDER,
    evaluation_strategy = "epoch",
    learning_rate=PRETRAIN_LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=TEMP_STAGE_LIMIT,
    num_train_epochs=PRETRAIN_EPOCH,
    predict_with_generate=True,
    fp16=FP16,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_processed,
    eval_dataset=train_val_processed,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# run the second stage of the training
print('training stage two')
trainer.train()

# save the model
print(f'saving model to {OUTPUT_PATH}')
trainer.save_model(os.path.join(MODELS_FOLDER, OUTPUT_MODEL))

print('done')
