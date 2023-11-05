from datasets import load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import numpy as np
import os
import warnings
import argparse

warnings.filterwarnings("ignore")

TRAINING_DATA_FOLDER = 'data/training/'
TRAINING_FOLDER = 'training/'

PREFIX = 'detoxify text: '

INPUT_MODEL = 't5-small'
OUTPUT_PATH = 'models/t5-small-detoxification'

TEMP_STAGE_LIMIT = 3

PRETRAIN_LR = 1e-2
TRAIN_LR = 2e-5

PRETRAIN_EPOCH = 4
TRAIN_EPOCH = 3

BATCH_SIZE = 64
FP16 = True

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python .\src\models\\train_model.py',
                    description='runs the training of the detoxification model.',
                    epilog='https://github.com/Demid65/text-detoxification')
                    
parser.add_argument('--dataset_path', type=str, metavar='TRAINING_DATA_FOLDER', dest='TRAINING_DATA_FOLDER',
                    help=f'Folder where split dataset is stored. Defaults to {TRAINING_DATA_FOLDER}', default=TRAINING_DATA_FOLDER)

parser.add_argument('--training_folder', type=str, metavar='TRAINING_FOLDER', dest='TRAINING_FOLDER',
                    help=f'Folder where intermediate training data is stored. Defaults to {TRAINING_FOLDER}', default=TRAINING_FOLDER)

parser.add_argument('--prefix', type=str, metavar='PREFIX', dest='PREFIX',
                    help=f'Prefix that is added to every model input. Defaults to "{PREFIX}"', default=PREFIX)

parser.add_argument('--model', type=str, metavar='INPUT_MODEL', dest='INPUT_MODEL',
                    help=f'Path to the pretrained model, should be a T5 model. Could be a hugging face model identifier. Defaults to {INPUT_MODEL}', default=INPUT_MODEL) 

parser.add_argument('--save_to', type=str, metavar='OUTPUT_PATH', dest='OUTPUT_PATH',
                    help=f'Place to save the resulting model into. Defaults to {OUTPUT_PATH}', default=OUTPUT_PATH)                            

parser.add_argument('--pretrain_lr', type=float, metavar='PRETRAIN_LR', dest='PRETRAIN_LR',
                    help=f'Learning rate for the pretraining stage. Defaults to {PRETRAIN_LR}', default=PRETRAIN_LR)  

parser.add_argument('--train_lr', type=float, metavar='TRAIN_LR', dest='TRAIN_LR',
                    help=f'Learning rate for the main training stage. Defaults to {TRAIN_LR}', default=TRAIN_LR)
                
parser.add_argument('--pretrain_epoch', type=int, metavar='PRETRAIN_EPOCH', dest='PRETRAIN_EPOCH',
                    help=f'Number of epochs for the pretraining stage. Defaults to {PRETRAIN_EPOCH}', default=PRETRAIN_EPOCH)

parser.add_argument('--train_epoch', type=int, metavar='TRAIN_EPOCH', dest='TRAIN_EPOCH',
                    help=f'Number of epochs for the main training stage. Defaults to {TRAIN_EPOCH}', default=TRAIN_EPOCH)

parser.add_argument('--batch_size', type=int, metavar='BATCH_SIZE', dest='BATCH_SIZE',
                    help=f'Batch size for training process. Defaults to {BATCH_SIZE}', default=BATCH_SIZE)

parser.add_argument('--no_fp16', dest='FP16', action='store_const', const=False, default=True,
                    help=f'Set this flag to disable FP16 precision. Set if CUDA is not available')

args = parser.parse_args()

TRAINING_DATA_FOLDER = args.TRAINING_DATA_FOLDER
TRAINING_FOLDER = args.TRAINING_FOLDER
PREFIX = args.PREFIX
INPUT_MODEL = args.INPUT_MODEL
OUTPUT_PATH = args.OUTPUT_PATH

PRETRAIN_LR = args.PRETRAIN_LR 
TRAIN_LR = args.TRAIN_LR
PRETRAIN_EPOCH = args.PRETRAIN_EPOCH
TRAIN_EPOCH = args.TRAIN_EPOCH
BATCH_SIZE = args.BATCH_SIZE
FP16 = args.FP16

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
trainer.save_model(OUTPUT_PATH)

print('done')
