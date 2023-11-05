from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import sys
import warnings

MODEL_PATH = 'models/t5-small-detoxification'

PREFIX = 'detoxify text: '
MAX_NEW_TOKENS = 128

INPUT = 'stdin'
OUTPUT = 'stdout'

# load model
print(f'loading model from {MODEL_PATH}')
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

if INPUT == 'stdin':
    input_file = sys.stdin
else: 
    input_file = open(INPUT)

if OUTPUT == 'stdout':
    out_file = sys.stdout
else: 
    out_file = open(OUTPUT, 'w')

if INPUT == 'stdin':
    print('ready')

while True:
    line = input_file.readline()
    if not line:
        break
    input_ids = tokenizer(PREFIX + line, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens = MAX_NEW_TOKENS)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    out_file.write(result + os.linesep)
 
input_file.close()
out_file.close()
print('done')

