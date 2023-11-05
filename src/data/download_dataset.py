from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

DATASET_LINK = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
EXTRACT_FOLDER = 'data/raw/'
DATASET_FILENAME = 'filtered.tsv'
DATASET_PATH = os.path.join(EXTRACT_FOLDER, DATASET_FILENAME)
INTERIM_FOLDER = 'data/interim/'
INTERIM_FILENAME = 'processed.csv'
INTERIM_PATH = os.path.join(INTERIM_FOLDER, INTERIM_FILENAME)

# Download and unzip the dataset
def download_and_unzip(url, extract_to='.'): # https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


print(f'downloading from {DATASET_LINK}')
download_and_unzip(DATASET_LINK, extract_to=EXTRACT_FOLDER)

print(f'reading from {DATASET_PATH}')
df = pd.read_csv(DATASET_PATH, sep='\t')

d = {'id': [], 'toxic' : [], 'detoxified': [], 'tox_score': [], 'detox_score': [], 'similarity': [], 'length_diff': []}

print('processing')
# process the dataset
for index, row in tqdm(df.iterrows()): # takes some time
    d['id'].append(row[0])
    d['similarity'].append(row['similarity'])
    d['length_diff'].append(row['lenght_diff']) # fix the typo
    
    ref = row['reference']
    trn = row['translation']
    
    # toxic - is the toxic version of the text
    # detoxified - is less toxic version of the text
    if row['ref_tox'] > row['trn_tox']:
        d['toxic'].append(ref)
        d['detoxified'].append(trn)
        d['tox_score'].append(row['ref_tox'])
        d['detox_score'].append(row['trn_tox'])
    else:
        d['toxic'].append(trn)
        d['detoxified'].append(ref)
        d['tox_score'].append(row['trn_tox'])
        d['detox_score'].append(row['ref_tox'])
        
df = pd.DataFrame(d)

# save the processed dataset to a file
print(f'saving to {INTERIM_PATH}')
df.to_csv(INTERIM_PATH, index=False)

print('done')