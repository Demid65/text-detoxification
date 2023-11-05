from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

DATASET_LINK = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
EXTRACT_FOLDER = 'data/raw/'
DATASET_FILENAME = 'filtered.tsv'
INTERIM_PATH = 'data/interim/processed.csv'

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python .\src\data\download_dataset.py',
                    description='downloads the dataset for text detoxification, preprocesses it and saves into the intermediate folder.',
                    epilog='https://github.com/Demid65/text-detoxification')
                    
parser.add_argument('--link', type=str, metavar='DATASET_LINK', dest='DATASET_LINK',
                    help=f'Link to the dataset. Defaults to {DATASET_LINK}', default=DATASET_LINK)

parser.add_argument('--extract_to', type=str, metavar='EXTRACT_FOLDER', dest='EXTRACT_FOLDER',
                    help=f'Folder where dataset is extracted to. Defaults to {EXTRACT_FOLDER}', default=EXTRACT_FOLDER)
                            
parser.add_argument('--save_to', type=str, metavar='OUTPUT_FILE', dest='INTERIM_PATH',
                    help=f'Path where processed dataset is saved. Defaults to {INTERIM_PATH}', default=INTERIM_PATH) 

args = parser.parse_args()

DATASET_LINK = args.DATASET_LINK
EXTRACT_FOLDER = args.EXTRACT_FOLDER
INTERIM_PATH = args.INTERIM_PATH
DATASET_PATH = os.path.join(EXTRACT_FOLDER, DATASET_FILENAME)


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