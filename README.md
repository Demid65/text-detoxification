# Text detoxification model for PMLDL course
 Model created by Demid Efremov as part of PMLDL course
 
 [d.efremov@innopolis.university](d.efremov@innopolis.university)

 The model aims to reudce the toxicity of texts by paraphrasing sentences using finetuned [t5-small](https://huggingface.co/t5-small) language model.

 ## Requirements
This projects uses HuggingFace transformer library as the method of running text2text models.

Requirements can be installed with:

```$ pip install -r requirements.txt```

 ## Usage
 1. Install the requirements
 2. Download the model [here](https://drive.google.com/file/d/1MHRR0vSojO89bHm0n_-fGCcwXxWKC1wn/view?usp=sharing), and unzip it into the models folder.
 3. Run the inference script with ```$ python ./src/models/predict_model.py```

 Command line parameters can be seen with
 
 ```$ python ./src/models/predict_model.py -h```

 ## Training
 1. Install the requirements
 2. Run the dataset downloader with ```$ python ./src/data/download_dataset.py```
 3. Run the dataset splitter with ```$ python ./src/data/split_dataset.py```
 4. Run the training with ```$ python ./src/models/train_model.py```

 Use -h flag with any script to see configurable parameters.
 Check out the notebooks for extra info.
 
 