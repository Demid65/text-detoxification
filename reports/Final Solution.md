# Introduction
This project aims to create a text detoxification model that would be able to reduce toxicity of given texts, using text2text models and transfer learning. 

# Data analysis
The [filtered ParaNMT-detox corpus (500K sentence pairs)](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip) was used as the primary dataset for this project, provided by the course team. Analysis of this dataset had shown that the dataset is of questionable quality, but has examples of different detoxification techniques, like swear word filtering, toxic phrase replacement and full paraphrasing of sentences. That insight influenced the whole model building process.
The dataset is also quite big in size, and only a part of it was used in the final training, because of limited processing power, but more of the dataset can be used in the future.

# Model Specification
The model used in this project is a transformer based on T5 architecture, which was developed in 2020 by Colin Raffel *et al.* The original paper can be found [there](https://jmlr.org/papers/volume21/20-074/20-074.pdf). This model was developed as a general framework for text2text applications and shows formidable results in translation tasks. The smallest model in the lineup ([t5-small](https://huggingface.co/t5-small)) with its 60M parameters was used in this project.

The model is small enough to be easily trainable on the available hardware (including google colab), while inference could be run with relative ease and speed on any hardware. 

# Training Process
The transfer learning technique was used in the training process: the original t5-small model was finetuned on the dataset. Training was split into two sections:

- Pre-training. The model is first trained on a small portion of dataset with high learning rate for a number of epochs. This allows the original model to be quickly finetuned to the task, and creates a somewhat accurate text detoxification model. The final model was pre-trained for 4 epochs on 20k samples from the dataset, until the point where it started overfitting.

- Training. After pre-training, the model is then trained on a more substantial portion of the dataset with much smaller learning rate, to improve its accuracy. Since the model is already capable at the task, this approach should yield well-performing model in shorter computational time. The final model was trained for 7 epochs on 150k samples from the dataset.

# Evaluation
The NLP tasks are usually very hard to evaluate because of unpredictable nature of human language, so it is hard to find the perfect metric for evaluation. In this project two metrics were used to judge the performance of the model: model loss and [BLEU metric](https://en.wikipedia.org/wiki/BLEU). The resulting model achieved BLEU score of 24% on the 10k evaluation dataset, and evaluation loss of 1.75 on the same data.

# Results
The result of this project is a finetunned t5-small model which is capable of reducing toxicity of given texts, which has decent performance, as well as a set of tools to process the original dataset and a traning script to create models for similar text2text applications. The model is lightweight and can easily be run on a GPU-less machine with little delays.
