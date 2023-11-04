# Stage 0: Dataset Exploration.

Before starting to build a solution, it is important to take a look at the problem we are trying to solve, more specifically at the dataset provided. In the [Notebook 0](../notebooks/0%20-%20Playing%20with%20the%20dataset.ipynb) I have explored the dataset, getting some interesting insight about it and the problem in general:

- For some cases, detotixification can be done only with a simple, context insensitive replacement or removal of swear word or toxic phases. But often times context heavily affects the meaning of the sentence. For extremely short or extremely toxic sentences, careful paraphrasing is required. 

- Strategies which are used to paraphrase the sentences in the dataset are inconsistent, leading to some sentences to become heavily butchered or completely meaningless. And while technically meaningless sentence can't really be considered toxic, we want to avoid such changes and preserve as much original meaning as possible. Especially that affects aforementioned short and very toxic sentences.

- Dataset has some metrics provided, like similarity and length difference. They however, seem to be completely useless in the context of the problem.

Here are some examples from the dataset to elaborate on these three points:

| toxic | detoxified | comment |
| ---- | ----- | ----- |
| I have orders to kill her. | I've got orders to put her down. | Replacing 'kill' with 'put down'. |
| Shit, this one I can't even pronounce.  | gosh, I can't even pronounce this. | Replacing 'shit' with 'gosh'. |
| Shut up, you two, 'said Granny.  | 'Be quiet, you two,' said Granny. | Replacing 'shut up' with 'be quiet'. |
| I like that shit.  | I love it. | Replacement of a commonly used phrase with a different one |
| Funny how Nazis are always the bad guys. | why are the Nazis always the bad guys? | Replacing the statement with a question, making the sentence more neutral. |
| I'll freeze him!  | Freezing him. | Replacing exclamation with a statement, making the sentence less emotional. |
| she was a killer.  | It was from the killer. | Completely losing the orignal meaning. |
| Real life starts the first time you fuck, kid.  | boy, real life starts up first. | Turning toxic in nature sentence into a meaningless one. |
| some killer! 	| The killer detail! | Detoxifying a short catch-prase, makes it loose all its meaning. |

The dataset provides a cosine similarity metric, which is not really useful, since for some really good examples ('I like that shit.' and 'I love it.') it is worse than for some bad examples ('The killer detail!' and 'some killer!'): 0.697344 versus 0.734141. Sometimes it is the other way around, making it pretty much random in relation to the task. Same goes for length difference, sentences like 'he's the tallest son of a bitch.' and 'So he's the Top dog.' are very different in length (0.363636) but have similar meaning and noticeable difference in toxicity.

# Hypothesis 1: RNNs

Since the problem requires some creative paraphrasing to preserve the meaning of original text while filtering as much toxicity as possible, simple algorithms like dictionary replacements and word filtering would have inconsistent performance, being effective only for small portion of the dataset. For the task of text detoxification a more complicated model should be used, and since provided texts can be represented as sequences of different lengths, an RNN would be a fitting model for that task.

# Hypothesis 2: Transformers

For Seq2Seq problems, a commonly used architecture of an RNN is a transformer. Those models are heavily used in text2text applications, so the fitting model for text detoxification should probably be a transformer.

# Hypothesis 3: Transfer Learning

Training of NLP transformer models usually takes a lot of time and effort, and usually consists of unsupervised and supervised stages. The dataset on the other hand is not of exceptionally high quality, containing many flaws that could harm the performance of the model. Both of those issues can be mitigated by using transfer learning. An already existing text2text model (which there are plenty of) could be finetuned on the dataset, reducing both the training and development time drastically. It is also likely that the model would retain much of it's prior knowledge about language, making it less vulnerable to dataset flaws.

This Hypothesis is explored in [Notebook 1](../notebooks/1%20-%20Exploration%20of%20existing%20solution.ipynb) with quite decent results. Just after one epoch of training, the model have learnt to replace swear words and toxic phrases, while still being hesitant to make drastic changes to the sentences, keeping meaning mostly intact. Some of the flaws present in the dataset however persisted, while some sentences were kept as is, so there are much room for the improvement.

I decided to use [T5-Small model](https://huggingface.co/t5-small) at this stage, since it one of the smallest pre-trained models for text2text applications, which has decent results at the translation tasks. The T5 family of models were developed by Colin Raffel *et al.* in 2020. The original paper can be found [there](https://jmlr.org/papers/volume21/20-074/20-074.pdf).

# Hypothesis 4: More finetuning

A single epoch of finetuning achieved certain results, which could likely be improved by running few extra epochs of training.