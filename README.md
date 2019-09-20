# Monolingual Translation with Language Model
An implimentation of transformer-based language model for sentence rewriting tasks
such as summarization, text simplification, paraphrase generaation,
style transfer, and grammatical error correction.
The following figure shows the architecture overview. This model recives a input 
that joint original sentence and simplified sentence by special token \<SEP\>, 
which means the delimiter. Then, the model geneate target sentences. 
This architechture is very simple, but have shown great result in 
text summarization task and text simplification task.  
<br>

<img src="https://user-images.githubusercontent.com/53220859/65313114-ccf70f00-dbce-11e9-822c-338fac8520e7.png" width="500">
<br>


## Installation
This code are depend on the following.
- python==3.6.5
- pytorch==1.1.0
- torchtext==0.3.1

```sh
git clone https://github.com/maru0kun/pytorch-translm.git
cd ./pytorch-translm
pip install -r requirements.txt
```
<br>


## Usages
### Pre-training
The dataset for fine-tuning must be text file.
The input sentence must be segmented to words by whitespace.
If you want to use gpu, please set option `--gpu`.

```sh
python train.py pretrain \
    --train ./path/to/train.txt \
    --savedir ./checkpoints/pre-trained \
    --gpu
```


### Fine-tuning
The dataset for fine-tuning must be tsv format.
The source senteces and target sentences must be segmented to words by whitespace.
If you want to use gpu, please set option `--gpu`.

```sh
python train.py finetune \
    --model ./checkpoints/pre-trained/checkpoint_best.pt \
    --train ./path/to/train.tsv \
    --valid ./path/valid.tsv \
    --savedir ./checkpoints/fine-tuned \
    --gpu
```

### Translation
In translation step, you must set option `--model` and `--input`.
You can set sentence length of model's output using `--maxlen` option (default: 100 tokens).

```sh
python generate.py \
    --model ./checkpoints/fine-tuned/checkpoint_best.pt \
    --input ./data/sample_test.txt \
    --gpu
```

<br>


## References
- [Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836)
- [Efficient Adaptation of Pretrained Transformers for Abstractive Summarization](https://arxiv.org/abs/1906.00138)

