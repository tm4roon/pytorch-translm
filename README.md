# Monolingual Translation with Language Model

Transformer-based Language Modelを単言語内翻訳に適用。原言語側の文と目的言語側の文を区切りトークン<SEP>で結合し、入力する。入力単語に対して、次の単語を予測するように出力させる。

- train.py: Transformer-based language modelの学習を行うスクリプト
- generate.py: 学習済みモデルをロードして、入力データを翻訳するスクリプト

<br>



## 環境構築

```python3
git clone https://github.com/marucha80t/monolingual_translation_with_language_model.git
cd ./monolingual_translation_with_language_model
pip install -r requirements.txt
```

<br>



## 使用方法

言語モデルの学習。

```python3
python train.py --train ./data/sample_train.tsv
                --valid ./data/sample_valid.tsv
                --savedir ./checkpoints
                --gpu
```

学習済み言語モデルによる翻訳。

```python3
python generate.py --model ./checkpoints/checkpoint_best.pt
                   --input ./data/sample_test.txt
                   --gpu
```

<br>



## 参考

- [Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836)
- [Efficient Adaptation of Pretrained Transformers for Abstractive Summarization](https://arxiv.org/abs/1906.00138)