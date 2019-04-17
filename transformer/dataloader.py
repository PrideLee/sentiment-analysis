from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import string


def tokenize(input):
    """
        Naive tokenizer, that lower-cases the input
        and splits on punctuation and whitespace
    """
    input = input.lower()
    # string.punctuation 定义32位英文标准符号，将标点替换为" "
    for p in string.punctuation:
        input = input.replace(p, " ")
        # split()以空格切分字符串为列表
    return input.strip().split()


def num2words(vocab, vec):
    """
        Converts a vector of word indicies
        to a list of strings
    """
    return [vocab.itos[i] for i in vec]


def get_imdb(batch_size, max_length):
    # Defines a datatype together with instructions for converting to Tensor.
    # lower: Whether to lowercase the text in this field. Default: False.
    # include_lengths: Whether to return a tuple of a padded minibatch and a list containing the lengths of each
    # examples, or just a padded minibatch. Default: False.
    # batch_first: Whether to produce tensors with the batch dimension first. Default: False.
    # tokenize: The function used to tokenize strings using this field into sequential examples. If "spacy", the SpaCy
    # English tokenizer is used. Default: str.split.
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize=tokenize, fix_length=max_length)
    # sequential: Whether the datatype represents sequential data. If False, no tokenization is applied. Default: True.
    # unk_token: The string token used to represent OOV words. Default: "<unk>".
    # pad_token: The string token used as padding. Default: "<pad>".
    LABEL = data.Field(sequential=False, unk_token=None, pad_token=None)

    print("Loading data..\n")

    # make splits for data
    datasets
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('len(test)', len(test))
    print("")

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='42B', dim=300, max_vectors=500000))
    LABEL.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits based on the batch_size
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=-1)

    return train_iter, test_iter, TEXT.vocab.vectors, TEXT.vocab


if __name__ == "__main__":
    """
        If run seperately, does a simple sanity check,
        by printing different values,
        and counting labels
    """
    train, test, vectors, vocab = get_imdb(1, 50)
    from collections import Counter

    print(list(enumerate(vocab.itos[:100])))
    cnt = Counter()
    for i, b in enumerate(iter(train)):
        if i > 2:
            break
        print(i, num2words(vocab, b.text[0][0].numpy()))
        cnt[b.label[0].item()] += 1
    print(cnt)
