from fastai.text import *
import html
import fire

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace('#36;', '$') \
        .replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n')\
        .replace(' @.@ ', '.').replace(' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls, lang='en'):
    # n_lbs= 1
    # raw_data has no label
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        # get all the label
        labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls + 1, len(df.columns)):
            # n_lbs+1=len(df.columns), will not come in this circular
            texts += f' {FLD} {i-n_lbls+1} ' + df[i].astype(str)

    texts = list(texts.apply(fixup).values)
    # tokenizer: 用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示，即对每个文本进行分词
    # partition_by_cores: "Split data in `a` equally among `n_cpus` cores"
    # process_all: process a list of text
    tok = Tokenizer(lang=lang).process_all(partition_by_cores(texts, n_cpus=4))
    return tok, list(labels)


def get_all(df, n_lbls, lang='en'):
    tok, labels = [], []

    # i为df的索引，r为Dataframe
    tok, labels = get_texts(df, n_lbls, lang=lang)
    # for i, r in enumerate(df):
    #     tok_, labels_ = get_texts(r, n_lbls, lang=lang)
    #     tok += tok_
    #     labels += labels_
    return tok, labels


# def create_toks(dir_path, chunksize=2400, n_lbls=1, lang='en'):
def create_toks(dir_path, n_lbls=1, lang='en'):
    # print(f'dir_path {dir_path} chunksize {chunksize} n_lbls {n_lbls} lang {lang}')
    # try:
    #     spacy.load(lang)
    # except OSError:
    #     # TODO handle tokenization of Chinese, Japanese, Korean
    #     print(f'spacy tokenization model is not installed for {lang}.')
    #     lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
    #     print(f'Command: python -m spacy download {lang}')
    #     sys.exit(1)
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    # pd.read_csv()chunksize 分块读取csv，chunksize为每次读取的行数
    # df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
    # df_val = pd.read_csv(dir_path / 'test.csv', header=None, chunksize=chunksize)
    df_trn = pd.read_csv(dir_path / 'train.csv', header=None)
    df_val = pd.read_csv(dir_path / 'test.csv', header=None)
    # create folder to saving the '.npy' file
    tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)
    # tok: 文本词典
    tok_trn, trn_labels = get_all(df_trn, n_lbls, lang=lang)
    tok_val, val_labels = get_all(df_val, n_lbls, lang=lang)


    np.save(tmp_path / 'tok_trn.npy', tok_trn)
    np.save(tmp_path / 'tok_val.npy', tok_val)
    np.save(tmp_path / 'lbl_trn.npy', trn_labels)
    np.save(tmp_path / 'lbl_val.npy', val_labels)

    trn_joined = [' '.join(o) for o in tok_trn]
    # 将词典写入txt
    open(tmp_path / 'joined.txt', 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__':
    # fire.Fire(create_toks)
    path = r'E:\CAS\UCAS\classes\text data mining\aclImdb'
    create_toks(path)
