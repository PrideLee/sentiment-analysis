import fire
from fastai_old.fastai.text import *
from fastai_old.fastai.lm_rnn import *


class EarlyStopping(Callback):
    def __init__(self, learner, save_path, enc_path=None, patience=5):
        super().__init__()
        self.learner = learner
        self.save_path = save_path
        self.enc_path = enc_path
        self.patience = patience

    def on_train_begin(self):
        self.best_val_loss = 100
        self.num_epochs_no_improvement = 0

    def on_epoch_end(self, metrics):
        val_loss = metrics[0]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            self.learner.save(self.save_path)
            if self.enc_path is not None:
                self.learner.save_encoder(self.enc_path)
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f'Stopping - no improvement after {self.patience+1} epochs')
            return True

    def on_train_end(self):
        print(f'Loading best model from {self.save_path}')
        self.learner.load(self.save_path)


def train_lm(dir_path, pretrain_path, cuda_id=0, cl=3, pretrain_id='wt103', lm_id='', bs=32,
             dropmult=1.0, backwards=False, lr=4e-3, preload=True, bpe=False, startat=0,
             use_clr=True, use_regular_schedule=False, use_discriminative=True, notrain=False, joined=False,
             train_file_id='', early_stopping=True):
    """
    Fine-tuning a language model pretrained on WikiText-103 data on the target task data.
    :param dir_path: the directory that contains the Wikipedia files
    :param pretrain_path: the path where the pretrained model is saved; if using the downloaded model, this is wt103
    :param cuda_id: the id of the GPU used for training the model
    :param cl: number of epochs to train the model
    :param pretrain_id: the id of the pretrained model; set to wt103 per default
    :param lm_id: the id used for saving the fine-tuned language model
    :param bs: the batch size used for training the model
    :param dropmult: the factor used to multiply the dropout parameters
    :param backwards: whether a backwords LM is teained
    :param lr: the learing rate
    :param preload: whether we loadd a pretrainde LM(True by default)
    :param bpe: whether we use byte-pair encoding (BPE)
    :param startat: can be used to continue fine-tuning a model; if >0, loads an already fine-tuned LM; can also be used
    to indicate the layer at which to start the gradual unfreezing (1 is last hidden layer, etc.); in the final model,
    we only used this for training the classifier
    :param use_clr: whether to use slanted triangular learning rates (STLR) (True by default)
    :param use_regular_schedule: whether to use a regular schedule (instead of STLR)
    :param use_discriminative: whether to use discriminative fine-tuning (True by default)
    :param notrain: whether to skip fine-tuning
    :param joined: whether to fine-tune the LM on the concatenation of training and validation data
    :param train_file_id: can be used to indicate different training files (e.g. to test training sizes)
    :param early_stopping: whether to use early stopping
    :return:
    """
    print(f'dir_path {dir_path}; pretrain_path {pretrain_path}; cuda_id {cuda_id}; '
          f'pretrain_id {pretrain_id}; cl {cl}; bs {bs}; backwards {backwards} '
          f'dropmult {dropmult}; lr {lr}; preload {preload}; bpe {bpe};'
          f'startat {startat}; use_clr {use_clr}; notrain {notrain}; joined {joined} '
          f'early stopping {early_stopping}')

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    joined_id = 'lm_' if joined else ''
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    lm_path = f'{PRE}{lm_id}lm'  # fwd_pretrain_wt103_lm.h5
    enc_path = f'{PRE}{lm_id}lm_enc'  # fwd_pretrain_wt103_lm_enc.h5

    dir_path = Path(dir_path)
    pretrain_path = Path(pretrain_path)
    pre_lm_path = pretrain_path / 'models' / f'{PRE}{pretrain_id}.h5'
    for p in [dir_path, pretrain_path, pre_lm_path]:
        assert p.exists(), f'Error: {p} does not exist.'

    bptt = 70
    em_sz, nh, nl = 400, 1150, 3
    # 优化方式，Adam
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_lm_path = dir_path / 'tmp' / f'trn_{joined_id}{IDS}{train_file_id}_bwd.npy'
        val_lm_path = dir_path / 'tmp' / f'val_{joined_id}{IDS}_bwd.npy'
    else:
        # trn_lm_path = dir_path + 'tmp/trn_ids.npy' or 'tmp/val_ids.npy'
        trn_lm_path = dir_path / 'tmp' / f'trn_{joined_id}{IDS}{train_file_id}.npy'
        val_lm_path = dir_path / 'tmp' / f'val_{joined_id}{IDS}.npy'

    print(f'Loading {trn_lm_path} and {val_lm_path}')
    trn_lm = np.load(trn_lm_path)
    # 拼接np.concatenate()
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.load(val_lm_path)
    val_lm = np.concatenate(val_lm)

    if bpe:
        vs = 30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))  # the frequence words
        vs = len(itos)  # the words number
    # LanguageModelLoader(): Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs, bptt)
    md = LanguageModelData(dir_path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropmult  # dropmult=1.0
    # md.get_model(): Returns a SequentialRNN model
    # drpoouti: dropout to apply to the input layer,
    # dropouth: dropout to apply to the activations going from one LSTM layer to another
    # wdrop: dropout used for a LSTM's internal (or hidden) recurrent weights
    # dropoute: dropout to apply to the embedding layer
    learner = md.get_model(opt_fn, em_sz, nh, nl,
                           dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
    # paritial()调用seq2seq，固定函数参数alpha=2，beta=1
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip = 0.3
    learner.metrics = [accuracy]
    wd = 1e-7

    lrs = np.array([lr / 6, lr / 3, lr, lr / 2]) if use_discriminative else lr
    #
    if preload and startat == 0:  # preload=True
        # import pre-training model weight
        wgts = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)
        if bpe:
            learner.model.load_state_dict(wgts)
        else:
            print(f'Loading pretrained weights...')
            # convert wgts['0.encoder.weight'] to array
            ew = to_np(wgts['0.encoder.weight'])
            row_m = ew.mean(0)
            # itos2: the word-to-token mapping
            itos2 = pickle.load(open(pretrain_path / 'tmp' / f'itos_wt103.pkl', 'rb'))
            # collections.defaultdict会返回一个类似dictionary的对象，注意是类似的对象，不是完全一样的对象。
            # 这个defaultdict和dict类，几乎是一样的，除了它重载了一个方法和增加了一个可写的实例变量。
            stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
            nw = np.zeros((vs, em_sz), dtype=np.float32)  # em_sz: embedding_size, vs: words dictionary nubs
            nb = np.zeros((vs,), dtype=np.float32)
            # 根据目标文本语料在原始pre-training模型中选择确定 weight
            for i, w in enumerate(itos):
                r = stoi2[w]
                if r >= 0:
                    nw[i] = ew[r]
                else:
                    nw[i] = row_m
            # T(): Convert numpy array into a pytorch tensor.
            wgts['0.encoder.weight'] = T(nw)
            wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
            wgts['1.decoder.weight'] = T(np.copy(nw))
            # model.load_state_dict(): 加载模型
            learner.model.load_state_dict(wgts)
            ##  learner.freeze_to(-1)
            ##  learner.fit(lrs, 1, wds=wd, use_clr=(6,4), cycle_len=1)
    elif preload:
        print('Loading LM that was already fine-tuned on the target data...')
        learner.load(lm_path)

    if not notrain:  # unfreeze=False
        learner.unfreeze()
        if use_regular_schedule:  # use_regular_schedule=False
            print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
            use_clr = None
            n_cycles = cl  # c1 the number of epoch
            cl = None
        else:
            n_cycles = 1
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
            print('Using early stopping...')
        # use_clr=True,
        learner.fit(lrs, n_cycles, wds=wd, use_clr=(32, 10) if use_clr else None, cycle_len=cl,
                    callbacks=callbacks)
        learner.save(lm_path)
        learner.save_encoder(enc_path)
    else:
        print('No more fine-tuning used. Saving original LM...')
        learner.save(lm_path)
        learner.save_encoder(enc_path)


if __name__ == '__main__':
    # fire.Fire(train_lm)
    path_dir = r'E:\CAS\UCAS\classes\text data mining\aclImdb'
    model_dir = r'E:\CAS\UCAS\classes\text data mining\aclImdb\wt103'
    train_lm(dir_path=path_dir, pretrain_path=model_dir, lm_id='pretraitn_wt103')
