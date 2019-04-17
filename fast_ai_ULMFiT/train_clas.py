# import fire
# from fastai.text import *
# from fastai_old.fastai.text import *
# from fastai_old.fastai.lm_rnn import *
#
#
# def freeze_all_but(learner, n):
#     c = learner.get_layer_groups()
#     for l in c:
#         set_trainable(l, False)
#     set_trainable(c[n], True)
#
#
# def train_clas(dir_path, cuda_id=0, lm_id='', clas_id=None, bs=224, cl=1, backwards=False, startat=0, unfreeze=True,
#                lr=0.01, dropmult=1.0, bpe=False, use_clr=True,
#                use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
#                from_scratch=False, train_file_id=''):
#     """
#      train the classifier on top of the fine-tuned language model with gradual unfreezing, discriminative fine-tuning,
#      and slanted triangular learning rates
#      DIR_PATH: the directory where the tmp and models folder are located
#      CUDA_ID: the id of the GPU used for training the model
#      LM_ID: the id of the fine-tuned language model that should be loaded
#      CLAS_ID: the id used for saving the classifier
#      BS: the batch size used for training the model
#      CL: the number of epochs to train the model with all layers unfrozen
#      BACKWARDS: whether a backwards LM is trained
#      STARTAT: whether to use gradual unfreezing (0) or load the pretrained model (1)
#      UNFREEZE: whether to unfreeze the whole network (after optional gradual unfreezing) or train only the final
#      classifier layer (default is True)
#      LR: the learning rate
#      DROPMULT: the factor used to multiply the dropout parameters
#      BPE: whether we use byte-pair encoding (BPE)
#      USE_CLR: whether to use slanted triangular learning rates (STLR) (True by default)
#      USE_REGULAR_SCHEDULE: whether to use a regular schedule (instead of STLR)
#      USE_DISCRIMINATIVE: whether to use discriminative fine-tuning (True by default)
#      LAST: whether to fine-tune only the last layer of the model
#      CHAIN_THAW: whether to use chain-thaw
#      FROM_SCRATCH: whether to train the model from scratch (without loading a pretrained model)
#      TRAIN_FILE_ID: can be used to indicate different training files (e.g. to test training sizes)`
#     """
#     print(
#         f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
#         f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; bpe {bpe}; use_clr {use_clr};'
#         f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
#         f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id}')
#     if not hasattr(torch._C, '_cuda_setDevice'):
#         print('CUDA not available. Setting device=-1.')
#         cuda_id = -1
#     torch.cuda.set_device(cuda_id)
#
#     PRE = 'bwd_' if backwards else 'fwd_'  # PRE='fwd_'
#     PRE = 'bpe_' + PRE if bpe else PRE
#     IDS = 'bpe' if bpe else 'ids'
#     train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'  # train_file_id=''
#     dir_path = Path(dir_path)
#     lm_id = lm_id if lm_id == '' else f'{lm_id}_'  # lm_id='pretrain_wt103'
#     clas_id = lm_id if clas_id is None else clas_id
#     clas_id = clas_id if clas_id == '' else f'{clas_id}_'
#     intermediate_clas_file = f'{PRE}{clas_id}clas_0'  # save path
#     final_clas_file = f'{PRE}{clas_id}clas_1'  # save path
#     lm_file = f'{PRE}{lm_id}lm_enc'  # lm_file='fwd_pretrain_wt103_lm_enc'
#     lm_path = dir_path / 'models' / f'{lm_file}.h5'
#
#     assert lm_path.exists(), f'Error: {lm_path} does not exist.'
#     # em_sz: embedding size, nh: number of hidden activation per LSTM layer, nl: n_layers
#     bptt, em_sz, nh, nl = 70, 400, 1150, 3
#     opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
#
#     if backwards:
#         trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}_bwd.npy')
#         val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
#     else:
#         # text data
#         trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}.npy')
#         val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
#
#     # text labels
#     trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn{train_file_id}.npy')
#     val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
#
#     assert trn_lbls.shape[1] == 1 and val_lbls.shape[
#         1] == 1, 'This classifier uses cross entropy loss and only support single label samples'
#     trn_lbls = trn_lbls.flatten()
#     val_lbls = val_lbls.flatten()
#     print('Trn lbls shape:', trn_lbls.shape)
#     # 让最小的label为0
#     trn_lbls -= trn_lbls.min()
#     val_lbls -= val_lbls.min()
#     c = int(trn_lbls.max()) + 1
#     print('Number of labels:', c)
#
#     if bpe:
#         vs = 30002
#     else:
#         itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
#         vs = len(itos)
#     # TextDataset class, which is the main dataset you should use for your NLP tasks,
#     trn_ds = TextDataset(trn_sent, trn_lbls)
#     val_ds = TextDataset(val_sent, val_lbls)
#     # Go through the text data by order of length with a bit of randomness.
#     trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs)
#     val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
#     # Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
#     # Batch_size = bs//2
#     # mum_workers:num_workers (int, optional): how many subprocesses to use for data loading.
#     # sampler (Sampler, optional): defines the strategy to draw samples from the dataset.
#     trn_dl = DataLoader(trn_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
#     val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
#     # Encapsulates DataLoaders and Datasets for training, validation, test. Base class for fastai *Data classes.
#     md = ModelData(dir_path, trn_dl, val_dl)
#
#     dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropmult
#     ## dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
#     ## dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
#     ## dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult
#
#     # pad_token (int): the int value used for padding text.
#     m = get_rnn_classifier(bptt, 20 * bptt, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
#                            layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
#                            dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
#     # Combines a ModelData object with a nn.Module object, such that you can train that module. md: model data,
#     # TextModel:Model, to_gpu: puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true.
#     learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
#     learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
#     # clip: gradient clip chosen to limit the change in the gradient to prevent exploding gradients Eg. .3
#     learn.clip = 25.
#     # metrics: metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
#     learn.metrics = [accuracy]
#
#     lrm = 2.6
#     # use discriminative fine-tuning (True by default)
#     if use_discriminative:
#         lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])
#     else:
#         lrs = lr
#     wd = 1e-6
#     # from_scratch=False, loading a pretrained model
#     if not from_scratch:
#         # load pre-training model
#         learn.load_encoder(lm_file)
#     else:
#         print('Training classifier from scratch. LM encoder is not loaded.')
#         use_regular_schedule = True
#     # startat=0,use gradual unfreezing(0).last=False, chain_thaw=False, from_sratch=False
#     if (startat < 1) and not last and not chain_thaw and not from_scratch:
#         learn.freeze_to(-1)
#         # Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)
#         # wds (float or list(float)): weight decay parameter(s).
#         # n_cycle (int)=1: number of cycles (or iterations) to fit the model for
#         # use_crl=True
#         learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
#                   use_clr=None if use_regular_schedule or not use_clr else (8, 3))
#         learn.freeze_to(-2)
#         learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 7,
#                   use_clr=None if use_regular_schedule or not use_clr else (8, 3))
#         learn.save(intermediate_clas_file)
#     elif startat == 1:
#         learn.load(intermediate_clas_file)
#
#     # chain_thaw=False
#     if chain_thaw:
#         lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
#         print('Using chain-thaw. Unfreezing all layers one at a time...')
#         n_layers = len(learn.get_layer_groups())
#         print('# of layers:', n_layers)
#         # fine-tune last layer
#         learn.freeze_to(-1)
#         print('Fine-tuning last layer...')
#         learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
#                   use_clr=None if use_regular_schedule or not use_clr else (8, 3))
#         n = 0
#         # fine-tune all layers up to the second-last one
#         while n < n_layers - 1:
#             print('Fine-tuning layer #%d.' % n)
#             freeze_all_but(learn, n)
#             learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
#                       use_clr=None if use_regular_schedule or not use_clr else (8, 3))
#             n += 1
#     # unfreeze = True, Sets every layer group to trainable
#     if unfreeze:
#         learn.unfreeze()
#     else:
#         # freeze_to(1) is to freeze layer group 0 and unfreeze the rest layer groups, and freeze_to(3) is to freeze
#         # layer groups 0, 1, and 2 but unfreeze the rest layer groups
#         learn.freeze_to(-3)
#     # last=False
#     if last:
#         print('Fine-tuning only the last layer...')
#         learn.freeze_to(-1)
#
#     if use_regular_schedule:
#         print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
#         use_clr = None
#         n_cycles = cl
#         cl = None
#     else:
#         n_cycles = 1
#     learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8, 8) if use_clr else None)
#     print('Plotting lrs...')
#     learn.sched.plot_lr()
#     learn.save(final_clas_file)
#
#
# if __name__ == '__main__':
#     path_dir = r'E:\CAS\UCAS\classes\text data mining\aclImdb'
#     train_clas(path_dir, cuda_id=0, lm_id='pretrain_wt103', clas_id='pretrain_wt103', cl=4)
#     # fire.Fire(train_clas)


import fire
from fastai_old.fastai.text import *
from fastai_old.fastai.lm_rnn import *


def freeze_all_but(learner, n):
    c = learner.get_layer_groups()
    for l in c: set_trainable(l, False)
    set_trainable(c[n], True)


def train_clas(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
               lr=0.01, dropmult=1.0, bpe=False, use_clr=True,
               use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
               from_scratch=False, train_file_id=''):
    print(
        f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
        f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; bpe {bpe}; use_clr {use_clr};'
        f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
        f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    intermediate_clas_file = f'{PRE}{clas_id}clas_0'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt, em_sz, nh, nl = 70, 400, 1150, 3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}_bwd.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')

    trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn{train_file_id}.npy')
    val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
    assert trn_lbls.shape[1] == 1 and val_lbls.shape[
        1] == 1, 'This classifier uses cross entropy loss and only support single label samples'
    trn_lbls = trn_lbls.flatten()
    val_lbls = val_lbls.flatten()
    print('Trn lbls shape:', trn_lbls.shape)
    trn_lbls -= trn_lbls.min()
    val_lbls -= val_lbls.min()
    c = int(trn_lbls.max()) + 1
    print('Number of labels:', c)

    if bpe:
        vs = 30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(itos)

    trn_ds = TextDataset(trn_sent, trn_lbls)
    val_ds = TextDataset(val_sent, val_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs // 2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    trn_dl = DataLoader(trn_ds, bs // 2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, trn_dl, val_dl)

    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropmult
    # dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
    # dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
    # dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

    m = get_rnn_classifier(bptt, 20 * bptt, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                           layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
                           dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip = 25.
    learn.metrics = [accuracy]

    lrm = 2.6
    if use_discriminative:
        lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])
    else:
        lrs = lr
    wd = 1e-6
    if not from_scratch:
        learn.load_encoder(lm_file)
    else:
        print('Training classifier from scratch. LM encoder is not loaded.')
        use_regular_schedule = True

    if (startat < 1) and not last and not chain_thaw and not from_scratch:
        learn.freeze_to(-1)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.freeze_to(-2)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.save(intermediate_clas_file)
    elif startat == 1:
        learn.load(intermediate_clas_file)

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
        print('Using chain-thaw. Unfreezing all layers one at a time...')
        n_layers = len(learn.get_layer_groups())
        print('# of layers:', n_layers)
        # fine-tune last layer
        learn.freeze_to(-1)
        print('Fine-tuning last layer...')
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        n = 0
        # fine-tune all layers up to the second-last one
        while n < n_layers - 1:
            print('Fine-tuning layer #%d.' % n)
            freeze_all_but(learn, n)
            learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 10,
                      use_clr=None if use_regular_schedule or not use_clr else (8, 3))
            n += 1

    if unfreeze:
        learn.unfreeze()
    else:
        learn.freeze_to(-3)

    if last:
        print('Fine-tuning only the last layer...')
        learn.freeze_to(-1)

    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles = cl
        cl = None
    else:
        n_cycles = 1
    learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8, 8) if use_clr else None)
    print('Plotting lrs...')
    learn.sched.plot_lr()
    learn.save(final_clas_file)


if __name__ == '__main__':
    # fire.Fire(train_clas)
    path_dir = r'E:\CAS\UCAS\classes\text data mining\aclImdb'
    train_clas(path_dir, cuda_id=0, lm_id='pretrain_wt103', clas_id='pretrain_wt103', cl=4)
