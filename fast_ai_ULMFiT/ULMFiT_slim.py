import pandas as pd
from fastai.text import *

path = r'E:\CAS\UCAS\classes\text data mining\aclImdb\\'


def main():
    # data processing
    # Language model data
    # data_lm = TextLMDataBunch.from_csv(path, 'train.csv')

    # Classifier model data
    # data_clas = TextClasDataBunch.from_csv(path, 'train.csv', vocab=data_lm.train_ds.vocab, bs=32)

    # csv2pkl
    # data_lm.save(path + 'data_lm_export.pkl')
    # data_clas.save(path + 'data_clas_export.pkl')

    # data read
    data_lm = load_data(path, 'data_lm_export.pkl', bs=2)
    data_clas = load_data(path, 'data_clas_export.pkl', bs=2)
    #
    # # Fine-tuning a language model
    # learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    # learn.fit_one_cycle(1, 1e-2).to('cuda:0')
    # learn.unfreeze()
    # learn.fit_one_cycle(1, 1e-3).to('cuda:0')
    # # save fine-tuning model
    # learn.save_encoder(path + 'ft_enc')

    # # training classifier
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    # learn.load_encoder(path + 'ft_enc')
    # data_clas.show_batch()
    # learn.fit_one_cycle(1, 1e-2)
    # learn.freeze_to(-2)
    # learn.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
    # learn.save_encoder(path + 'temp_enc')
    # learn.unfreeze()
    learn.load_encoder(path + 'temp_enc')
    data_clas.show_batch()
    learn.fit_one_cycle(10, slice(2e-3 / 100, 2e-3)).to("cuda")
    learn.save_encoder(path + 'final_ft_enc')


if __name__ == '__main__':
    main()
