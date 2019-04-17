import torch
from torch import optim
from torch import nn
import time
from dataloader import get_imdb
from model import Net

try:
    # try to import tqdm for progress updates
    from tqdm import tqdm
except ImportError:
    # on failure, make tqdm a noop
    def tqdm(x):
        return x

try:
    # try to import visdom for visualisation of attention weights
    import visdom
    from helpers import plot_weights

    vis = visdom.Visdom()
except ImportError:
    vis = None
    pass


def val(model, test, vocab, device, epoch_num, path_saving):
    """
        Evaluates model on the test set
    """
    # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will
    # work in eval model instead of training mode.

    model.eval()
    print("\nValidating..")
    if not vis is None:
        visdom_windows = None
    # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you
    # won’t be able to backprop (which you don’t want in an eval script).
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i, b in enumerate(tqdm(test)):
            if not vis is None and i == 0:
                visdom_windows = plot_weights(model, visdom_windows, b, vocab, vis)
            model_out = model(b.text[0].to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        with open(path_saving + '_val_results', 'a', encoding='utf-8') as file:
            temp = "epoach:{}, correct:{}%, correct samples/total samples{}/{}".format(epoch_num, correct / total,
                                                                                       correct, total)
            file.write(temp + '\n')
        print(temp)
    return correct / total


def train(max_length, model_size, epochs, learning_rate, device, num_heads, num_blocks, dropout, train_word_embeddings,
          batch_size, save_path):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    # train: train iterator
    # test: test iterator
    # vectors: train data word vector
    # vocab: train data vocab
    train, test, vectors, vocab = get_imdb(batch_size, max_length=max_length)
    # creat the transformer net
    model = Net(model_size=model_size, embeddings=vectors, max_length=max_length, num_heads=num_heads,
                num_blocks=num_blocks, dropout=dropout, train_word_embeddings=train_word_embeddings).to(device)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_correct = 0
    with open(save_path + '_train_results', 'a', encoding='utf-8') as file_re:
        for i in range(0, epochs + 1):
            loss_sum = 0.0
            model.train()
            # train data has been spited many batch, tadm: print progress bar
            for j, b in enumerate(iter(tqdm(train))):
                optimizer.zero_grad()
                model_out = model(b.text[0].to(device))
                # calculate loss
                loss = criterion(model_out, b.label.to(device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print('\n **********************************************')
            loss_temp = "Epoch: {}, Loss mean: {}\n".format(i, loss_sum / j)
            file_re.write(loss_temp + '\n')
            print(loss_temp)
            # Validate on test-set every epoch
            if i % 5 == 0:
                val_correct = val(model, test, vocab, device, i, save_path)
            if val_correct > best_correct:
                best_correct = val_correct
                best_model = model
    torch.save(best_model, save_path + '_model.pkl')


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")
    ap.add_argument("--max_length", default=500, type=int, help="Maximum sequence length, sequences longer than this \
                                                                are truncated")
    ap.add_argument("--model_size", default=128, type=int, help="Hidden size for all hidden layers of the model")
    ap.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    ap.add_argument("--learning_rate", default=0.0001, type=float, dest="learning_rate",
                    help="Learning rate for optimizer")
    ap.add_argument("--device", default="cuda:0", dest="device", help="Device to use for training and evaluation \
                                                                      e.g. (cpu, cuda:0)")
    ap.add_argument("--num_heads", default=4, type=int, dest="num_heads", help="Number of attention heads in the \
                                                                               Transformer network")
    ap.add_argument("--num_blocks", default=1, type=int, dest="num_blocks",
                    help="Number of blocks in the Transformer network")
    ap.add_argument("--dropout", default=0.5, type=float, dest="dropout", help="Dropout (not keep_prob, but probability \
                                                            of ZEROING during training, i.e. keep_prob = 1 - dropout)")
    ap.add_argument("--train_word_embeddings", type=bool, default=True, dest="train_word_embeddings",
                    help="Train GloVE word embeddings")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--save_path", default=r'E:\CAS\UCAS\classes\text data mining\transformer\results\\' +
                                           time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    dest="save_path",
                    help="The path to save the results")
    args = vars(ap.parse_args())
    train(**args)
