import torch


def get_pos_onehot(length):
    # initial zero matrix [length, length]
    onehot = torch.zeros(length, length)
    # torch.arrange(length).long() 生成[0-length]的tensor，.view(-1, 1)转变为length行，1列
    idxs = torch.arange(length).long().view(-1, 1)
    # onehot.scatter_(1, idxs, 1)生成[idxs, idxs]的对角单位阵
    onehot.scatter_(1, idxs, 1)
    return onehot


if __name__ == "__main__":
    print(get_pos_onehot(3))
