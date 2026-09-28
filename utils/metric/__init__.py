import torch


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum().clamp_min(1)
        acc = torch.diag(h) / h.sum(1).clamp_min(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)).clamp_min(1)
        return acc_global, acc, iu

    def format(self, classes):
        acc_global, acc, iu = self.compute()
        rows = ['class\tacc\tiou']
        for name, a, j in zip(classes, (acc * 100).tolist(), (iu * 100).tolist()):
            rows.append(f'{name}\t{a:.2f}\t{j:.2f}')
        return (
            f'global correct: {acc_global.item() * 100:.1f}\n'
            f'mean correct: {acc.mean().item() * 100:.1f}\n'
            f'mean IoU: {iu.mean().item() * 100:.1f}\n' + '\n'.join(rows)
        )
