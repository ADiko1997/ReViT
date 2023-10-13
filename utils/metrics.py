import torch

def topk_correct(preds:torch.tensor, labels:torch.tensor, ks:list):
    """
    Note: This algorithm works only for classification problems whose labels are int in range [0 : (num_classes-1)]
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """

    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"

    #find top k predictions for each sample
    _top_k_vals, _top_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )

    _top_k_inds = _top_k_inds.t() #reshape to (top_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(_top_k_inds)

    #Indeces are equall to labels i.e. classes
    top_k_correct = _top_k_inds.eq(rep_max_k_labels)
    num_correct = [top_k_correct[:k, :].float().sum() for k in ks]

    return num_correct


def topk_accuracies(preds, labels, ks):
    """
    Computes top-k accuracies
    Note: This algorithm works only for classification problems whose labels are int in range [0 : (num_classes-1)]
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        Accuracies for each k
    """

    num_topk_correct = topk_correct(preds=preds, labels=labels, ks=ks)
    return [(x/preds.size(0)) * 100.0 for x in num_topk_correct]