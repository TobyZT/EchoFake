import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len == max_len:
        return x

    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def preprocess(batch):
    wav = batch["path"]["array"]  # np.ndarray [T]
    wav = pad_random(wav, max_len=64000).astype(np.float32)
    return {"input_values": wav, "labels": batch["label"]}


def preprocess_for_test(batch):
    wav = batch["path"]["array"]  # np.ndarray [T]
    wav = pad(wav, max_len=64000).astype(np.float32)
    return {"input_values": wav, "labels": batch["label"]}


def data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = [f["label"] for f in features]
    return {
        "input_values": torch.tensor(input_values),
        "labels": torch.tensor(labels),
    }


def compute_f1(labels, preds):
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    return precision, recall, f1


def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro"),
    }


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )  # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )  # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(pred, label):
    """Returns equal error rate (EER) and the corresponding threshold."""
    if isinstance(pred, torch.Tensor):
        pred, label = pred.cpu().numpy(), label.cpu().numpy()

    pred, label = np.array(pred).flatten(), np.array(label).flatten()
    target_scores = pred[label == 1.0]
    nontarget_scores = pred[label == 0.0]
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer  # , thresholds[min_index]


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
