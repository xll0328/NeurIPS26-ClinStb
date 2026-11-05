from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.clinstab_objective import stability_objective
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
import random
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from layers.Autoformer_EncDec import series_decomp

warnings.filterwarnings("ignore")


def _safe_auc(labels, probabilities, num_classes):
    one_hot = torch.nn.functional.one_hot(
        torch.as_tensor(labels, dtype=torch.long), num_classes=num_classes
    ).numpy()
    try:
        auroc = roc_auc_score(one_hot, probabilities, multi_class="ovr")
        auprc = average_precision_score(one_hot, probabilities, average="macro")
    except ValueError:
        auroc = float("nan")
        auprc = float("nan")
    return auroc, auprc

class AdaptivePGD(nn.Module):
    def __init__(self, model, eps=0.3, alpha=None, steps=100, rho=0.75, max_restarts=2, random_start=True, device=None):
        super().__init__()
        self.model = model
        self.eps = float(eps)
        self.alpha = float(alpha if alpha is not None else eps * 0.1)
        self.steps = int(steps)
        self.rho = float(rho)
        self.max_restarts = int(max_restarts)
        self.random_start = bool(random_start)
        self.device = device or next(model.parameters()).device
        if self.eps < 0 or self.alpha <= 0 or self.steps < 1:
            raise ValueError("eps must be non-negative; alpha and steps must be positive")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")

    def forward(self, inputs, labels, padding_mask, stddev, model_orig_outputs=None, custom_loss_fn=None):
        inputs = inputs.detach().to(self.device)
        labels = labels.detach().to(self.device)
        padding_mask = padding_mask.detach().to(self.device)
        if model_orig_outputs is not None:
            model_orig_outputs = model_orig_outputs.detach().to(self.device)
        outputs = []
        for start in range(0, inputs.size(0), 1024):
            stop = min(start + 1024, inputs.size(0))
            original = inputs[start:stop]
            target = labels[start:stop]
            mask = padding_mask[start:stop]
            reference = None if model_orig_outputs is None else model_orig_outputs[start:stop]
            adversarial = original.clone()
            if self.random_start:
                adversarial = adversarial + torch.randn_like(adversarial) * stddev
                adversarial = torch.maximum(torch.minimum(adversarial, original + self.eps), original - self.eps)
            best = adversarial.detach().clone()
            best_loss = float('-inf')
            step_size = self.alpha
            history = []
            restarts = 0
            was_training = self.model.training
            self.model.eval()
            try:
                for _ in range(self.steps):
                    adversarial.requires_grad_(True)
                    logits = self.model(adversarial, mask, None, None)
                    if custom_loss_fn is not None and reference is not None:
                        loss = custom_loss_fn(logits, reference, target)
                    elif reference is not None:
                        loss = nn.functional.mse_loss(logits, reference)
                    else:
                        loss = nn.functional.cross_entropy(logits, target.long())
                    value = float(loss.detach())
                    history.append(value)
                    if value > best_loss:
                        best_loss = value
                        best = adversarial.detach().clone()
                    if len(history) >= 10 and max(history[-10:]) <= history[-10] + 1e-6 and restarts < self.max_restarts:
                        restarts += 1
                        adversarial = best + torch.randn_like(best) * stddev * 0.1
                        adversarial = torch.maximum(torch.minimum(adversarial, original + self.eps), original - self.eps).detach()
                        step_size = self.alpha
                        history.clear()
                        continue
                    gradient = torch.autograd.grad(loss, adversarial)[0]
                    if len(history) >= 2:
                        step_size *= 1.0 / self.rho if value > history[-2] else self.rho
                        step_size = min(max(step_size, self.eps * 0.01), self.eps)
                    adversarial = adversarial.detach() + step_size * gradient.sign()
                    adversarial = torch.maximum(torch.minimum(adversarial, original + self.eps), original - self.eps).detach()
            finally:
                self.model.train(was_training)
            outputs.append(best)
        return torch.cat(outputs, dim=0)

class Exp_Classification_Rob(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):

        train_data, _ = self._get_data(flag="TRAIN")
        self.args.seq_len = train_data.max_seq_len
        self.args.pred_len = 0
        self.args.enc_in = train_data.X.shape[2]
        self.args.num_class = len(np.unique(train_data.y))

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 1e-5),
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(-1).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )

        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        auroc, auprc = _safe_auc(trues, probs, self.args.num_class)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro", zero_division=0),
            "Recall": recall_score(trues, predictions, average="macro", zero_division=0),
            "F1": f1_score(trues, predictions, average="macro", zero_division=0),
            "AUROC": auroc,
            "AUPRC": auprc,
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()

        return total_loss, metrics_dict

    def test_robustness(self, test_data, test_loader, criterion, stddev=2.0):
        total_loss = []
        preds = []
        trues = []

        from layers.Autoformer_EncDec import series_decomp

        perturbation_avg = series_decomp(kernel_size=13)

        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                residual, trend = perturbation_avg(batch_x)

                noisy_batch_x = batch_x + residual * stddev

                if self.swa:
                    outputs = self.swa_model(noisy_batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(noisy_batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(-1).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )

        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        auroc, auprc = _safe_auc(trues, probs, self.args.num_class)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro", zero_division=0),
            "Recall": recall_score(trues, predictions, average="macro", zero_division=0),
            "F1": f1_score(trues, predictions, average="macro", zero_division=0),
            "AUROC": auroc,
            "AUPRC": auprc,
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()

        return total_loss, metrics_dict

    def _load_state_dict(self, model, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=True)

    def _build_teacher(self, checkpoint_path):
        teacher_args = copy.copy(self.args)
        teacher_args.model = getattr(self.args, "teacher_model", "Medformer")
        teacher = self.model_dict[teacher_args.model].Model(teacher_args).float().to(self.device)
        self._load_state_dict(teacher, checkpoint_path)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def train(self, setting, pretrained_model_path=None, teacher_checkpoint=None):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        path = os.path.join("./checkpoints", setting)
        os.makedirs(path, exist_ok=True)
        student_checkpoint = pretrained_model_path or os.path.join(path, "checkpoint.pth")
        if not os.path.exists(student_checkpoint):
            raise FileNotFoundError(student_checkpoint)
        if not teacher_checkpoint or not os.path.exists(teacher_checkpoint):
            raise FileNotFoundError(teacher_checkpoint or "teacher checkpoint")

        self._load_state_dict(self.model, student_checkpoint)
        teacher_model = self._build_teacher(teacher_checkpoint)
        lambda_rob = float(getattr(self.args, "lambda_rob", 1.0))
        lambda_con = float(getattr(self.args, "lambda_con", 0.5))
        lambda_know = float(getattr(self.args, "lambda_know", 0.5))
        temperature = float(getattr(self.args, "temperature", 2.0))
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        def attack_loss(outputs_adv, outputs_orig, labels):
            robust_ce = nn.CrossEntropyLoss()(outputs_adv, labels.long())
            consistency = nn.MSELoss()(
                nn.Softmax(dim=1)(outputs_adv),
                nn.Softmax(dim=1)(outputs_orig),
            )
            return robust_ce + lambda_con * consistency

        attack = AdaptivePGD(
            self.model,
            eps=self.args.eps,
            alpha=self.args.alpha,
            steps=self.args.steps,
            rho=self.args.rho,
            max_restarts=self.args.max_restarts,
            random_start=True,
            device=self.device,
        )
        optimizer = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.args.train_epochs))
        )
        criterion = self._select_criterion()
        best_vali_f1 = float("-inf")

        for epoch in range(max(1, int(self.args.train_epochs))):
            self.model.train()
            all_perturbed_data = []
            all_original_data = []
            all_labels = []
            all_padding_masks = []

            for batch_x, label, padding_mask in train_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                with torch.no_grad():
                    outputs_orig = self.model(batch_x, padding_mask, None, None)
                perturbed_batch_x = attack(
                    batch_x,
                    label,
                    padding_mask,
                    self.args.stddev,
                    outputs_orig,
                    attack_loss,
                )
                all_perturbed_data.append(perturbed_batch_x.detach())
                all_original_data.append(batch_x.detach())
                all_labels.append(label.detach())
                all_padding_masks.append(padding_mask.detach())

            for _ in range(max(1, int(getattr(self.args, "iters_per_epoch", 1)))):
                for batch_x, perturbed_batch_x, label, padding_mask in zip(
                    all_original_data,
                    all_perturbed_data,
                    all_labels,
                    all_padding_masks,
                ):
                    optimizer.zero_grad()
                    clean_logits = self.model(batch_x, padding_mask, None, None)
                    robust_logits = self.model(perturbed_batch_x, padding_mask, None, None)
                    with torch.no_grad():
                        teacher_logits = teacher_model(batch_x, padding_mask, None, None)
                    loss, _ = stability_objective(
                        clean_logits,
                        robust_logits,
                        label,
                        teacher_logits=teacher_logits,
                        lambda_rob=lambda_rob,
                        lambda_con=lambda_con,
                        lambda_know=lambda_know,
                        temperature=temperature,
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    optimizer.step()

            scheduler.step()
            if self.swa:
                self.swa_model.update_parameters(self.model)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            print(
                f"Epoch: {epoch + 1}, Validation F1: {val_metrics_dict['F1']:.5f}, "
                f"Validation loss: {vali_loss:.5f}"
            )
            if val_metrics_dict["F1"] > best_vali_f1:
                best_vali_f1 = val_metrics_dict["F1"]
                model_to_save = self.swa_model if self.swa else self.model
                torch.save(model_to_save.state_dict(), os.path.join(path, "rob_checkpoint.pth"))

        return self.model

    def test(self, setting, model_path=None, test=0):
        """测试鲁棒模型"""
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            print("Loading robust model")
            if model_path is None:
                model_path = os.path.join(
                    "./checkpoints", setting, "rob_checkpoint.pth"
                )

            if not os.path.exists(model_path):
                raise Exception(f"No robust model found at {model_path}")

            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        rob_test_loss, rob_test_metrics = self.test_robustness(
            test_data, test_loader, criterion, stddev=self.args.stddev
        )

        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/robust/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, F1: {val_metrics_dict['F1']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, F1: {test_metrics_dict['F1']:.5f}\n"
            f"Robust Test results --- Loss: {rob_test_loss:.5f}, "
            f"Accuracy: {rob_test_metrics['Accuracy']:.5f}, F1: {rob_test_metrics['F1']:.5f}\n"
        )

        file_name = "result_classification_robust.txt"
        with open(os.path.join(folder_path, file_name), "a") as f:
            f.write(setting + "  \n")
            f.write(
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, F1: {val_metrics_dict['F1']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, F1: {test_metrics_dict['F1']:.5f}\n"
                f"Robust Test results --- Loss: {rob_test_loss:.5f}, "
                f"Accuracy: {rob_test_metrics['Accuracy']:.5f}, F1: {rob_test_metrics['F1']:.5f}\n"
            )
            f.write("\n\n")

        return
