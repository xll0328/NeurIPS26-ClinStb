from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
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

class AutoPGD(nn.Module):
    """
    AutoPGD攻击实现，用于分类任务的鲁棒性训练
    基于"Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
    """
    def __init__(self, model, eps=0.3, alpha=None, steps=100, rho=0.75, max_restarts=2, random_start=True, device=None):
        super(AutoPGD, self).__init__()
        self.model = model
        self.eps = eps
        # 设置初始步长，如果未指定则使用eps的10%
        self.alpha = alpha if alpha is not None else eps * 0.1
        self.steps = steps
        self.rho = rho  # 用于调整步长的参数
        self.max_restarts = max_restarts  # 最大重启次数
        self.random_start = random_start
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

    def forward(self, inputs, labels, padding_mask, stddev, model_orig_outputs=None, custom_loss_fn=None):
        """
        计算对抗样本，可以使用原始输出进行对比

        参数:
            inputs: 输入数据
            labels: 真实标签
            padding_mask: 填充掩码
            stddev: 随机扰动的标准差
            model_orig_outputs: 原始输入的模型输出(可选)
            custom_loss_fn: 自定义损失函数(可选)
        返回:
            对抗样本
        """
        # 确保数据在GPU上，并转换为所需精度
        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        padding_mask = padding_mask.clone().detach().to(self.device)
        if model_orig_outputs is not None:
            model_orig_outputs = model_orig_outputs.clone().detach().to(self.device)

        # 分批处理以减少内存压力
        batch_size = inputs.size(0)
        sub_batch_size = 1024 # 增大子批量大小以提高GPU利用率

        all_adv_inputs = []

        # 清理初始缓存
        torch.cuda.empty_cache()

        # 启用混合精度计算
        with torch.cuda.amp.autocast(enabled=True):
            # 分批次处理
            for start_idx in range(0, batch_size, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, batch_size)

                # 获取当前子批次
                sub_inputs = inputs[start_idx:end_idx]
                sub_labels = labels[start_idx:end_idx]
                sub_padding_mask = padding_mask[start_idx:end_idx]
                sub_model_orig_outputs = None
                if model_orig_outputs is not None:
                    sub_model_orig_outputs = model_orig_outputs[start_idx:end_idx]

                # 初始化对抗样本
                adv_inputs = sub_inputs.clone().detach()

                # 添加初始随机扰动
                if self.random_start:
                    noise = torch.randn_like(adv_inputs, device=self.device) * stddev
                    adv_inputs = adv_inputs + noise
                    adv_inputs = torch.clamp(adv_inputs, min=sub_inputs-self.eps, max=sub_inputs+self.eps).detach()
                    del noise

                # 初始化最佳损失和最佳对抗样本
                best_loss = None
                best_adv = adv_inputs.clone()

                # 初始化步长
                step_size = self.alpha

                # 跟踪损失历史用于重启策略
                loss_history = []
                restart_counter = 0
                current_step = 0

                try:
                    # 设置为评估模式以加速推理
                    was_training = self.model.training
                    self.model.eval()

                    while current_step < self.steps:
                        # 仅在需要时清理缓存，每10步而不是每一步
                        if current_step % 10 == 0 and current_step > 0:
                            torch.cuda.empty_cache()

                        # 启用梯度计算
                        adv_inputs.requires_grad = True

                        # 前向传递
                        outputs = self.model(adv_inputs, sub_padding_mask, None, None)

                        # 计算损失 - 尝试使用更高效的损失计算
                        if custom_loss_fn is not None and sub_model_orig_outputs is not None:
                            # 使用自定义损失函数
                            cost = custom_loss_fn(outputs, sub_model_orig_outputs, sub_labels)
                        elif sub_model_orig_outputs is not None:
                            # 使用输出一致性损失
                            cost = -nn.MSELoss(reduction='sum')(outputs, sub_model_orig_outputs) / outputs.size(0)
                        else:
                            # 使用标准分类损失
                            cost = -nn.CrossEntropyLoss()(outputs, sub_labels.long())

                        # 更新最佳结果
                        with torch.no_grad():
                            current_loss = cost.item()
                            loss_history.append(current_loss)

                            if best_loss is None or current_loss > best_loss:
                                best_loss = current_loss
                                best_adv = adv_inputs.clone()

                            # 检查是否需要重启（如果最近10步没有改善）
                            if len(loss_history) >= 10:
                                recent_max = max(loss_history[-10:])
                                if recent_max <= loss_history[-10] + 1e-6 and restart_counter < self.max_restarts:
                                    # 执行重启
                                    restart_counter += 1
                                    # 使用当前最佳样本加小随机噪声重新开始
                                    adv_inputs = best_adv.clone()
                                    noise = torch.randn_like(adv_inputs, device=self.device) * stddev * 0.1
                                    adv_inputs = adv_inputs + noise
                                    adv_inputs = torch.clamp(adv_inputs, min=sub_inputs-self.eps, max=sub_inputs+self.eps).detach()

                                    # 重置步长和损失历史
                                    step_size = self.alpha
                                    loss_history = []

                                    # 释放临时变量
                                    del noise, outputs
                                    if cost.requires_grad:
                                        cost.detach_()
                                    del cost
                                    torch.cuda.empty_cache()
                                    continue

                            # 自适应步长调整
                            if len(loss_history) >= 2:
                                if current_loss > loss_history[-2]:
                                    # 损失增加，增大步长
                                    step_size *= (1.0 / self.rho)
                                else:
                                    # 损失减小，减小步长
                                    step_size *= self.rho

                                # 限制步长在合理范围内
                                step_size = min(step_size, self.eps)
                                step_size = max(step_size, self.eps * 0.01)

                        # 计算梯度 - 避免保留图和创建图，加速计算
                        grad = torch.autograd.grad(cost, adv_inputs, retain_graph=False, create_graph=False)[0]

                        # 高效更新对抗样本 - 直接使用in-place操作减少内存分配
                        adv_inputs = adv_inputs.detach() + step_size * grad.sign()

                        # 确保扰动在epsilon范围内
                        delta = torch.clamp(adv_inputs - sub_inputs, min=-self.eps, max=self.eps)
                        adv_inputs = (sub_inputs + delta).detach()

                        # 立即释放不需要的变量
                        del outputs, cost, grad, delta

                        current_step += 1

                    # 恢复模型的训练状态
                    if was_training:
                        self.model.train()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "NVML_SUCCESS" in str(e):
                        print(f"内存错误在步骤 {current_step}，返回当前最佳结果")
                        # 恢复模型的训练状态
                        if was_training:
                            self.model.train()
                    else:
                        # 恢复模型的训练状态后抛出异常
                        if was_training:
                            self.model.train()
                        raise e

                # 添加当前子批次结果
                all_adv_inputs.append(best_adv.detach())

                # 释放临时变量
                del adv_inputs, best_adv
                torch.cuda.empty_cache()

        # 高效合并所有子批次的结果
        try:
            return torch.cat(all_adv_inputs, dim=0)
        finally:
            # 最终清理
            del all_adv_inputs
            torch.cuda.empty_cache()

class IntermediateAutoPGD(nn.Module):
    """
    应用于模型中间层的AutoPGD攻击
    """
    def __init__(self, model, eps=0.3, alpha=None, steps=100, rho=0.75, max_restarts=2, random_start=True, device='cpu'):
        super(IntermediateAutoPGD, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha if alpha is not None else eps * 0.1
        self.steps = steps
        self.rho = rho
        self.max_restarts = max_restarts
        self.random_start = random_start
        self.device = device

    def forward(self, inputs, labels, padding_mask, stddev, custom_loss_fn=None):
        """
        对中间层特征应用AutoPGD扰动
        """
        # 第一步：执行正常前向传递以获取中间层特征
        _ = self.model(inputs, padding_mask, None, None)
        intermediate_features = self.model.get_intermediate_features()

        # 确保中间层特征已正确获取
        if intermediate_features is None:
            raise ValueError("无法获取中间层特征")

        # 克隆中间层特征，作为扰动的起点
        perturbed_features = [x.clone().detach() for x in intermediate_features]

        # 添加初始随机扰动
        if self.random_start:
            for i in range(len(perturbed_features)):
                noise = torch.randn_like(perturbed_features[i]) * stddev * 0.1
                perturbed_features[i] = perturbed_features[i] + noise

        # 初始化最佳损失和最佳扰动
        best_loss = None
        best_perturbed_features = [x.clone() for x in perturbed_features]

        # 初始化步长和损失历史
        step_size = self.alpha
        loss_history = []
        restart_counter = 0
        current_step = 0

        while current_step < self.steps:
            # 为每个特征启用梯度计算
            for i in range(len(perturbed_features)):
                perturbed_features[i].requires_grad = True

            # 设置中间层扰动并执行前向传递
            delta_features = [p - o for p, o in zip(perturbed_features, intermediate_features)]
            self.model.set_intermediate_perturbation(delta_features)
            outputs = self.model(inputs, padding_mask, None, None)

            # 计算攻击损失
            if custom_loss_fn is not None:
                # 清除扰动标志，获取原始输出进行比较
                self.model.clear_intermediate_perturbation()
                with torch.no_grad():
                    orig_outputs = self.model(inputs, padding_mask, None, None)

                # 重新设置扰动并计算损失
                self.model.set_intermediate_perturbation(delta_features)
                cost = custom_loss_fn(outputs, orig_outputs, labels)
            else:
                # 使用默认的负交叉熵损失
                cost = -nn.CrossEntropyLoss()(outputs, labels.long())

            # 更新最佳结果
            with torch.no_grad():
                current_loss = cost.item()
                loss_history.append(current_loss)

                if best_loss is None or current_loss > best_loss:
                    best_loss = current_loss
                    best_perturbed_features = [x.clone() for x in perturbed_features]

                # 检查是否需要重启
                if len(loss_history) >= 10:
                    recent_max = max(loss_history[-10:])
                    if recent_max <= loss_history[-10] + 1e-6 and restart_counter < self.max_restarts:
                        restart_counter += 1
                        perturbed_features = [x.clone() for x in best_perturbed_features]
                        for i in range(len(perturbed_features)):
                            noise = torch.randn_like(perturbed_features[i]) * stddev * 0.1
                            perturbed_features[i] = perturbed_features[i] + noise

                        step_size = self.alpha
                        loss_history = []
                        continue

                # 自适应步长调整
                if len(loss_history) >= 2:
                    if current_loss > loss_history[-2]:
                        step_size *= (1.0 / self.rho)
                    else:
                        step_size *= self.rho

                    step_size = min(step_size, self.eps)
                    step_size = max(step_size, self.eps * 0.01)

            # 计算梯度并更新扰动
            grads = torch.autograd.grad(cost, perturbed_features, retain_graph=False, create_graph=False)

            # 更新每个特征的扰动
            for i in range(len(perturbed_features)):
                perturbed_features[i] = perturbed_features[i].detach() + step_size * grads[i].sign()

                # 确保扰动在epsilon范围内
                delta = torch.clamp(
                    perturbed_features[i] - intermediate_features[i],
                    min=-self.eps,
                    max=self.eps
                )
                perturbed_features[i] = (intermediate_features[i] + delta).detach()

            current_step += 1

        # 计算最终的扰动
        delta_features = [p - o for p, o in zip(best_perturbed_features, intermediate_features)]

        # 清除中间扰动标志
        self.model.clear_intermediate_perturbation()

        return delta_features

class PGD(nn.Module):
    """
    PGD攻击实现，用于分类任务的鲁棒性训练
    """
    def __init__(self, model, eps=0.3, alpha=0.03, steps=50, random_start=True, device='cpu'):
        super(PGD, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device

    # 在PGD类中修改forward方法，使用特定损失更新扰动
    def forward(self, inputs, labels, padding_mask, stddev, model_orig_outputs=None):
        """
        计算对抗样本，可以使用原始输出进行对比
        """
        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        padding_mask = padding_mask.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_inputs = inputs.clone().detach()

        if self.random_start:
            # 添加高斯噪声作为初始扰动
            noise = torch.randn_like(adv_inputs) * stddev
            adv_inputs = adv_inputs + noise
            adv_inputs = adv_inputs.detach()

        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            outputs = self.model(adv_inputs, padding_mask, None, None)

            if model_orig_outputs is not None:
                # 如果提供原始输出，使用输出一致性损失
                cost = nn.MSELoss()(outputs, model_orig_outputs)
            else:
                # 否则使用标准分类损失
                cost = loss(outputs, labels.long())

            grad = torch.autograd.grad(cost, adv_inputs, retain_graph=False, create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs = (inputs + delta).detach()

        return adv_inputs


class Exp_Classification_Rob(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # 模型构建与原始分类任务相同
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len
        self.args.pred_len = 0
        self.args.enc_in = test_data.X.shape[2]
        self.args.num_class = len(np.unique(test_data.y))

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def compare_models_on_dataset(self, student_model, teacher_model, data_loader, stddev=None):
        """在整个数据集上比较student和teacher模型在不同扰动级别下的表现"""
        # 预定义扰动级别
        noise_levels = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        # 初始化结果字典
        results = {}

        # 设置为评估模式
        student_model.eval()
        if teacher_model is not None:
            teacher_model.eval()

        # 收集所有预测结果和真实标签
        total = 0

        # 收集所有预测和标签，用于计算指标
        student_orig_preds = []
        student_orig_labels = []
        student_adv_preds = {level: [] for level in noise_levels}
        student_adv_labels = {level: [] for level in noise_levels}

        teacher_orig_preds = [] if teacher_model is not None else None
        teacher_orig_labels = [] if teacher_model is not None else None
        teacher_adv_preds = {level: [] for level in noise_levels} if teacher_model is not None else None
        teacher_adv_labels = {level: [] for level in noise_levels} if teacher_model is not None else None

        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                total += label.size(0)

                # 原始数据的预测
                outputs_student_orig = student_model(batch_x, padding_mask, None, None)
                _, pred_student_orig = torch.max(outputs_student_orig, 1)

                # 收集原始数据的预测和标签
                student_orig_preds.append(outputs_student_orig.detach())
                student_orig_labels.append(label)

                if teacher_model is not None:
                    outputs_teacher_orig = teacher_model(batch_x, padding_mask, None, None)
                    _, pred_teacher_orig = torch.max(outputs_teacher_orig, 1)
                    teacher_orig_preds.append(outputs_teacher_orig.detach())
                    teacher_orig_labels.append(label)

                # 对每个扰动级别进行测试
                for level in noise_levels:
                    # 生成固定扰动数据
                    noise = torch.randn_like(batch_x) * level
                    perturbed_batch_x = batch_x + noise

                    # Student模型在扰动数据上的预测
                    outputs_student_adv = student_model(perturbed_batch_x, padding_mask, None, None)
                    student_adv_preds[level].append(outputs_student_adv.detach())
                    student_adv_labels[level].append(label)

                    # Teacher模型在扰动数据上的预测(如果有)
                    if teacher_model is not None:
                        outputs_teacher_adv = teacher_model(perturbed_batch_x, padding_mask, None, None)
                        teacher_adv_preds[level].append(outputs_teacher_adv.detach())
                        teacher_adv_labels[level].append(label)

        # 计算原始数据的所有指标 - Student模型
        student_orig_metrics = self._calculate_metrics(
            student_orig_preds,
            student_orig_labels,
            self.args.num_class
        )

        # 保存Student原始指标
        for metric_name, metric_value in student_orig_metrics.items():
            results[f"Student原始_{metric_name}"] = metric_value

        # 计算每个扰动级别的指标
        for level in noise_levels:
            student_adv_metrics = self._calculate_metrics(
                student_adv_preds[level],
                student_adv_labels[level],
                self.args.num_class
            )

            # 保存Student扰动指标
            for metric_name, metric_value in student_adv_metrics.items():
                results[f"Student扰动_{level}_{metric_name}"] = metric_value

        # 如果有Teacher模型，计算Teacher的指标
        if teacher_model is not None:
            teacher_orig_metrics = self._calculate_metrics(
                teacher_orig_preds,
                teacher_orig_labels,
                self.args.num_class
            )

            # 保存Teacher原始指标
            for metric_name, metric_value in teacher_orig_metrics.items():
                results[f"Teacher原始_{metric_name}"] = metric_value

            for level in noise_levels:
                teacher_adv_metrics = self._calculate_metrics(
                    teacher_adv_preds[level],
                    teacher_adv_labels[level],
                    self.args.num_class
                )

                # 保存Teacher扰动指标
                for metric_name, metric_value in teacher_adv_metrics.items():
                    results[f"Teacher扰动_{level}_{metric_name}"] = metric_value

        # 为了向后兼容，保留原来的键名
        results["Student原始"] = results["Student原始_Accuracy"]
        for level in noise_levels:
            results[f"Student扰动_{level}"] = results[f"Student扰动_{level}_Accuracy"]

        if teacher_model is not None:
            results["Teacher原始"] = results["Teacher原始_Accuracy"]
            for level in noise_levels:
                results[f"Teacher扰动_{level}"] = results[f"Teacher扰动_{level}_Accuracy"]

        return results

    def _calculate_metrics(self, preds_list, labels_list, num_classes):
        """计算完整的指标集合"""
        preds = torch.cat(preds_list, 0)
        labels = torch.cat(labels_list, 0)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = labels.flatten().cpu().numpy()
        probs = probs.cpu().numpy()

        # 生成one-hot编码
        trues_onehot = torch.nn.functional.one_hot(
            labels.reshape(-1).to(torch.long),
            num_classes=num_classes
        ).float().cpu().numpy()

        # 计算各项指标
        metrics = {
            "Accuracy": accuracy_score(trues, predictions) * 100,  # 转换为百分比
            "Precision": precision_score(trues, predictions, average="macro") * 100,
            "Recall": recall_score(trues, predictions, average="macro") * 100,
            "F1": f1_score(trues, predictions, average="macro") * 100,
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr") * 100,
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro") * 100
        }

        return metrics

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
        probs = torch.nn.functional.softmax(preds)
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

        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()

        return total_loss, metrics_dict

    def test_robustness(self, test_data, test_loader, criterion, stddev=2.0):
        """测试模型在序列分解噪声扰动下的表现"""
        total_loss = []
        preds = []
        trues = []

        # 导入series_decomp
        from layers.Autoformer_EncDec import series_decomp

        # 创建序列分解实例
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

                # 使用series_decomp进行时间序列分解
                residual, trend = perturbation_avg(batch_x)

                # 使用残差部分作为扰动，乘以标准差系数
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
        probs = torch.nn.functional.softmax(preds)
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

        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()

        return total_loss, metrics_dict

    # def test_robustness(self, test_data, test_loader, criterion, stddev=2.0):
    #     """测试模型在不同类型噪声扰动下的表现"""
    #     total_loss = []
    #     preds = []
    #     trues = []

    #     # 定义简单的序列分解函数（如果您有import的series_decomp，可以直接使用）
    #     def simple_decomp(x, window=13):
    #         # 这是一个简单的移动平均实现，用于模拟series_decomp功能
    #         # 实际应用中请替换为您项目中的实际series_decomp函数
    #         b, t, c = x.shape
    #         padding = window // 2
    #         # 使用卷积实现移动平均
    #         kernel = torch.ones(1, 1, window, device=x.device) / window
    #         x_reshaped = x.permute(0, 2, 1).unsqueeze(1)  # [b, 1, c, t]
    #         # 使用反射填充避免边界效应
    #         x_padded = torch.nn.functional.pad(x_reshaped, (padding, padding), mode='reflect')
    #         # 应用卷积
    #         trend = torch.nn.functional.conv2d(x_padded, kernel, stride=1, padding=0)
    #         trend = trend.squeeze(1).permute(0, 2, 1)  # 恢复到 [b, t, c]
    #         residual = x - trend
    #         return residual, trend

    #     if self.swa:
    #         self.swa_model.eval()
    #     else:
    #         self.model.eval()

    #     with torch.no_grad():
    #         for i, (batch_x, label, padding_mask) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             padding_mask = padding_mask.float().to(self.device)
    #             label = label.to(self.device)

    #             # 使用序列分解而不是简单的高斯噪声
    #             residual, trend = simple_decomp(batch_x, window=13)
    #             # 使用残差作为扰动基础，并乘以标准差系数
    #             noisy_batch_x = batch_x + residual * stddev

    #             # 可以添加额外的扰动类型
    #             # 例如：添加一些高频振荡
    #             freq_noise = torch.sin(torch.linspace(0, 10*np.pi, batch_x.shape[1]).to(self.device))
    #             freq_noise = freq_noise.view(1, -1, 1).repeat(batch_x.shape[0], 1, batch_x.shape[2])
    #             noisy_batch_x = noisy_batch_x + freq_noise * (stddev * 0.2)

    #             if self.swa:
    #                 outputs = self.swa_model(noisy_batch_x, padding_mask, None, None)
    #             else:
    #                 outputs = self.model(noisy_batch_x, padding_mask, None, None)

    #             pred = outputs.detach().cpu()
    #             loss = criterion(pred, label.long().cpu())
    #             total_loss.append(loss)

    #             preds.append(outputs.detach())
    #             trues.append(label)

    #     total_loss = np.average(total_loss)

    #     preds = torch.cat(preds, 0)
    #     trues = torch.cat(trues, 0)
    #     probs = torch.nn.functional.softmax(preds)
    #     trues_onehot = (
    #         torch.nn.functional.one_hot(
    #             trues.reshape(-1).to(torch.long),
    #             num_classes=self.args.num_class,
    #         )
    #         .float()
    #         .cpu()
    #         .numpy()
    #     )

    #     predictions = torch.argmax(probs, dim=1).cpu().numpy()
    #     probs = probs.cpu().numpy()
    #     trues = trues.flatten().cpu().numpy()

    #     metrics_dict = {
    #         "Accuracy": accuracy_score(trues, predictions),
    #         "Precision": precision_score(trues, predictions, average="macro"),
    #         "Recall": recall_score(trues, predictions, average="macro"),
    #         "F1": f1_score(trues, predictions, average="macro"),
    #         "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
    #         "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
    #     }

    #     if self.swa:
    #         self.swa_model.train()
    #     else:
    #         self.model.train()

    #     return total_loss, metrics_dict

    def calculate_accuracy(outputs, labels):
        """计算分类准确率"""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total * 100.0

    def train(self, setting, pretrained_model_path=None):
        """用于鲁棒训练的方法"""
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        # if self.args.seed == 41:
        # elif self.args.seed == 42:
        # elif self.args.seed == 43:
        # elif self.args.seed == 44:
        # elif self.args.seed == 45:
        # pretrained_model_path = os.path.join('./checkpoints/', setting, 'checkpoint.pth')

        # 设置保存路径
        # path = (
        #     "./checkpoints/"
        #     + self.args.task_name
        #     + "/"
        #     + self.args.model_id
        #     + "/"
        #     + self.args.model
        #     + "/"
        #     + setting
        #     + "/"
        # )
        # path = ("./checkpoints/"
        #         + "/"
        #         + setting
        #         + "/")
        # if pretrained_model_path:
        #     path = os.path.dirname(pretrained_model_path) + "/"
        # else:
        #     path = ("./checkpoints/" + "/" + setting + "/")
        # path = ("./checkpoints/" + "/" + setting + "/")
        path = (
            "./checkpoints/"
            + setting
            + "/"
        )
        pretrained_model_path = os.path.join(path, 'checkpoint.pth')
        if not os.path.exists(path):
            os.makedirs(path)

        # 加载预训练模型作为teacher模型

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path, map_location=self.device)

            # 处理带有module.前缀的状态字典
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]  # 移除'module.'前缀
                    else:
                        name = k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            # 尝试加载处理后的状态字典
            self.model.load_state_dict(state_dict, strict=False)

            # 创建teacher模型
            teacher_model = copy.deepcopy(self.model)
            teacher_model.is_teacher = True  # 标记为教师模型
            teacher_model.eval()
            # 冻结teacher模型的参数
            for param in teacher_model.parameters():
                param.requires_grad = False
        else:
            print(f"Pretrained model not found at {pretrained_model_path}. Training from scratch.")
            teacher_model = None

        # # 初始化PGD攻击
        # pgd_attack = PGD(
        #     self.model,
        #     eps=self.args.eps,
        #     alpha=self.args.alpha,
        #     steps=self.args.steps,
        #     random_start=True,
        #     device=self.device
        # )
        def custom_attack_loss(outputs_adv, outputs_orig, labels):
            """
            自定义损失函数，与训练时使用的损失函数类似
            输入:
                outputs_adv: 扰动数据的模型输出
                outputs_orig: 原始数据的模型输出
                labels: 真实标签
            返回:
                组合损失值 - 对于攻击，我们希望返回一个正值(越大越好)
            """
            # 1. 分类损失 - 希望预测错误，所以是负的交叉熵
            loss_cls = -nn.CrossEntropyLoss()(outputs_adv, labels.long())

            # 2. 一致性损失 - 希望输出差异最大
            loss_adv = -nn.MSELoss()(nn.Softmax(dim=1)(outputs_adv),
                                    nn.Softmax(dim=1)(outputs_orig)) * 10 * self.args.w_adv

            # 对于攻击，我们返回一个需要被最大化的值
            return loss_cls + loss_adv

        # 初始化AutoPGD攻击
        autopgd_attack = AutoPGD(
            self.model,
            eps=self.args.eps,
            alpha=self.args.alpha,
            steps=self.args.steps,
            rho=self.args.rho,
            max_restarts=self.args.max_restarts,
            random_start=True,
            device=self.device
        )

        # # 初始化中间层AutoPGD攻击
        # intermediate_autopgd = IntermediateAutoPGD(
        #     self.model,
        #     eps=self.args.eps * 0.5,  # 可以使用不同的扰动强度
        #     alpha=self.args.alpha * 0.5,
        #     steps=self.args.steps,
        #     rho=self.args.rho,
        #     max_restarts=self.args.max_restarts,
        #     random_start=True,
        #     device=self.device
        # )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        time_now = time.time()
        train_steps = len(train_loader)

        # 不使用早停，而是固定训练轮数进行鲁棒训练
        training_epochs = 10  # 可以根据需要调整鲁棒训练的轮数

        # 用于记录最佳验证性能的模型
        best_vali_f1 = float('-inf')

        train_loss = []
        train_loss_orig = []  # 原始数据损失
        train_loss_cls = []  # 分类损失
        train_loss_adv = []  # 一致性损失
        train_loss_distill = []  # 蒸馏损失

        for epoch in range(training_epochs):
            iter_count = 0
            epoch_loss = []  # 记录整个epoch的损失
            epoch_loss_orig = []
            epoch_loss_cls = []  # 分类损失
            epoch_loss_adv = []  # 一致性损失
            epoch_loss_distill = []  # 蒸馏损失

            self.model.train()
            epoch_time = time.time()

            # 第一步：为当前epoch生成一次扰动数据（所有batch）
            print(f"Epoch {epoch+1}/{training_epochs}: 生成扰动数据...")
            all_perturbed_data = []
            all_orig_outputs = []
            all_labels = []
            all_padding_masks = []

            # 首先为所有批次生成扰动数据
            # with torch.no_grad():

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 获取原始输出 - 这部分不需要梯度
                with torch.no_grad():
                    outputs_orig = self.model(batch_x, padding_mask, None, None)

                # 生成扰动数据 - 这部分需要梯度
                perturbed_batch_x = autopgd_attack(
                    batch_x, label, padding_mask, self.args.stddev,
                    outputs_orig, custom_attack_loss
                )

                # 保存扰动数据和原始输出供多次训练使用
                all_perturbed_data.append(perturbed_batch_x.clone())
                all_orig_outputs.append(outputs_orig.clone())
                all_labels.append(label.clone())
                all_padding_masks.append(padding_mask.clone())

                # 显示进度
                if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                    print(f"  已生成 {i+1}/{len(train_loader)} 批次的扰动数据")

            # 第二步：对同一扰动数据进行多次训练迭代
            # 使用args.iters_per_epoch参数控制内层循环次数
            iters_per_epoch = getattr(self.args, 'iters_per_epoch', 10)  # 默认为1
            print(f"Epoch {epoch+1}: 对扰动数据进行 {iters_per_epoch} 次迭代训练")

            for iter_in_epoch in range(iters_per_epoch):
                iter_loss = []
                iter_loss_orig = []
                iter_loss_cls = []
                iter_loss_adv = []
                iter_loss_distill = []

                for i in range(len(all_perturbed_data)):
                    iter_count += 1
                    model_optim.zero_grad()

                    # 获取数据
                    perturbed_batch_x = all_perturbed_data[i]
                    outputs_orig = all_orig_outputs[i]
                    label = all_labels[i]
                    padding_mask = all_padding_masks[i]

                    batch_x = batch_x.float().to(self.device) if 'batch_x' in locals() else None
                    if batch_x is None or batch_x.size(0) != perturbed_batch_x.size(0):
                        # 获取对应的原始数据
                        # 这里假设我们存储了对应的原始数据，如果没有则需要从数据加载器中重新获取
                        with torch.no_grad():
                            batch_x = perturbed_batch_x.clone() - (perturbed_batch_x.clone() - perturbed_batch_x.clone()).detach()

                    outputs_orig_current = self.model(batch_x, padding_mask, None, None)
                    # 前向传播 - 使用扰动数据
                    outputs_adv = self.model(perturbed_batch_x, padding_mask, None, None)

                    # 损失函数计算
                    # 0. 原始数据学生模型的输出和ground truth对比 - 原始分类损失 (新增)
                    loss_orig = criterion(outputs_orig_current, label.long())
                    iter_loss_orig.append(loss_orig.item())

                    # 1. 扰动数据学生模型的输出和ground truth对比 - 主分类损失
                    loss_cls = criterion(outputs_adv, label.long())
                    iter_loss_cls.append(loss_cls.item())

                    # 2. 扰动数据学生模型的输出和原始数据学生模型的输出对比 - 一致性损失
                    loss_adv = nn.MSELoss()(nn.Softmax(dim=1)(outputs_adv),
                                            nn.Softmax(dim=1)(outputs_orig)) * 10 * self.args.w_adv
                    iter_loss_adv.append(loss_adv.item())

                    # 3. 如果有teacher模型，原始数据学生模型的输出和教师模型的输出对比 - 知识蒸馏损失
                    if teacher_model is not None:
                        with torch.no_grad():
                            outputs_teacher = teacher_model(all_perturbed_data[i], padding_mask, None, None)

                        loss_distill = nn.KLDivLoss(reduction='batchmean')(
                            nn.LogSoftmax(dim=1)(outputs_adv),
                            nn.Softmax(dim=1)(outputs_teacher)
                        ) * self.args.w_cls
                        iter_loss_distill.append(loss_distill.item())
                    else:
                        loss_distill = 0
                        iter_loss_distill.append(0)

                    # 总损失
                    loss = 3 * loss_orig + loss_cls + loss_adv + loss_distill
                    iter_loss.append(loss.item())

                    # 反向传播和优化
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

                    # 每50个批次显示进度
                    if (i + 1) % 50 == 0 or (i + 1) == len(all_perturbed_data):
                        print(f"  内部迭代 {iter_in_epoch+1}/{iters_per_epoch}, 批次 {i+1}/{len(all_perturbed_data)}, "
                            f"损失: {np.mean(iter_loss[-50:]):.5f}")

                # 记录当前迭代的平均损失
                epoch_loss.extend(iter_loss)
                epoch_loss_orig.extend(iter_loss_orig)
                epoch_loss_cls.extend(iter_loss_cls)
                epoch_loss_adv.extend(iter_loss_adv)
                epoch_loss_distill.extend(iter_loss_distill)

                # 每次内部迭代结束后打印平均损失
                print(f"  Epoch {epoch+1}, 内部迭代 {iter_in_epoch+1}/{iters_per_epoch} 完成")
                print(f"  平均损失 -原始分类: {np.mean(iter_loss_orig):.5f}, 扰动分类: {np.mean(iter_loss_cls):.5f}, 一致性: {np.mean(iter_loss_adv):.5f}, "
                    f"蒸馏: {np.mean(iter_loss_distill):.5f}, 总计: {np.mean(iter_loss):.5f}")

            # 更新SWA模型
            if self.swa:
                self.swa_model.update_parameters(self.model)

            # 计算并打印epoch平均损失
            train_loss = np.average(epoch_loss)
            train_loss_orig = np.average(epoch_loss_orig)
            train_loss_cls = np.average(epoch_loss_cls)
            train_loss_adv = np.average(epoch_loss_adv)
            train_loss_distill = np.average(epoch_loss_distill)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {} | 平均损失 - 原始: {:.5f}, 分类: {:.5f}, 一致性: {:.5f}, 蒸馏: {:.5f}, 总计: {:.5f}".format(
                epoch + 1, train_loss_orig, train_loss_cls, train_loss_adv, train_loss_distill, train_loss
            ))
            train_loss = np.average(train_loss)
            # 在每个epoch结束时，验证完成后，添加以下代码

            # 在打印标准验证结果后添加
            # 进行详细的模型对比评估
            # 先执行验证
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
            rob_test_loss, rob_test_metrics = self.test_robustness(
                test_data, test_loader, criterion, stddev=self.args.stddev
            )

            # # 先打印标准结果
            # print(
            #     f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.5f}\n"
            #     f"Validation results --- Loss: {vali_loss:.5f}, "
            #     f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, F1: {val_metrics_dict['F1']:.5f}\n"
            #     f"Test results --- Loss: {test_loss:.5f}, "
            #     f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, F1: {test_metrics_dict['F1']:.5f}\n"
            #     f"Robust Test results --- Loss: {rob_test_loss:.5f}, "
            #     f"Accuracy: {rob_test_metrics['Accuracy']:.5f}, F1: {rob_test_metrics['F1']:.5f}\n"
            # )
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f}, "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
                f"Robust Test results --- Loss: {rob_test_loss:.5f}, "
                f"Accuracy: {rob_test_metrics['Accuracy']:.5f}, "
                f"Precision: {rob_test_metrics['Precision']:.5f}, "
                f"Recall: {rob_test_metrics['Recall']:.5f}, "
                f"F1: {rob_test_metrics['F1']:.5f}, "
                f"AUROC: {rob_test_metrics['AUROC']:.5f}, "
                f"AUPRC: {rob_test_metrics['AUPRC']:.5f}\n"
            )
            # 其余代码不变
            print("\n===== 模型在测试集上的详细性能对比 =====")
            comparison_results = self.compare_models_on_dataset(
                self.model if not self.swa else self.swa_model,
                teacher_model,
                test_loader
            )

            # 打印原始数据准确率
            print(f"\n----- 原始数据性能 -----")
            print(f"Student模型 - 准确率: {comparison_results['Student原始']:.2f}%, "
                f"精确率: {comparison_results['Student原始_Precision']:.2f}%, "
                f"召回率: {comparison_results['Student原始_Recall']:.2f}%, "
                f"F1: {comparison_results['Student原始_F1']:.2f}%, "
                f"AUROC: {comparison_results['Student原始_AUROC']:.2f}%, "
                f"AUPRC: {comparison_results['Student原始_AUPRC']:.2f}%")

            if teacher_model is not None:
                print(f"Teacher模型 - 准确率: {comparison_results['Teacher原始']:.2f}%, "
                    f"精确率: {comparison_results['Teacher原始_Precision']:.2f}%, "
                    f"召回率: {comparison_results['Teacher原始_Recall']:.2f}%, "
                    f"F1: {comparison_results['Teacher原始_F1']:.2f}%, "
                    f"AUROC: {comparison_results['Teacher原始_AUROC']:.2f}%, "
                    f"AUPRC: {comparison_results['Teacher原始_AUPRC']:.2f}%")

                # 计算原始数据上的差异
                student_vs_teacher_orig_diff = comparison_results["Student原始"] - comparison_results["Teacher原始"]
                if student_vs_teacher_orig_diff > 0:
                    print(f"学生模型在原始数据下比教师模型的准确率高: {student_vs_teacher_orig_diff:.2f}%")
                else:
                    print(f"学生模型在原始数据下比教师模型的准确率低: {-student_vs_teacher_orig_diff:.2f}%")

            print("\n----- 不同扰动级别下的性能 -----")

            # 打印不同扰动级别的性能
            # noise_levels = [0.1, 0.5, 1.0, 1.5, 2.0]
            noise_levels = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            for level in noise_levels:
                print(f"\n扰动级别: {level}")
                student_acc = comparison_results[f"Student扰动_{level}"]
                student_drop = comparison_results["Student原始"] - student_acc

                print(f"Student模型在扰动级别: {level}下 - 准确率: {student_acc:.2f}%, 下降: {student_drop:.2f}%, "
                    f"精确率: {comparison_results[f'Student扰动_{level}_Precision']:.2f}%, "
                    f"召回率: {comparison_results[f'Student扰动_{level}_Recall']:.2f}%, "
                    f"F1: {comparison_results[f'Student扰动_{level}_F1']:.2f}%, "
                    f"AUROC: {comparison_results[f'Student扰动_{level}_AUROC']:.2f}%, "
                    f"AUPRC: {comparison_results[f'Student扰动_{level}_AUPRC']:.2f}%")

                if teacher_model is not None:
                    teacher_acc = comparison_results[f"Teacher扰动_{level}"]
                    teacher_drop = comparison_results["Teacher原始"] - teacher_acc

                    print(f"Teacher模型在扰动级别: {level}下 - 准确率: {teacher_acc:.2f}%, 下降: {teacher_drop:.2f}%, "
                        f"精确率: {comparison_results[f'Teacher扰动_{level}_Precision']:.2f}%, "
                        f"召回率: {comparison_results[f'Teacher扰动_{level}_Recall']:.2f}%, "
                        f"F1: {comparison_results[f'Teacher扰动_{level}_F1']:.2f}%, "
                        f"AUROC: {comparison_results[f'Teacher扰动_{level}_AUROC']:.2f}%, "
                        f"AUPRC: {comparison_results[f'Teacher扰动_{level}_AUPRC']:.2f}%")

                    # 比较学生和教师模型在当前扰动级别的性能
                    if student_acc > teacher_acc:
                        print(f"学生模型在扰动级别{level}上比教师模型准确率高: {student_acc - teacher_acc:.2f}%")
                    else:
                        print(f"学生模型在扰动级别{level}上比教师模型准确率低: {teacher_acc - student_acc:.2f}%")

                    # 比较鲁棒性改进
                    if student_drop < teacher_drop:
                        improvement = teacher_drop - student_drop
                        print(f"鲁棒性改进: {improvement:.2f}% (相比Teacher模型)")
                    else:
                        degradation = student_drop - teacher_drop
                        print(f"鲁棒性下降: {degradation:.2f}% (相比Teacher模型)")

            print("=======================================\n")

            # 保存表现最好的模型
            if val_metrics_dict['F1'] > best_vali_f1:
                best_vali_f1 = val_metrics_dict['F1']
                if self.swa:
                    torch.save(self.swa_model.state_dict(), path + "rob_checkpoint.pth")
                else:
                    torch.save(self.model.state_dict(), path + "rob_checkpoint.pth")
                print(f"Model saved at epoch {epoch+1} with validation F1: {best_vali_f1:.5f}")
        # 在epoch结束时，计算并打印平均损失
        # 在epoch结束部分添加这些行
        avg_loss_cls = np.average(train_loss_cls)
        avg_loss_adv = np.average(train_loss_adv)
        avg_loss_distill = np.average(train_loss_distill)
        avg_loss_total = np.average(train_loss)

        print("Epoch: {} | 平均损失 - 分类: {:.5f}, 一致性: {:.5f}, 蒸馏: {:.5f}, 总计: {:.5f}".format(
            epoch + 1, avg_loss_cls, avg_loss_adv, avg_loss_distill, avg_loss_total
        ))
        print("Robust training finished!")

        return

    def test(self, setting, model_path=None, test=0):
        """测试鲁棒模型"""
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            print("Loading robust model")
            if model_path is None:
                path = (
                    "./checkpoints/"
                    + self.args.task_name
                    + "/"
                    + self.args.model_id
                    + "/"
                    + self.args.model
                    + "/"
                    + setting
                    + "/"
                )
                model_path = path + "rob_checkpoint.pth"

            if not os.path.exists(model_path):
                raise Exception(f"No robust model found at {model_path}")

            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # 测试模型在噪声扰动下的鲁棒性
        rob_test_loss, rob_test_metrics = self.test_robustness(
            test_data, test_loader, criterion, stddev=self.args.stddev
        )

        # 结果保存
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