from tqdm.auto import tqdm
from IPython.display import clear_output
from collections import defaultdict
import torch.nn as nn
import torch
import copy
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import seaborn as sns
sns.set(style="darkgrid")


class ModelTrainer(ABC):
    def __init__(self, model, optimizer, 
                 loss_function=nn.CrossEntropyLoss(), 
                 epochs=10, 
                 save_model=None, scheduler=None, metric="acc.", early_stopping_rounds=10):

        self.model = model
        self.optimizer = self.reset_optimizer(optimizer)
        self.loss_function = loss_function
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = scheduler
        self.save_model = save_model
        self.logs = defaultdict(list)
        self.metric = metric
        self.best_val_metric = np.NINF
        self.best_iter = -1
        self.best_model = self.model
        self.early_stopping_rounds = early_stopping_rounds 

    def reset_optimizer(self, optimizer):
        return type(optimizer)(self.model.parameters())

    def set_params(self, optimizer, loss_function, epochs, 
                   save_model, scheduler,  metric,
                   early_stopping_rounds):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.scheduler = scheduler
        self.save_model = save_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds

    @abstractmethod
    def get_loss(self, output, y):
        loss = self.loss_function(output, y).float()
        return loss

    @abstractmethod
    def get_metric(self, output, y):
        prediction = self.predict_from_output(output)
        prediction, y = prediction.detach().cpu().numpy() , y.detach().cpu().numpy() 
        metric = - self.metric(y, prediction)
        return metric

    @abstractmethod
    def predict_from_output(self, output):
        prediction = output
        return prediction

    def feed_forward(self, batch):
        x = batch["img"].to(self.device)
        y = batch["target"].to(self.device)
        output = self.model(x)
        output = torch.max(output, axis=1)[0]
        return y, output

    def _train_epoch(self, dataloader):
        self.model.train()

        for batch in tqdm(dataloader):
            y, output = self.feed_forward(batch)

            loss = self.get_loss(output, y)
            metric = self.get_metric(output, y)
            
            self.logs["loss"].append(loss.item())
            self.logs["acc"].append(metric)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        logs = defaultdict(list)
        all_y = []
        all_output = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                y, output = self.feed_forward(batch)
                all_output.append(output)
                all_y.append(y)


            output = torch.cat(all_output)
            y = torch.cat(all_y)

            del all_output
            del all_y

            loss = self.get_loss(output, y)
            metric = self.get_metric(output, y)

        return {"acc" : [metric], "loss" : [loss.item()]}

    def find_lr(self, dataloader, min_lr=1e-5, max_lr=1.0, use_best=False):
        self.model.to(self.device)
        self.model.train()
        total_steps = len(dataloader)
        step = 0
        logs = defaultdict(list)

        model_state = copy.deepcopy(self.model.state_dict())
        default_lr = self.optimizer.param_groups[0]["lr"]

        for batch in tqdm(dataloader):
            t = step / total_steps
            lr = torch.exp((1 - t) * torch.log(torch.tensor(min_lr)) + t * torch.log(torch.tensor(max_lr)))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            y, output = self.feed_forward(batch)

            loss = self.get_loss(output, y)
            metric = self.get_metric(output, y)

            logs["loss"].append(loss.item())
            logs["acc"].append(metric)
            logs["lr"].append(lr.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step += 1

        self.model.load_state_dict(model_state)
        del model_state
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = default_lr

        fig, axes = fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))
        plt.sca(axes[0])
        plt.plot(logs['lr'], logs['acc'])
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Accuracy/lr")
        plt.xlabel("learning rate")
        plt.ylabel("Acc")

        plt.sca(axes[1])
        plt.plot(logs['lr'], logs['loss'])
        plt.xscale('log')
        plt.title("Loss/lr")
        plt.xlabel("learning rate")
        plt.ylabel("Loss")
        plt.ylim(top=np.quantile(logs["loss"], 0.9))
        best_lr = logs["lr"][np.argmin(logs["loss"][:np.argmax(logs["loss"])])] * 1e-1
        plt.axvline(best_lr, color="red")
        if use_best:
            self.set_lr(best_lr)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_logs(self, logs=None):
        logs = logs or self.logs
        clear_output()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))
        for i, metric in enumerate(["acc", "loss"]):
            plt.sca(axes[i])
            plt.plot(logs[metric], zorder=1)
            plt.plot(savgol_filter(logs[metric], logs["steps"][0] - 1 + (logs["steps"][0] % 2) , 1), label="train rolling", zorder=1)

            plt.scatter(logs['steps'], logs[f'val_{metric}'], marker='+', s=180, c='orange', label='val', zorder=2)

            argmax_ind = np.argmax(logs[f"val_{metric}"]) if metric != "loss" else np.argmin(logs[f"val_{metric}"])
            plt.scatter(logs['steps'][argmax_ind], logs[f'val_{metric}'][argmax_ind], marker='+', s=180, c='red', zorder=2)
            plt.plot(logs['steps'], logs[f'val_{metric}'], '--', c='yellow', label="val rolling", zorder=2)
            plt.text(logs["steps"][-1] * 0.9, logs[f"val_{metric}"][-1] * 0.9,
                     f"Val {metric}. {len(logs[f'val_{metric}'])}: %0.04f" %logs[f"val_{metric}"][-1],
                    bbox=dict(boxstyle='round', facecolor='#fff700', alpha=0.5))
        
            plt.scatter(logs['steps'][argmax_ind], logs[f"val_{metric}"][argmax_ind], marker='+', s=180, c='red', zorder=2)
            if argmax_ind != len(logs[f"val_{metric}"]) - 1:
                plt.text(logs["steps"][argmax_ind] * 0.9, logs[f"val_{metric}"][argmax_ind] *1.1, 
                        f"Val {metric}. {argmax_ind + 1}: %0.04f" %logs[f"val_{metric}"][argmax_ind],
                        bbox=dict(boxstyle='round', facecolor='#ff9900', alpha=0.5))
            
            plt.title(metric)
            plt.xlabel("steps")
            plt.ylabel("metric")
            plt.ylim(bottom=min(np.quantile(logs[metric], 0.1) - 0.1 * np.abs(np.quantile(logs[metric], 0.1)),
                                min(logs[f"val_{metric}"]) - 0.1 * np.abs(min(logs[f"val_{metric}"]))),
                     top=max(np.quantile(logs[metric], 0.9) + 0.1 * np.abs(np.quantile(logs[metric], 0.9)),
                             max(logs[f"val_{metric}"]) + 0.1 * np.abs(max(logs[f"val_{metric}"]))))
            plt.legend()

        plt.show()

    def use_best_model(self):
        self.model = self.best_model
        for log in self.logs:
            self.logs[log] = self.logs[log][:self.best_iter * 
                                                ((not (("val" in log) or ("steps" in log))) * self.logs["steps"][0] + 1)]
        self.optimizer = self.reset_optimizer(self.optimizer) #сюда нужно допелить параметры

    def predict(self, dataloader):
        self.model.eval()
        res = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                y, output = self.feed_forward(batch)

                prediction = self.predict_from_output(output)
                res.append(prediction.data)

        return torch.cat(res)

    def fit(self, train_loader, val_loader=None,
            optimizer=None, 
            loss_function=None, 
            epochs=None, 
            save_model=None,
            scheduler=None,
            metric=None,
            early_stopping_rounds=None):

        self.set_params(optimizer or self.optimizer, 
                loss_function or self.loss_function,
                epochs or self.epochs,
                save_model or self.save_model,
                scheduler or self.scheduler,
                metric or self.metric,
                early_stopping_rounds or self.early_stopping_rounds)

        self.model.to(self.device)

        epoch = 0
        epochs_wo_improvement = 0
        while (epoch < self.epochs) and (epochs_wo_improvement < self.early_stopping_rounds):
            epochs_wo_improvement += 1
            self._train_epoch(train_loader)
            
            #if val_loader:
            val_logs = self.validate(val_loader)
            for k, v in val_logs.items():
                self.logs[f'val_{k}'].extend(v)

            if self.save_model and self.logs["val_acc"][-1] > self.best_val_metric:
                torch.save(self.model.state_dict(), self.save_model)
                self.best_val_metric = self.logs["val_acc"][-1]
                self.best_model = copy.deepcopy(self.model)
                self.best_iter = len(self.logs["val_acc"])
                epochs_wo_improvement = 0

            if self.scheduler:
                self.scheduler.step()
        
            self.logs['steps'].append(len(self.logs['loss']))
            clear_output()
            self.plot_logs(self.logs)
            epoch += 1


