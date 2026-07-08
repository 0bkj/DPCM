import os

import einops
import torch.nn as nn
import torch.optim as optim
from datasets.loader import load_data_and_data_loaders
import utils as utils
import numpy as np
import torch
import logging
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import build_memory_index, evaluate, get_anomaly_segments
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from seriate import seriate
from python_tsp.heuristics import solve_tsp_simulated_annealing
import shutil
logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"@ Using device: {self.device}")
        self.create_model()
        self.shuffle_channel_flag = False

    def load_data(self, split):
        """
        Load data and define batch data loaders
        """
        if split == "train":
            if not hasattr(self, "train_dataset") and not hasattr(self, "train_loader"):
                dataset, loader = load_data_and_data_loaders(
                    file=self.args.dataset,
                    split=split,
                    window_size=self.args.window_size,
                    step=1,
                    batch_size=self.args.batch_size,
                    data_ratio=1.0
                )
                self.train_dataset = dataset
                self.train_loader = loader
            else:
                dataset, loader = self.train_dataset, self.train_loader
        elif split == "test":
            if not hasattr(self, "test_dataset") and not hasattr(self, "test_loader"):
                dataset, loader = load_data_and_data_loaders(
                    file=self.args.dataset,
                    split=split,
                    window_size=self.args.window_size,
                    step=self.args.window_size,
                    batch_size=self.args.batch_size,
                    data_ratio=1.0
                )
                self.test_dataset = dataset
                self.test_loader = loader
            else:
                dataset, loader = self.test_dataset, self.test_loader   
        elif split == "val":
            if not hasattr(self, "val_dataset") and not hasattr(self, "val_loader"):
                dataset, loader = load_data_and_data_loaders(
                    file=self.args.dataset,
                    split=split,
                    window_size=self.args.window_size,
                    step=self.args.window_size,
                    batch_size=self.args.batch_size,
                    data_ratio=1.0
                )
                self.val_dataset = dataset
                self.val_loader = loader
            else:
                dataset, loader = self.val_dataset, self.val_loader
        return dataset, loader

    def create_model(self):
        """
        Set up VQ-VAE model with components defined in ./models/ folder
        """
        from models.model import Model
        self.model = Model(
            self.args
        ).to(self.device)

    def load_model(self):
        if not hasattr(self, "loaded"):
            self.model.load_state_dict(
                torch.load("./results/" + self.args.filename + ".pth", weights_only=False)["model"]
            )
            self.model.eval()
            print("@ loaded VQ-VAE model from ./results/" + self.args.filename + ".pth")
            print("@ set model to eval mode")
            self.loaded = True
        else:
            print("@ model already loaded")
        
    def train_update(self, x):
        """
        Forward pass of model
        """
        (
            embedding_loss,
            x_hat,
            perplexity1,
            perplexity2,
        ) = self.model(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + embedding_loss
        # results
        if hasattr(self,"results"):
            self.results["recon_errors"] = recon_loss.cpu().detach().numpy()
            self.results["time_perplexities"] = perplexity1.cpu().detach().numpy()
            self.results["freq_perplexities"] = perplexity2.cpu().detach().numpy()
            self.results["emb_loss"] = embedding_loss.cpu().detach().numpy()
        return loss

    def test_update(self, x):

        (
            embedding_loss,
            x_hat,
            perplexity1,
            perplexity2
        ) = self.model(x)
        # std = x.std(dim=1, keepdim=True) + 1e-6
        # recon_loss = ((x_hat - x)** 2 / std).mean(dim=-1)
        recon_loss = torch.mean((x_hat - x)**2,dim=-1)
        return recon_loss,x_hat
        
    def save_indices_update(self,x):
        b,t,c = x.shape
        (
            embedding_loss,
            x_hat,
            perplexity1,
            perplexity2,
            time_indices,
            freq_indices
        ) = self.model(x,indices=True)
        time_indices = time_indices[..., 0:1]
        freq_indices = freq_indices[..., 0:1]
        time_indices = einops.rearrange(time_indices, '(b c) t k -> (b t k) c', b=b, c=c)
        freq_indices = einops.rearrange(freq_indices, '(b c) t k -> (b t k) c', b=b, c=c)
        self.time_indices.append(time_indices)
        self.freq_indices.append(freq_indices)
        
    def test_template(self, dataset, loader):
        """
        test template
        """
        ori_data = []
        recon_data = []
        labels = []
        losses = []
        self.model.eval()
        with torch.inference_mode():
            for x, label in loader:
                x = x.to(self.device, non_blocking=True)
                recon_loss,x_hat = self.test_update(x)
                ori_data.append(x.cpu().detach().numpy())
                recon_data.append(x_hat.cpu().detach().numpy())
                labels.append(label)
                losses.append(recon_loss.cpu().detach().numpy())
        ori_data = np.concatenate(ori_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_data = np.concatenate(recon_data, axis=0).reshape(-1, dataset.data.shape[-1])
        labels = np.concatenate(labels, axis=0).reshape(-1)
        losses = np.concatenate(losses, axis=0).reshape(-1)
        
        return ori_data, recon_data, labels, losses
    
    def train(self):
        """
        Set up optimizer and training loop
        """
        print("\n\n"+"-" * 20 + "Train" + "-" * 20)
        params_group = [
            {
                "params" : self.model.time_recon.parameters(),
                "lr" : self.args.learning_rate,
                "weight_decay" : 0.01
            },
            {
                "params" : self.model.freq_recon.parameters(),
                "lr" : self.args.learning_rate,
                "weight_decay" : 0.01
            }
        ]
        self.model.train()
        optimizer = optim.AdamW(
            params_group, lr=self.args.learning_rate, amsgrad=True
        )
        training_data, training_loader = self.load_data(split="train")

        self.results = {}
        total_updates = 0
        self.active_codes = None
        for epoch in range(1, self.args.epochs + 1):
            for x, _ in training_loader:
                x = x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = self.train_update(x)
                loss.backward()
                optimizer.step()
                if total_updates % self.args.log_interval == 0:
                    if self.args.save:
                        utils.save_model_and_results(self.model,None,None,self.args.filename,)
                    print(
                        f'Update # {total_updates},Recon Error: {(self.results["recon_errors"]):.4f}, Emb_loss: {(self.results["emb_loss"]):.4f}, time_Perplexity: {(self.results["time_perplexities"]):.0f}, freq_Perplexity: {(self.results["freq_perplexities"]):.0f}'
                    )
                total_updates += 1
                
        print("-" * 20 + "End Train" + "-" * 20)

    def test(self):
        """
        test model 
        """
        print("\n\n"+"-" * 20 + "test".center(20,"-") + "-" * 20)
        self.load_model()
        test_data, test_loader = self.load_data(split="test")
        self.model.eval()
        ori_data, recon_data, labels, losses = self.test_template(
           test_data, test_loader
        )
        
        self.evaluate(losses,labels)
        self.draw_loss(labels, losses, prefix="test")
        print("-" * 20 + "End Test".center(20,"-") + "-" * 20)
             
    def visual_train(self):
        """
        visualize reconstruction results
        """

        print("\n\n"+"-" * 20 + "Visualizing" + "-" * 20)
        os.makedirs("visual/" + self.args.filename, exist_ok=True)
        self.load_model()
        
        # choose channel to visualize
        channels = self.args.visual_channel if self.args.visual_channel is not None else [0]
        
        validation_data, validation_loader = self.load_data(split="val")

        ori_data, recon_data, labels, losses = self.test_template(validation_data, validation_loader) 
        
        # plotting settings
        self.draw_pic(ori_data,recon_data,labels,losses,channels,"train",False)
        
        print("-" * 20 + "End Visual Train" + "-" * 20)
         
    def visual_test(self):
        """
        visualize reconstruction results
        """
        
        print("\n\n"+"-" * 20 + "Visualizing".center(20) + "-" * 20)
        os.makedirs("visual/" + self.args.filename, exist_ok=True)
        self.load_model()
        
        # choose channel to visualize
        channels = self.args.visual_channel if self.args.visual_channel is not None else [0]
        validation_data, validation_loader = self.load_data(split="test")

        ori_data, recon_data, labels, losses = self.test_template(validation_data, validation_loader)

        self.evaluate(losses,labels)
        # logger.info(f"[Number of Anomaly segments]: {len(seg)}")

        self.draw_pic(ori_data, recon_data, labels, losses, channels, "test", True)
        print("-" * 20 + "End Visual Test".center(20) + "-" * 20)
        
    def save_indices(self):
        '''
        save quantized indices
        '''
        if not (os.path.exists(f"data/indices/time_{self.args.filename}.npy") and os.path.exists(f"data/indices/freq_{self.args.filename}.npy")):
            print("\n\n"+"-" * 20 + "Saving Indices".center(20) + "-" * 20)
            os.makedirs("data/indices/", exist_ok=True)
            validation_data, validation_loader = self.load_data(split="val")
            
            self.time_indices = []
            self.freq_indices = []
            for x, _ in validation_loader:
                x = x.to(self.device, non_blocking=True)
                self.save_indices_update(x)
            self.time_indices = torch.cat(self.time_indices).to(dtype=torch.float32)
            self.time_indices = self.time_indices.cpu().numpy()
            np.save(f"data/indices/time_{self.args.filename}.npy", self.time_indices)
            
            self.freq_indices = torch.cat(self.freq_indices).to(dtype=torch.float32)
            self.freq_indices = self.freq_indices.cpu().numpy()
            np.save(f"data/indices/freq_{self.args.filename}.npy", self.freq_indices)
            print("-" * 20 + "End Saving Indices".center(20) + "-" * 20)
              
    def load_indices(self, split="train"):
        if split == "train":
            if not hasattr(self, "indices_train_dataset_and_loader"):
                t_dataset, t_loader = load_data_and_data_loaders(
                    file=f"time_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.time_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                f_dataset, f_loader = load_data_and_data_loaders(
                    file=f"freq_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.freq_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                self.indices_train_dataset_and_loader = (t_dataset, t_loader, f_dataset, f_loader)
            else:
                t_dataset, t_loader, f_dataset, f_loader = self.indices_train_dataset_and_loader
        elif split == "test":
            if not hasattr(self, "indices_test_dataset_and_loader"):
                t_dataset, t_loader = load_data_and_data_loaders(
                    file=f"time_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.time_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                f_dataset, f_loader = load_data_and_data_loaders(
                    file=f"freq_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.freq_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                self.indices_test_dataset_and_loader = (t_dataset, t_loader, f_dataset, f_loader)
            else:
                t_dataset, t_loader, f_dataset, f_loader = self.indices_test_dataset_and_loader
        elif split == "val":
            if not hasattr(self, "indices_val_dataset_and_loader"):
                t_dataset, t_loader = load_data_and_data_loaders(
                    file=f"time_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.time_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                f_dataset, f_loader = load_data_and_data_loaders(
                    file=f"freq_{self.args.filename}.npy",
                    split=split,
                    window_size=self.args.window_size // self.args.freq_recon.patch_size,
                    step=1,
                    batch_size=128,
                    data_ratio=1.0,
                )
                self.indices_val_dataset_and_loader = (t_dataset, t_loader, f_dataset, f_loader)
            else:
                t_dataset, t_loader, f_dataset, f_loader = self.indices_val_dataset_and_loader
        return t_dataset, t_loader, f_dataset, f_loader

    def build_ngram(self):
        from models.n_gram import TimeNGramDetector, VarNGramDetector
        import time
        self.load_model()
        start = time.time()
        self.save_indices()
        t_dataset, t_loader, f_dataset, f_loader = self.load_indices(split="train")
        train_codes = t_dataset.data
        if self.shuffle_channel_flag:
            train_codes, _ = self.shuffle_channel(train_codes, seed=self.seed)

        train_codes = einops.rearrange(train_codes, 't c -> 1 c t')
        print(f"提取训练集 codebook 索引耗时: {time.time() - start:.2f} 秒")
        start = time.time()
        self.time_ngram = TimeNGramDetector(
            channels=train_codes.shape[1],
            order=3,
            alpha=0.1,
        ).fit(train_codes)
        print(f"训练通道级 n-gram 模型耗时: {time.time() - start:.2f} 秒")
        start = time.time()
        self.var_ngram = VarNGramDetector(
            order=2,
            alpha=0.1,
        ).fit(train_codes)
        print(f"训练联合状态 n-gram 模型耗时: {time.time() - start:.2f} 秒")
        pass

    def test_channel_update(self, dataset, test_loader):
        ori_data = []
        recon_data = []
        labels = []
        recon_losses = []
        time_ngram_scores = []
        var_ngram_scores = []
        with torch.inference_mode():
            for x, label in test_loader:
                b, t, c = x.shape
                x = x.to(self.device, non_blocking=True)
                # VQ-VAE 前向
                _, x_hat, _, _, time_indices, _ = self.model(x, indices=True)
                # time_indices shape: ((b*c), latent_t, k)
                codes = time_indices[..., 0]  # 只取第一个 codebook
                codes = einops.rearrange(codes, "(b c) t -> b c t", b=b, c=c)
                codes = codes.detach().cpu().numpy().astype(np.int32, copy=False)
                if self.shuffle_channel_flag:
                    codes, indices = self.shuffle_channel(codes, seed=self.seed)
                ch_scores = self.time_ngram.score(codes, aggregate=self.args.aggregate, topk=self.args.topk)
                joint_scores = self.var_ngram.score(codes, aggregate=self.args.aggregate, topk=self.args.topk)
                recon_loss = torch.mean((x_hat - x) ** 2, dim=-1).detach().cpu().numpy()
                def upsample_latent_scores(scores, target_length):
                    """
                    scores:
                        [b, latent_t] or [b, c, latent_t]
                    return:
                        [b, target_length] or [b, c, target_length]
                    """
                    latent_time = scores.shape[-1]
                    repeat_factor = max(1, int(np.ceil(target_length / latent_time)))
                    scores = np.repeat(scores, repeat_factor, axis=-1)
                    return scores[..., :target_length]
                ch_scores = upsample_latent_scores(ch_scores, t)
                joint_scores = upsample_latent_scores(joint_scores, t)

                ori_data.append(x.cpu().detach().numpy())
                recon_data.append(x_hat.cpu().detach().numpy())
                labels.append(label.detach().cpu().numpy())
                recon_losses.append(recon_loss)
                time_ngram_scores.append(ch_scores)
                var_ngram_scores.append(joint_scores)

        labels = np.concatenate(labels, axis=0).reshape(-1)
        ori_data = np.concatenate(ori_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_data = np.concatenate(recon_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_losses = np.concatenate(recon_losses, axis=0).reshape(-1)
        time_ngram_scores = np.concatenate(time_ngram_scores, axis=0).reshape(-1)
        var_ngram_scores = np.concatenate(var_ngram_scores, axis=0).reshape(-1)

        return ori_data, recon_data, recon_losses, time_ngram_scores, var_ngram_scores, labels

    def test_channel(self):
        
        print("\n\n" + "-" * 20 + "Channel Test".center(25) + "-" * 20)
        self.load_model()
        self.shuffle_channel_flag = False
        self.seed = 41
        self.build_ngram()
        # -------------------------------
        # 加载测试数据
        # -------------------------------
        dataset, test_loader = self.load_data(split="test")
        self.model.eval()

        def min_max_normalize(values, eps=1e-8):
            v_min = values.min()
            v_max = values.max()
            return (values - v_min) / (v_max - v_min + eps)
        
        ori_data, recon_data, recon_losses, time_ngram_scores, var_ngram_scores, labels = self.test_channel_update(dataset, test_loader)
        recon_losses = min_max_normalize(recon_losses)
        time_ngram_scores = min_max_normalize(time_ngram_scores)
        var_ngram_scores = min_max_normalize(var_ngram_scores)
        
        time_loss = recon_losses + self.args.alpha * time_ngram_scores
        var_loss = recon_losses + self.args.beta * var_ngram_scores
        losses = recon_losses + self.args.alpha * time_ngram_scores + self.args.beta * var_ngram_scores

        print("@ var score")
        self.evaluate(var_loss,labels)
        self.draw_loss(labels, var_loss, prefix="channel_var_time")
        print("@ time score")
        self.evaluate(time_loss,labels)
        self.draw_loss(labels, time_loss, prefix="channel_time_time")
        print("@ recon loss + time score + time_time score")
        self.evaluate(losses,labels)
        self.draw_loss(labels, losses, prefix="channel_test")
         
    def draw_pic(self, ori_data, recon_data, labels, loss, channels, prefix, draw_seg=False):
        os.makedirs(f"visual/{self.args.filename}", exist_ok=True)
        draw_length = min(len(ori_data), 10000)
        seg = []
        if draw_seg:
            seg = get_anomaly_segments(labels, anomaly_val=1)
        for i in channels:
            fig, ax = plt.subplots(figsize=(max(200, len(ori_data) // 200), 10))
            ax.plot(ori_data[:, i], label="Original")
            ax.plot(recon_data[:, i], label="Reconstruction")
            if draw_seg and len(seg) > 0:
                for s, e in seg[:draw_length]:
                    ax.axvspan(s, e, color="red", alpha=0.3, linewidth=0)
            mean = np.mean(ori_data[:draw_length, i])
            std = np.std(ori_data[:draw_length, i])
            std = np.where(std == 0, 1, std)
            h_bound = mean + std * 5
            l_bound = mean - std * 5
            ax.set_ylim(l_bound, h_bound)
            # ax.set_xticks(np.arange(0, len(ori_data), step=50))
            ax.set_title(f"Channel {i} Overview")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(f"visual/{self.args.filename}/{prefix}_{i}_overview.png")
            plt.close(fig)

    def draw_loss(self, labels, loss, prefix):
        os.makedirs(f"visual/{self.args.filename}", exist_ok=True)
        plt.figure(figsize=(max(50, len(labels) // 100), 5))
        loss = np.clip(loss, 0, max(np.percentile(loss, 50),10))
        plt.plot(loss, label="Reconstruction Loss")
        index = labels == 1
        x = np.arange(len(labels))
        x = x[index]
        y = (labels*loss)[index]
        plt.scatter(x, y, label="Anomaly Label", marker="x", color="red")
        plt.legend(loc="upper right")
        plt.title("Loss and Anomaly Label")
        plt.xlabel("Time Steps")
        plt.savefig(f"visual/{self.args.filename}/{prefix}_loss.png")
        plt.close()
    
    def draw_indices(self):
        print("\n\n" + "-" * 20 + "Draw indices".center(25) + "-" * 20)
        self.load_model()
        num = self.model.time_recon.n_embeddings
        shutil.rmtree("visual/indices", ignore_errors=True)
        os.makedirs("visual/indices", exist_ok=True)
        draw_indice = 0
        num_of_row = 10
        draw_seq = []
        indices = []
        for i, mask in enumerate(range(num)):
            draw_indice += 1
            indice = torch.tensor([i],device=self.device)
            x = einops.rearrange(indice, 't -> 1 t')
            seq = self.model.time_recon.decode_indices(x)
            seq = seq.flatten().cpu().detach().numpy()
            draw_seq.append(seq)
            indices.append(i)
        row = (draw_indice-1) // num_of_row + 1
        col = num_of_row if draw_indice >= num_of_row else draw_indice
        fig, axes = plt.subplots(row, col, figsize=(3 * col, 5 * row), sharey=True)
        for i, (ax_r, indice) in enumerate(zip(axes, indices)):
            if row == 1:
                ax = ax_r
                ax.plot(draw_seq[i],linewidth=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Codebook {indice}")
                ax.grid(True, linestyle="--", alpha=0.5)
            else:
                for j, ax in enumerate(ax_r):
                    if i*num_of_row+j >= draw_indice:
                        ax.axis("off")
                        continue
                    ax.plot(draw_seq[i*num_of_row+j],linewidth=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # ax.set_title(f"Codebook {indice}")
                    ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"visual/indices/time_total.png",dpi=500)
        plt.close()   


        # freq 
        num = self.model.freq_recon.n_embeddings
        draw_indice = 0
        draw_seq = []
        indices = []
        for i, mask in enumerate(range(num)):
            draw_indice += 1
            indice = torch.tensor([i],device=self.device)
            x = einops.rearrange(indice, 't -> 1 t')
            seq = self.model.freq_recon.decode_indices(x)
            seq = seq.flatten().cpu().detach().numpy()
            draw_seq.append(seq)
            indices.append(i)
        row = (draw_indice-1) // num_of_row + 1
        col = num_of_row if draw_indice >= num_of_row else draw_indice
        fig, axes = plt.subplots(row, col, figsize=(3 * col, 5 * row), sharey=True)
        for i, (ax_r, indice) in enumerate(zip(axes, indices)):
            if row == 1:
                ax = ax_r
                ax.plot(draw_seq[i],linewidth=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Codebook {indice}")
                ax.grid(True, linestyle="--", alpha=0.5)
            else:
                for j, ax in enumerate(ax_r):
                    if i*num_of_row+j >= draw_indice:
                        ax.axis("off")
                        continue
                    ax.plot(draw_seq[i*num_of_row+j],linewidth=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # ax.set_title(f"Codebook {indice}")
                    ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"visual/indices/freq_total.png",dpi=500)
        plt.close()  
        print("\n\n" + "-" * 20 + "End draw indices".center(25) + "-" * 20)

    def get_channel_anomaly_score(self, dataset, test_loader):
        self.load_model()
        self.build_ngram()
        ori_data = []
        recon_data = []
        labels = []
        recon_losses = []
        channel_ngram_scores = []
        joint_ngram_scores = []
        with torch.inference_mode():
            for x, label in test_loader:
                b, t, c = x.shape
                x = x.to(self.device, non_blocking=True)
                # VQ-VAE 前向
                _, x_hat, _, _, time_indices, _ = self.model(x, indices=True)
                # time_indices shape: ((b*c), latent_t, k)
                codes = time_indices[..., 0]  # 只取第一个 codebook
                codes = einops.rearrange(codes, "(b c) t -> b c t", b=b, c=c)
                codes = codes.detach().cpu().numpy().astype(np.int32, copy=False)
                ch_scores = self.time_ngram.score(codes, aggregate="none", topk=self.args.topk)
                joint_scores = self.var_ngram.score(codes, aggregate="none", topk=self.args.topk)
                recon_loss = ((x_hat - x) ** 2).detach().cpu().numpy()
                def upsample_latent_scores(scores, target_length):
                    """
                    scores:
                        [b, latent_t] or [b, c, latent_t]
                    return:
                        [b, target_length] or [b, c, target_length]
                    """
                    latent_time = scores.shape[-1]
                    repeat_factor = max(1, int(np.ceil(target_length / latent_time)))
                    scores = np.repeat(scores, repeat_factor, axis=-1)
                    return scores[..., :target_length]
                ch_scores = upsample_latent_scores(ch_scores, t)
                ch_scores = einops.rearrange(ch_scores, "b c t -> (b t) c")
                joint_scores = upsample_latent_scores(joint_scores, t)
                joint_scores = einops.rearrange(joint_scores, "b c t -> (b t) c")

                ori_data.append(x.cpu().detach().numpy())
                recon_data.append(x_hat.cpu().detach().numpy())
                labels.append(label.detach().cpu().numpy())
                recon_losses.append(recon_loss)
                channel_ngram_scores.append(ch_scores)
                joint_ngram_scores.append(joint_scores)
                
        labels = np.concatenate(labels, axis=0).reshape(-1)
        ori_data = np.concatenate(ori_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_data = np.concatenate(recon_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_losses = np.concatenate(recon_losses, axis=0).reshape(-1, dataset.data.shape[-1])
        channel_ngram_scores = np.concatenate(channel_ngram_scores, axis=0).reshape(-1, dataset.data.shape[-1])
        joint_ngram_scores = np.concatenate(joint_ngram_scores, axis=0).reshape(-1, dataset.data.shape[-1])

        return ori_data, recon_data, recon_losses, channel_ngram_scores, joint_ngram_scores, labels

    def evaluate(self, pred, labels):
        utils.evaluate(pred, labels, self.args.window_size)
        
    def shuffle_channel(self, data, seed=None):
        """
        Randomly shuffle the channel order of data.

        Args:
            data: shape [time_length, num_channels]
            seed: Optional random seed for reproducibility.

        Returns:
            shuffled_data: data with shuffled channel order
            shuffled_indices: channel permutation order
        """
        num_channels = data.shape[1]

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(num_channels)

        shuffled_data = data[:, shuffled_indices]

        print(f"shuffle channel order: {shuffled_indices}")
        print(f"shuffled data shape {shuffled_data.shape}")

        return shuffled_data, shuffled_indices
    
    def inverse_shuffle_channel(self, data, shuffled_indices):
        """
        Restore shuffled channel order back to original order.

        Args:
            data: shuffled data, shape [time_length, num_channels]
            shuffled_indices: permutation order used in shuffle_channel()

        Returns:
            restored_data: data restored to original channel order
        """
        inverse_indices = np.argsort(shuffled_indices)

        restored_data = data[:, inverse_indices]

        print(f"inverse channel order: {inverse_indices}")
        print(f"restored data shape {restored_data.shape}")

        return restored_data