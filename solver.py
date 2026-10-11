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
from utils import build_memory_index, evaluate, get_anomaly_segments, write_metric
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
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
        try:
            if not hasattr(self, "loaded"):
                self.model.load_state_dict(
                    torch.load(f"./results/{self.args.filename}_seed{self.args.seed}.pth", weights_only=False)["model"]
                )
                self.model.eval()
                print("@ loaded VQ-VAE model from ./results/" + self.args.filename + ".pth")
                print("@ set model to eval mode")
                self.loaded = True
                return True
            else:
                print("@ model already loaded")
                return False
        except:
            return False
        
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
        for epoch in tqdm(range(1, self.args.epochs + 1), desc="Training Epochs"):
            for x, _ in tqdm(training_loader, desc=f"Epoch {epoch}", leave=False):
                x = x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = self.train_update(x)
                loss.backward()
                optimizer.step()
                total_updates += 1
            tqdm.write(f'Update # {total_updates},Recon Error: {(self.results["recon_errors"]):.4f}, Emb_loss: {(self.results["emb_loss"]):.4f}, time_Perplexity: {(self.results["time_perplexities"]):.0f}, freq_Perplexity: {(self.results["freq_perplexities"]):.0f}')
        if self.args.save:
            utils.save_model_and_results(
                self.model,
                self.results,
                vars(self.args).copy(),
                f"{self.args.filename}_seed{self.args.seed}",
                epoch=self.args.epochs,
                global_step=total_updates,
            )
        print("-" * 20 + "End Train" + "-" * 20)
             
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

        # self.evaluate(losses,labels)
        # logger.info(f"[Number of Anomaly segments]: {len(seg)}")

        self.draw_pic(ori_data, recon_data, labels, losses, channels, "test", True)
        print("-" * 20 + "End Visual Test".center(20) + "-" * 20)
        
    def save_indices(self):
        '''
        save quantized indices
        '''
        if getattr(self, "_indices_saved_for_current_solver", False):
            return
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
        self._indices_saved_for_current_solver = True
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
        """Fit the temporal scorer and the variable n-gram scorer on normal codes."""
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
            order=self.args.time_ngram_order,
            alpha=0.1,
        ).fit(train_codes)
        print(f"训练通道级 n-gram 模型耗时: {time.time() - start:.2f} 秒")
        start = time.time()
        self.var_ngram = VarNGramDetector(
            order=self.args.var_ngram_order,
            alpha=0.1,
        ).fit(train_codes)
        print(f"训练联合状态 n-gram 模型耗时: {time.time() - start:.2f} 秒")
        pass

    def collect_time_codes(self, split="train"):
        """Extract first-codebook states from a normal split without training."""
        dataset, loader = self.load_data(split=split)
        collected = []
        self.model.eval()
        with torch.inference_mode():
            for x, _ in loader:
                b, _, c = x.shape
                x = x.to(self.device, non_blocking=True)
                _, _, _, _, time_indices, _ = self.model(x, indices=True)
                codes = einops.rearrange(
                    time_indices[..., 0], "(b c) t -> b c t", b=b, c=c
                )
                collected.append(codes.detach().cpu().numpy().astype(np.int32, copy=False))
        if not collected:
            raise RuntimeError(f"No prototype states were extracted from {split} split")
        return np.concatenate(collected, axis=0)

    def build_chow_liu_vcpm(self):
        """Fit a post-hoc Chow--Liu VCPM only on normal training states."""
        from models.n_gram import ChowLiuVCPMDetector

        self.load_model()
        train_codes = self.collect_time_codes(split="train")
        self.chow_liu_vcpm = ChowLiuVCPMDetector(
            n_states=int(self.args.time_recon.n_embeddings),
            state_modulo=int(self.args.time_recon.n_embeddings),
            gamma=getattr(self.args, "vcpm_gamma", 0.1),
            tree_weight=getattr(self.args, "vcpm_tree_weight", "nmi"),
            epsilon=getattr(self.args, "vcpm_epsilon", 1e-6),
        ).fit(train_codes)
        print(f"Chow--Liu VCPM: fitted {len(self.chow_liu_vcpm.edges)} dependency edges")

    def test_channel_update(
        self, dataset, test_loader, continuous_temporal=False, variable_detector=None
    ):
        """Score a split and optionally evaluate temporal tokens as one causal stream.

        The validation/test loaders are chronological and use non-overlapping raw
        windows.  Concatenate their latent tokens in that order so temporal AR or
        n-gram context crosses a window boundary instead of backing off at every
        window start.  Variable scores and reconstruction remain defined per
        window, as in the original DPCM implementation.
        """
        ori_data = []
        recon_data = []
        labels = []
        recon_losses = []
        var_ngram_scores = []
        time_code_windows = []
        variable_detector = variable_detector or self.var_ngram
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
                time_code_windows.append(codes)
                joint_scores = variable_detector.score(codes, aggregate=self.args.aggregate, topk=self.args.topk)
                recon_loss = torch.mean((x_hat - x) ** 2, dim=-1).detach().cpu().numpy()
                latent_time = joint_scores.shape[-1]
                repeat_factor = max(1, int(np.ceil(t / latent_time)))
                joint_scores = np.repeat(joint_scores, repeat_factor, axis=-1)[..., :t]

                ori_data.append(x.cpu().detach().numpy())
                recon_data.append(x_hat.cpu().detach().numpy())
                labels.append(label.detach().cpu().numpy())
                recon_losses.append(recon_loss)
                var_ngram_scores.append(joint_scores)

        # ``[window, channel, latent] -> [1, channel, global_latent_time]``.
        # This preserves the chronological window order from a non-shuffled
        # loader while allowing a causal AR context to survive window boundaries.
        time_code_windows = np.concatenate(time_code_windows, axis=0)
        windows, channels, latent_time = time_code_windows.shape
        if continuous_temporal:
            continuous_codes = time_code_windows.transpose(1, 0, 2).reshape(
                1, channels, windows * latent_time
            )
            temporal_scores = self.time_ngram.score(
                continuous_codes, aggregate=self.args.aggregate, topk=self.args.topk
            ).reshape(windows, latent_time)
        else:
            temporal_scores = self.time_ngram.score(
                time_code_windows, aggregate=self.args.aggregate, topk=self.args.topk
            )
        repeat_factor = max(1, int(np.ceil(t / latent_time)))
        time_ngram_scores = np.repeat(temporal_scores, repeat_factor, axis=-1)[..., :t]

        labels = np.concatenate(labels, axis=0).reshape(-1)
        ori_data = np.concatenate(ori_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_data = np.concatenate(recon_data, axis=0).reshape(-1, dataset.data.shape[-1])
        recon_losses = np.concatenate(recon_losses, axis=0).reshape(-1)
        time_ngram_scores = time_ngram_scores.reshape(-1)
        var_ngram_scores = np.concatenate(var_ngram_scores, axis=0).reshape(-1)

        return ori_data, recon_data, recon_losses, time_ngram_scores, var_ngram_scores, labels

    def fit_channel_score_calibration(
        self, calibration_dataset, calibration_loader, continuous_temporal=False,
        variable_detector=None,
    ):
        """Fit legacy and empirical upper-tail calibrators on normal data only."""
        _, _, recon_losses, time_scores, variable_scores, _ = self.test_channel_update(
            calibration_dataset, calibration_loader,
            continuous_temporal=continuous_temporal,
            variable_detector=variable_detector,
        )
        def bounds(values):
            return float(values.min()), float(values.max())

        return {
            "legacy": {
                "reconstruction": bounds(recon_losses),
                "temporal": bounds(time_scores),
                "variable": bounds(variable_scores),
            },
            # Keep the complete normal reference distribution for ECDF upper
            # tails.  Unlike min-max, this does not make a modality's fusion
            # weight depend on one extreme calibration point.
            "ecdf": {
                "reconstruction": np.sort(recon_losses.astype(np.float64, copy=False)),
                "temporal": np.sort(time_scores.astype(np.float64, copy=False)),
                "variable": np.sort(variable_scores.astype(np.float64, copy=False)),
            },
        }

    def apply_channel_score_calibration(self, scores, calibration_bounds):
        """Apply legacy upper-tail calibration to scores."""
        lower, upper = calibration_bounds
        calibrated = (scores - lower) / (upper - lower + 1e-5)
        return np.clip(calibrated, 0.0, 1.0)

    @staticmethod
    def ecdf_upper_tail_evidence(scores, sorted_reference):
        """Map high-is-anomalous scores to -log empirical upper-tail mass.

        p(s) = (1 + #{c in calibration: c >= s}) / (n + 1).  The finite
        sample correction makes the largest possible evidence -log(1/(n+1))
        and exposes, rather than hides, tail saturation.
        """
        values = np.asarray(scores, dtype=np.float64)
        reference = np.asarray(sorted_reference, dtype=np.float64)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError("ECDF reference must be a non-empty sorted vector")
        rank_below = np.searchsorted(reference, values, side="left")
        p_upper = (reference.size - rank_below + 1.0) / (reference.size + 1.0)
        evidence = -np.log(p_upper)
        diagnostics = {
            "calibration_size": int(reference.size),
            "saturation_fraction": float(np.mean(values > reference[-1])),
            "maximum_evidence": float(-np.log(1.0 / (reference.size + 1.0))),
        }
        return evidence.astype(np.float32, copy=False), diagnostics
    
    def get_channel_score_components(self, variable_detector=None):
        """Return calibrated score components and test labels without evaluation side effects."""
        self.load_model()
        self.build_ngram()
        # Apply the same continuous token sequence convention to calibration
        # and testing, so temporal n-gram contexts do not reset at boundaries.
        continuous_temporal = True
        calibration_dataset, calibration_loader = self.load_data(split="val")
        test_dataset, test_loader = self.load_data(split="test")
        score_calibration = self.fit_channel_score_calibration(
            calibration_dataset, calibration_loader,
            continuous_temporal=continuous_temporal,
            variable_detector=variable_detector,
        )
        _, _, reconstruction, temporal, variable, labels = self.test_channel_update(
            test_dataset, test_loader,
            continuous_temporal=continuous_temporal,
            variable_detector=variable_detector,
        )
        tail_reconstruction, reconstruction_tail_info = self.ecdf_upper_tail_evidence(
            reconstruction, score_calibration["ecdf"]["reconstruction"]
        )
        tail_temporal, temporal_tail_info = self.ecdf_upper_tail_evidence(
            temporal, score_calibration["ecdf"]["temporal"]
        )
        tail_variable, variable_tail_info = self.ecdf_upper_tail_evidence(
            variable, score_calibration["ecdf"]["variable"]
        )
        return {
            "raw_reconstruction": reconstruction,
            "raw_temporal": temporal,
            "raw_variable": variable,
            "reconstruction": self.apply_channel_score_calibration(
                reconstruction, score_calibration["legacy"]["reconstruction"]
            ),
            "temporal": self.apply_channel_score_calibration(
                temporal, score_calibration["legacy"]["temporal"]
            ),
            "variable": (
                self.apply_channel_score_calibration(
                    variable, score_calibration["legacy"]["variable"]
                )
            ),
            "labels": labels,
            "calibration": score_calibration,
            "tail_evidence": {
                "reconstruction": tail_reconstruction,
                "temporal": tail_temporal,
                "variable": tail_variable,
            },
            "tail_diagnostics": {
                "reconstruction": reconstruction_tail_info,
                "temporal": temporal_tail_info,
                "variable": variable_tail_info,
            },
        }
    
    def report_origin_vcpm(self, prefix, minmax_vcpm, ecdf_vcpm, ablation):
        components = self.get_channel_score_components()
        recon_losses = components["reconstruction"]
        time_ngram_scores = components["temporal"]
        var_ngram_scores = components["variable"]
        labels = components["labels"]

        if minmax_vcpm:
            time_loss = recon_losses + time_ngram_scores
            var_loss = recon_losses + var_ngram_scores
            losses = recon_losses + time_ngram_scores + var_ngram_scores
            result_file = f"{self.args.filename}.csv"
            result_file = os.path.join(prefix,result_file)
            with open(result_file, "a+") as f:
                if ablation:
                    write_metric(f,"dpr", self.args.seed, self.evaluate(recon_losses,labels), write_head=True)
                    write_metric(f,"dpr+tstm", self.args.seed, self.evaluate(time_loss,labels))
                    write_metric(f,"dpr+vcpm", self.args.seed, self.evaluate(var_loss,labels))
                    write_metric(f,"dpcm", self.args.seed, self.evaluate(losses,labels))
                else:
                    write_metric(f,"dpcm", self.args.seed, self.evaluate(losses,labels), write_head=True)
                f.write("\n")

        if ecdf_vcpm:
            tail_evidence = components["tail_evidence"]
            # Equal-weight evidence fusion.  These are already comparable
            # normal-tail evidences, so no subsequent min-max normalization occurs.
            ecdf_time_loss = (tail_evidence["reconstruction"] + tail_evidence["temporal"]) / 2.0
            ecdf_var_loss = (tail_evidence["reconstruction"] + tail_evidence["variable"]) / 2.0
            ecdf_losses = (tail_evidence["reconstruction"] + tail_evidence["temporal"] + tail_evidence["variable"]) / 3.0
            ecdf_result_file = f"{self.args.filename}_ecdf_fusion.csv"
            ecdf_result_file = os.path.join(prefix,ecdf_result_file)
            with open(ecdf_result_file, "a+") as f:
                if ablation:
                    write_metric(f, "dpr", self.args.seed, self.evaluate(recon_losses, labels), write_head=True)
                    write_metric(f, "dpr+tstm", self.args.seed, self.evaluate(ecdf_time_loss, labels))
                    write_metric(f, "dpr+vcpm", self.args.seed, self.evaluate(ecdf_var_loss, labels))
                    write_metric(f, "dpcm", self.args.seed, self.evaluate(ecdf_losses, labels))
                else:
                    write_metric(f, "dpr", self.args.seed, self.evaluate(ecdf_losses, labels), write_head=True)
                f.write("\n")   

    def report_chowliu_vcpm(self, prefix, minmax_tree, ecdf_tree, ablation):
        self.build_chow_liu_vcpm()
        tree_components = self.get_channel_score_components(variable_detector=self.chow_liu_vcpm)
        recon_losses = tree_components["reconstruction"]
        labels = tree_components["labels"]

        if minmax_tree:
            tree_var_scores = tree_components["variable"]
            tree_time_scores = tree_components["temporal"]
            tree_time_loss = recon_losses + tree_time_scores
            tree_var_loss = recon_losses + tree_var_scores
            tree_losses = recon_losses +  tree_time_scores + tree_var_scores
            chowliu_result_file =  f"{self.args.filename}_chowliu.csv"
            chowliu_result_file = os.path.join(prefix,chowliu_result_file)
            with open(chowliu_result_file, "a+") as f:
                if ablation:
                    write_metric(f,"dpr", self.args.seed, self.evaluate(recon_losses,labels), write_head=True)
                    write_metric(f,"dpr+tstm", self.args.seed, self.evaluate(tree_time_loss,labels))
                    write_metric(f,"dpr+vcpm", self.args.seed, self.evaluate(tree_var_loss,labels))
                    write_metric(f,"dpcm", self.args.seed, self.evaluate(tree_losses,labels))
                else:
                    write_metric(f,"dpcm", self.args.seed, self.evaluate(tree_losses,labels), write_head=True)
                f.write("\n")

        if ecdf_tree:
            tail_evidence = tree_components["tail_evidence"]
            # Equal-weight evidence fusion.  These are already comparable
            # normal-tail evidences, so no subsequent min-max normalization occurs.
            ecdf_time_loss = (tail_evidence["reconstruction"] + tail_evidence["temporal"]) / 2.0
            ecdf_var_loss = (tail_evidence["reconstruction"] + tail_evidence["variable"]) / 2.0
            ecdf_losses = (tail_evidence["reconstruction"] + tail_evidence["temporal"] + tail_evidence["variable"]) / 3.0
            ecdf_result_file = f"{self.args.filename}_chowliu_ecdf_fusion.csv"
            ecdf_result_file = os.path.join(prefix, ecdf_result_file)
            with open(ecdf_result_file, "a+") as f:
                if ablation:
                    write_metric(f, "dpr", self.args.seed, self.evaluate(recon_losses, labels), write_head=True)
                    write_metric(f, "dpr+tstm", self.args.seed, self.evaluate(ecdf_time_loss, labels))
                    write_metric(f, "dpr+vcpm", self.args.seed, self.evaluate(ecdf_var_loss, labels))
                    write_metric(f, "dpcm", self.args.seed, self.evaluate(ecdf_losses, labels))
                else:
                    write_metric(f, "dpcm", self.args.seed, self.evaluate(ecdf_losses, labels), write_head=True)
                f.write("\n")   
        
    def test(self):
        print("\n\n" + "-" * 20 + "Channel Test".center(25) + "-" * 20)
        self.shuffle_channel_flag = False
        self.model.eval()
        reports = self.args.report_variant.split(" ")
        minmax_vcpm = "minmax_vcpm" in reports
        ecdf_vcpm = "ecdf_vcpm" in reports
        minmax_tree = "minmax_tree" in reports
        ecdf_tree = "ecdf_tree" in reports
        prefix = "report"
        ablation = self.args.ablation
        os.makedirs(prefix, exist_ok=True)
        self.report_origin_vcpm(prefix, minmax_vcpm, ecdf_vcpm, ablation)
        self.report_chowliu_vcpm(prefix, minmax_tree, ecdf_tree, ablation)


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
        return evaluate(
            pred,
            labels,
            self.args.window_size,
            getattr(self.args, "metrics", "auc"),
        )
        
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
