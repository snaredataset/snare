import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.single_cls import SingleClassifier


class ZeroShotClassifier(SingleClassifier):

    def __init__(self, cfg, train_ds, val_ds):
        super().__init__(cfg, train_ds, val_ds)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_model(self):
        pass

    def configure_optimizers(self):
        pass

    def forward(self, batch):
        (img1_n_feats, img2_n_feats), lang_feats, ans, (key1, key2), annotation, is_visual = batch

        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        # normalize
        img1_n_feats = img1_n_feats / img1_n_feats.norm(dim=-1, keepdim=True)
        img2_n_feats = img2_n_feats / img2_n_feats.norm(dim=-1, keepdim=True)
        lang_feats = lang_feats / lang_feats.norm(dim=-1, keepdim=True)

        # aggregate
        img1_feats = self.aggregator(img1_n_feats)
        img2_feats = self.aggregator(img2_n_feats)

        bs = img1_feats.shape[0]
        probs = []
        for b in range(bs):
            im = torch.stack([img1_feats[b], img2_feats[b]], dim=0)
            lang = torch.stack([lang_feats[b], lang_feats[b]], dim=0)

            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * im @ lang.t()

            prob = logits_per_image[:,0].softmax(-1)
            probs.append(prob)

        # cat probs
        probs = torch.stack(probs, dim=0)

        # num steps taken (8 for all views)
        bs = lang_feats.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_feats.device)
        num_steps = num_steps * (self.num_views if self.aggregator_type in ['maxpool', 'mean', 'gru'] else 1)

        test_mode = (ans[0] == -1)
        if not test_mode:
            # one-hot labels of answers
            labels = F.one_hot(ans)

            return {
                'probs': probs,
                'labels': labels,
                'is_visual': is_visual,
                'num_steps': num_steps,
            }
        else:
            return {
                'probs': probs,
                'num_steps': num_steps,
            }

    def training_step(self, batch, batch_idx):
        # nothing to train
        pass

    def validation_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(8):
            out = self.forward(batch)
            losses = self._criterion(out)

            loss = losses['loss']
            probs = out['probs']
            labels = out['labels']
            visual = out['is_visual']
            num_steps = out['num_steps']

            metrics = self.compute_metrics(labels, loss, probs, visual, num_steps)
            all_view_results[view] = metrics

        mean_val_loss = np.mean([m['val_loss'].detach().cpu().float() for m in all_view_results.values()])
        mean_val_acc = np.mean([m['val_acc'] for m in all_view_results.values()])

        return dict(
            val_loss=mean_val_loss,
            val_acc=mean_val_acc,
            all_view_results=all_view_results,
        )