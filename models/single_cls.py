import numpy as np
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import wandb

import models.aggregator as agg


class SingleClassifier(LightningModule):

    def __init__(self, cfg, train_ds, val_ds):
        super().__init__()

        self.cfg = cfg
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.dropout = self.cfg['train']['dropout']

        # input dimensions
        self.feats_backbone = self.cfg['train']['feats_backbone']
        self.img_feat_dim = 512
        self.lang_feat_dim = 512
        self.num_views = 8

        # choose aggregation method
        agg_cfg = dict(self.cfg['train']['aggregator'])
        agg_cfg['input_dim'] = self.img_feat_dim
        self.aggregator_type = self.cfg['train']['aggregator']['type']
        self.aggregator = agg.names[self.aggregator_type](agg_cfg)

        # build network
        self.build_model()

        # val progress
        self.best_val_acc = -1.0
        self.best_val_res = None

        # test progress
        self.best_test_acc = -1.0
        self.best_test_res = None

        # results save path
        self.save_path = Path(os.getcwd())

        # log with wandb
        self.log_data = self.cfg['train']['log']
        if self.log_data:
            self.run = wandb.init(
                project=self._cfg['wandb']['logger']['project'],
                config=self._cfg['train'],
                settings=wandb.Settings(show_emoji=False),
                reinit=True
            )
            wandb.run.name = self._cfg['wandb']['logger']['run_name']

    def build_model(self):
        # image encoder
        self.img_fc = nn.Sequential(
            nn.Identity()
        )

        # language encoder
        self.lang_fc = nn.Sequential(
            nn.Identity()
        )

        # finetuning layers for classification
        self.cls_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg['train']['lr'])

    def smoothed_cross_entropy(self, pred, target, alpha=0.1):
        # From ShapeGlot (Achlioptas et. al)
        # https://github.com/optas/shapeglot/blob/master/shapeglot/models/neural_utils.py
        n_class = pred.size(1)
        one_hot = target
        one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)

    def _criterion(self, out):
        probs = out['probs']
        labels = out['labels']

        loss = self.smoothed_cross_entropy(probs, labels)

        return {
            'loss': loss
        }

    def forward(self, batch):
        (img1_n_feats, img2_n_feats), lang_feats, ans, (key1, key2), annotation, is_visual = batch

        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        # aggregate
        img1_feats = self.aggregator(img1_n_feats)
        img2_feats = self.aggregator(img2_n_feats)

        # lang encoding
        lang_enc = self.lang_fc(lang_feats)

        # normalize
        if self.cfg['train']['normalize_feats']:
            img1_feats = img1_feats / img1_feats.norm(dim=-1, keepdim=True)
            img2_feats = img2_feats / img2_feats.norm(dim=-1, keepdim=True)
            lang_enc = lang_enc / lang_enc.norm(dim=-1, keepdim=True)

        # img1 prob
        img1_enc = self.img_fc(img1_feats)
        img1_prob = self.cls_fc(torch.cat([img1_enc, lang_enc], dim=-1))

        # img2 prob
        img2_enc = self.img_fc(img2_feats)
        img2_prob = self.cls_fc(torch.cat([img2_enc, lang_enc], dim=-1))

        # cat probs
        probs = torch.cat([img1_prob, img2_prob], dim=-1)

        # num steps taken (8 for all views)
        bs = lang_enc.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_enc.device)
        if self.aggregator_type in ['maxpool', 'mean', 'gru']:
            num_steps = num_steps * 8
        elif self.aggregator_type in ['two_random_index']:
            num_steps = num_steps * 2

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
        out = self.forward(batch)

        # classifier loss
        losses = self._criterion(out)

        if self.log_data:
            wandb.log({
                'tr/loss': losses['loss'],
            })

        return dict(
            loss=losses['loss']
        )

    def check_correct(self, b, labels, probs):
        right_prob = probs[b][labels[b].argmax()]
        wrong_prob = probs[b][labels[b].argmin()]
        correct = right_prob > wrong_prob
        return correct

    def validation_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch)
            losses = self._criterion(out)

            loss = losses['loss']
            probs = out['probs']
            labels = out['labels']
            visual = out['is_visual']
            num_steps = out['num_steps']

            probs = F.softmax(probs, dim=-1)
            metrics = self.compute_metrics(labels, loss, probs, visual, num_steps)
            all_view_results[view] = metrics

        mean_val_loss = np.mean([m['val_loss'].detach().cpu().float() for m in all_view_results.values()])
        mean_val_acc = np.mean([m['val_acc'] for m in all_view_results.values()])

        return dict(
            val_loss=mean_val_loss,
            val_acc=mean_val_acc,
            all_view_results=all_view_results,
        )

    def compute_metrics(self, labels, loss, probs, visual, num_steps):
        batch_size = probs.shape[0]
        val_total, val_correct, val_pl_correct = 0, 0, 0.
        visual_total, visual_correct, pl_visual_correct = 0, 0, 0.
        nonvis_total, nonvis_correct, pl_nonvis_correct = 0, 0, 0.
        for b in range(batch_size):
            correct = self.check_correct(b, labels, probs)

            if correct:
                val_correct += 1
                val_pl_correct += 1. / num_steps[b]
            val_total += 1

            if bool(visual[b]):
                if correct:
                    visual_correct += 1
                    pl_visual_correct += 1. / num_steps[b]
                visual_total += 1
            else:
                if correct:
                    nonvis_correct += 1
                    pl_nonvis_correct += 1. / num_steps[b]
                nonvis_total += 1

        val_acc = float(val_correct) / val_total
        val_pl_acc = float(val_pl_correct) / val_total
        val_visual_acc = float(visual_correct) / visual_total
        val_pl_visual_acc = float(pl_visual_correct) / visual_total
        val_nonvis_acc = float(nonvis_correct) / nonvis_total
        val_pl_nonvis_acc = float(pl_nonvis_correct) / nonvis_total

        return dict(
            val_loss=loss,
            val_acc=val_acc,
            val_pl_acc=val_pl_acc,
            val_correct=val_correct,
            val_pl_correct=val_pl_correct,
            val_total=val_total,
            val_visual_acc=val_visual_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_visual_correct=visual_correct,
            val_pl_visual_correct=pl_visual_correct,
            val_visual_total=visual_total,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
            val_nonvis_correct=nonvis_correct,
            val_pl_nonvis_correct=pl_nonvis_correct,
            val_nonvis_total=nonvis_total,
        )

    def validation_epoch_end(self, all_outputs, mode='vl'):
        n_view_res = {}
        for view in range(self.num_views):

            view_res = {
                'val_loss': 0.0,

                'val_correct': 0,
                'val_pl_correct': 0,
                'val_total': 0,

                'val_visual_correct': 0,
                'val_pl_visual_correct': 0,
                'val_visual_total': 0,

                'val_nonvis_correct': 0,
                'val_pl_nonvis_correct': 0,
                'val_nonvis_total': 0,
            }

            for output in all_outputs:
                metrics = output['all_view_results'][view]

                view_res['val_loss'] += metrics['val_loss'].item()

                view_res['val_correct'] += metrics['val_correct']
                view_res['val_pl_correct'] += int(metrics['val_pl_correct'])
                view_res['val_total'] += metrics['val_total']

                view_res['val_visual_correct'] += metrics['val_visual_correct']
                view_res['val_pl_visual_correct'] += int(metrics['val_pl_visual_correct'])
                view_res['val_visual_total'] += metrics['val_visual_total']

                view_res['val_nonvis_correct'] += metrics['val_nonvis_correct']
                view_res['val_pl_nonvis_correct'] += int(metrics['val_pl_nonvis_correct'])
                view_res['val_nonvis_total'] += metrics['val_nonvis_total']

            view_res['val_loss'] = float(view_res['val_loss']) / len(all_outputs)

            view_res['val_acc'] = float(view_res['val_correct']) / view_res['val_total']
            view_res['val_pl_acc'] = float(view_res['val_pl_correct']) / view_res['val_total']

            view_res['val_visual_acc'] = float(view_res['val_visual_correct']) / view_res['val_visual_total']
            view_res['val_pl_visual_acc'] = float(view_res['val_pl_visual_correct']) / view_res['val_visual_total']

            view_res['val_nonvis_acc'] = float(view_res['val_nonvis_correct']) / view_res['val_nonvis_total']
            view_res['val_pl_nonvis_acc'] = float(view_res['val_pl_nonvis_correct']) / view_res['val_nonvis_total']

            n_view_res[view] = view_res

        mean_val_loss = np.mean([r['val_loss'] for r in n_view_res.values()])

        val_acc = sum([r['val_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_visual_acc = sum([r['val_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_nonvis_acc = sum([r['val_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        val_pl_acc = sum([r['val_pl_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_pl_visual_acc = sum([r['val_pl_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_pl_nonvis_acc = sum([r['val_pl_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        res = {
            f'{mode}/loss': mean_val_loss,
            f'{mode}/acc': val_acc,
            f'{mode}/acc_visual': val_visual_acc,
            f'{mode}/acc_nonvis': val_nonvis_acc,
            f'{mode}/pl_acc': val_pl_acc,
            f'{mode}/pl_acc_visual': val_pl_visual_acc,
            f'{mode}/pl_acc_nonvis': val_pl_nonvis_acc,
            f'{mode}/all_view_res': n_view_res,
        }

        # test (ran once at the end of training)
        if mode == 'test':
            self.best_test_res = dict(res)

        # val (keep track of best results)
        else:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_res = dict(res)

        # results to save
        results_dict = self.best_test_res if mode == 'test' else self.best_val_res

        best_loss = results_dict[f'{mode}/loss']
        best_acc = results_dict[f'{mode}/acc']
        best_acc_visual = results_dict[f'{mode}/acc_visual']
        best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']
        best_pl_acc = results_dict[f'{mode}/pl_acc']
        best_pl_acc_visual = results_dict[f'{mode}/pl_acc_visual']
        best_pl_acc_nonvis = results_dict[f'{mode}/pl_acc_nonvis']

        seed = self.cfg['train']['random_seed']
        json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')

        # save results
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, sort_keys=True, indent=4)

        # print best result
        print("\nBest-----:")
        print(f'Best {mode} Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) | Val Loss: {best_loss:0.8f} ')
        print("------------")

        if self.log_data:
            wandb.log(res)
        return dict(
            val_loss=mean_val_loss,
            val_acc=val_acc,
            val_visual_acc=val_visual_acc,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_acc=val_pl_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
        )

    def test_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch)
            probs = out['probs']
            num_steps = out['num_steps']
            objects = batch[3]
            annotation = batch[4]

            probs = F.softmax(probs, dim=-1)
            pred_ans = probs.argmax(-1)

            all_view_results[view] = dict(
                annotation=annotation,
                objects=objects,
                pred_ans=pred_ans,
                num_steps=num_steps,
            )

        return dict(
            all_view_results=all_view_results,
        )

    def test_epoch_end(self, all_outputs, mode='test'):
        test_results = {v: list() for v in range(self.num_views)}

        for out in all_outputs:
            for view in range(self.num_views):
                view_res = out['all_view_results']
                bs = view_res[view]['pred_ans'].shape[0]
                for b in range(bs):
                    test_results[view].append({
                        'annotation': view_res[view]['annotation'][b],
                        'objects': (
                            view_res[view]['objects'][0][b],
                            view_res[view]['objects'][1][b],
                        ),
                        'pred_ans': int(view_res[view]['pred_ans'][b]),
                        'num_steps': int(view_res[view]['num_steps'][b]),
                    })

        test_pred_save_path = self.save_path
        if not os.path.exists(test_pred_save_path):
            os.makedirs(test_pred_save_path)

        model_type = self.__class__.__name__.lower()
        json_file = os.path.join(test_pred_save_path, f'{model_type}_results.json')
        with open(json_file, 'w') as f:
            json.dump(test_results, f, sort_keys=True, indent=4)

