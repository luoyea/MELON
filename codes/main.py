import math
import csv
import os
import random
import sys
from time import time

import numpy as np
import torch
import torch.optim as optim

from utility.parser import parse_args
from utility.batch_test import *
from Models import *

outer_args = parse_args()


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def append_csv_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_metrics(ret):
    return {
        'precision@10': float(ret['precision'][0]),
        'recall@10': float(ret['recall'][0]),
        'ndcg@10': float(ret['ndcg'][0]),
        'precision@20': float(ret['precision'][-1]),
        'recall@20': float(ret['recall'][-1]),
        'ndcg@20': float(ret['ndcg'][-1]),
        'auc': float(ret.get('auc', 0.0))
    }


def inject_modality_noise(features, ratio, mode='none', seed=0):
    if ratio <= 0 or mode == 'none':
        return features
    rng = np.random.RandomState(seed)
    noisy = features.copy()
    n_items = noisy.shape[0]
    n_noisy = int(n_items * ratio)
    if n_noisy <= 0:
        return noisy

    idx = rng.choice(n_items, n_noisy, replace=False)
    if mode == 'gaussian':
        std = np.std(noisy, axis=0, keepdims=True) + 1e-8
        noise = rng.normal(loc=0.0, scale=0.1, size=(n_noisy, noisy.shape[1])) * std
        noisy[idx] = noisy[idx] + noise
    elif mode == 'zero':
        noisy[idx] = 0.0
    elif mode == 'shuffle':
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        noisy[idx] = noisy[shuffled]
    else:
        raise ValueError('Unsupported noise_mode: {}'.format(mode))

    return noisy


class Trainer(object):
    def __init__(self, data_config, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.train_items = data_config['train_items']
        self.nonzero_idx_img = data_config['nonzero_idx_img']
        self.nonzero_idx_txt = data_config['nonzero_idx_txt']
        self.ui_index = {}
        index = 0
        for k in range(len(self.train_items)):
            sorted_list = sorted(self.train_items[k])
            for v in sorted_list:
                pair = str(k) + '_' + str(v)
                self.ui_index[pair] = index
                index += 1
        self.feat_embed_dim = args.feat_embed_dim
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.has_norm = args.has_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.lamb = self.regs[1]
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delta = args.delta
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.nonzero_idx = data_config['nonzero_idx']

        # Ablation / MQ extension args
        self.use_mqm = bool(args.use_mqm)
        self.use_aqg = bool(args.use_aqg)
        self.fixed_weight = bool(args.fixed_weight)
        self.n_aspects = args.n_aspects
        self.eta_mq = args.eta_mq
        self.lambda_ent = args.lambda_ent
        self.mq_hidden_dim = args.mq_hidden_dim
        self.q_lambda = args.q_lambda
        self.noise_ratio = args.noise_ratio
        self.noise_mode = args.noise_mode
        self.save_gate_weights = bool(args.save_gate_weights)
        self.gate_log_interval = args.gate_log_interval
        self.gate_log_path = os.path.join('logs', '{}_{}_gate_stats.csv'.format(self.dataset, self.model_name))
        self.run_log_path = os.path.join('logs', '{}_{}_run.log'.format(self.dataset, self.model_name))
        self.epoch_log_path = os.path.join('logs', '{}_{}_epoch_metrics.csv'.format(self.dataset, self.model_name))
        self.summary_log_path = os.path.join('logs', '{}_{}_summary.csv'.format(self.dataset, self.model_name))
        self.global_summary_path = os.path.join('logs', 'all_experiments_summary.csv')
        self.best_epoch = -1
        self.best_val_result = None

        base_image_feats = np.load('data/{}/image_feat.npy'.format(self.dataset))
        base_text_feats = np.load('data/{}/text_feat.npy'.format(self.dataset))
        self.image_feats = inject_modality_noise(base_image_feats, self.noise_ratio, self.noise_mode, seed=args.seed + 11)
        self.text_feats = inject_modality_noise(base_text_feats, self.noise_ratio, self.noise_mode, seed=args.seed + 23)

        self.model = MELON(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.nonzero_idx_img,
            self.nonzero_idx_txt,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.train_items,
            self.n_layers,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            use_mqm=self.use_mqm,
            use_aqg=self.use_aqg,
            fixed_weight=self.fixed_weight,
            n_aspects=self.n_aspects,
            eta_mq=self.eta_mq,
            lambda_ent=self.lambda_ent,
            mq_hidden_dim=self.mq_hidden_dim,
            q_lambda=self.q_lambda,
        )

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        if self.save_gate_weights:
            with open(self.gate_log_path, 'w', encoding='utf-8') as f:
                f.write('epoch,w_img,w_txt,q_img,q_txt,use_mqm,use_aqg,fixed_weight,n_aspects,noise_ratio,noise_mode\n')
        for p in [self.epoch_log_path, self.summary_log_path]:
            if os.path.exists(p):
                os.remove(p)

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            users, neg_items = [1], [1]
            user_ice, item_ice, user_mce, item_mce, img_query, txt_query, _, _, uv_agg, ut_agg, v_rel_mlp, t_rel_mlp, image_features, text_features = self.model(users, neg_items)
        result = test_torch(
            user_ice,
            item_ice,
            user_mce,
            item_mce,
            img_query,
            txt_query,
            uv_agg,
            ut_agg,
            image_features,
            text_features,
            v_rel_mlp,
            t_rel_mlp,
            users_to_test,
            is_val,
            self.adj,
            self.alpha,
            self.beta,
            self.gamma,
            self.model,
        )
        return result

    def _build_model(self):
        return MELON(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.nonzero_idx_img,
            self.nonzero_idx_txt,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.train_items,
            self.n_layers,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            use_mqm=self.use_mqm,
            use_aqg=self.use_aqg,
            fixed_weight=self.fixed_weight,
            n_aspects=self.n_aspects,
            eta_mq=self.eta_mq,
            lambda_ent=self.lambda_ent,
            mq_hidden_dim=self.mq_hidden_dim,
            q_lambda=self.q_lambda,
        )

    def train(self):
        nonzero_idx = torch.tensor(self.nonzero_idx, device=self.device).long().T
        self.adj = torch.sparse.FloatTensor(
            nonzero_idx,
            torch.ones((nonzero_idx.size(1)), device=self.device),
            (self.n_users, self.n_items),
        ).to_dense().to(self.device)
        stopping_step = 0
        best_recall = 0

        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0.0, 0.0, 0.0, 0.0
            n_batch = data_generator.n_train // args.batch_size + 1
            self.model.reset_gate_monitor()

            for _ in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                users, pos_items, neg_items = data_generator.sample()
                pos_pairs = []
                for i in range(len(users)):
                    pos_pair = str(users[i]) + '_' + str(pos_items[i])
                    pos_pairs.append(self.ui_index[pos_pair])

                user_ice, item_ice, user_mce, item_mce, img_query, txt_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, _, _, _, _ = self.model(users, neg_items)

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(
                    user_ice, item_ice, user_mce, item_mce,
                    img_query, txt_query,
                    comp_rel_v, comp_rel_t,
                    comp_rel_v_neg, comp_rel_t_neg,
                    users, pos_items, neg_items, pos_pairs
                )

                batch_emb_loss = self.decay * batch_emb_loss
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                del user_ice, item_ice, user_mce, item_mce, img_query, txt_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.lr_scheduler.step()

            if math.isnan(loss):
                print('ERROR: loss is nan.')
                sys.exit()

            perf_str = 'Pre_Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
            print(perf_str)

            if self.model.enable_mq_component and epoch % self.gate_log_interval == 0:
                gate_stats = self.model.get_gate_monitor()
                gate_msg = 'GateStats epoch=%d w_img=%.5f w_txt=%.5f q_img=%.5f q_txt=%.5f' % (
                    epoch, gate_stats['w_img'], gate_stats['w_txt'], gate_stats['q_img'], gate_stats['q_txt'])
                print(gate_msg)
                if self.save_gate_weights:
                    with open(self.gate_log_path, 'a', encoding='utf-8') as f:
                        f.write('{},{:.6f},{:.6f},{:.6f},{:.6f},{},{},{},{},{},{}\n'.format(
                            epoch, gate_stats['w_img'], gate_stats['w_txt'], gate_stats['q_img'], gate_stats['q_txt'],
                            int(self.use_mqm), int(self.use_aqg), int(self.fixed_weight), self.n_aspects,
                            self.noise_ratio, self.noise_mode
                        ))

            if epoch % args.verbose != 0:
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            t3 = time()

            if args.verbose > 0:
                perf_str = 'Pre_Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss)
                perf_str_value10 = 'precision@10=[%.5f], recall@10=[%.5f] , ndcg@10=[%.5f]' % (
                    ret['precision'][0], ret['recall'][0], ret['ndcg'][0])
                perf_str_value20 = 'precision@20=[%.5f], recall@20=[%.5f] , ndcg@20=[%.5f]' % (
                    ret['precision'][-1], ret['recall'][-1], ret['ndcg'][-1])
                print(perf_str)
                print(perf_str_value10)
                print(perf_str_value20)

                epoch_row = {
                    'dataset': self.dataset,
                    'model_name': self.model_name,
                    'epoch': epoch,
                    'split': 'val',
                    'loss': float(loss),
                    'mf_loss': float(mf_loss),
                    'emb_loss': float(emb_loss),
                    'reg_loss': float(reg_loss),
                    **flatten_metrics(ret),
                    'use_mqm': int(self.use_mqm),
                    'use_aqg': int(self.use_aqg),
                    'fixed_weight': int(self.fixed_weight),
                    'n_aspects': int(self.n_aspects),
                    'noise_ratio': float(self.noise_ratio),
                    'noise_mode': self.noise_mode,
                    'seed': int(args.seed)
                }
                append_csv_row(
                    self.epoch_log_path,
                    list(epoch_row.keys()),
                    epoch_row
                )

            if ret['recall'][-1] > best_recall:
                best_recall = ret['recall'][-1]
                stopping_step = 0
                self.best_epoch = epoch
                self.best_val_result = ret
                torch.save({self.model_name: self.model.state_dict()}, './models/' + self.dataset + '_' + self.model_name)
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        self.model = self._build_model()
        self.model.load_state_dict(torch.load('./models/' + self.dataset + '_' + self.model_name, map_location=torch.device('cpu'))[self.model_name])
        self.model.to(self.device)
        test_ret = self.test(users_to_test, is_val=False)
        print('Final ', test_ret)

        summary_row = {
            'dataset': self.dataset,
            'model_name': self.model_name,
            'best_epoch': int(self.best_epoch),
            'best_val_recall@10': float(self.best_val_result['recall'][0]) if self.best_val_result is not None else 0.0,
            'best_val_recall@20': float(self.best_val_result['recall'][-1]) if self.best_val_result is not None else 0.0,
            'best_val_ndcg@10': float(self.best_val_result['ndcg'][0]) if self.best_val_result is not None else 0.0,
            'best_val_ndcg@20': float(self.best_val_result['ndcg'][-1]) if self.best_val_result is not None else 0.0,
            'final_test_precision@10': float(test_ret['precision'][0]),
            'final_test_recall@10': float(test_ret['recall'][0]),
            'final_test_ndcg@10': float(test_ret['ndcg'][0]),
            'final_test_precision@20': float(test_ret['precision'][-1]),
            'final_test_recall@20': float(test_ret['recall'][-1]),
            'final_test_ndcg@20': float(test_ret['ndcg'][-1]),
            'final_test_auc': float(test_ret.get('auc', 0.0)),
            'use_mqm': int(self.use_mqm),
            'use_aqg': int(self.use_aqg),
            'fixed_weight': int(self.fixed_weight),
            'n_aspects': int(self.n_aspects),
            'noise_ratio': float(self.noise_ratio),
            'noise_mode': self.noise_mode,
            'seed': int(args.seed),
            'model_path': os.path.join('models', self.dataset + '_' + self.model_name),
            'gate_log_path': self.gate_log_path if self.save_gate_weights else ''
        }
        append_csv_row(
            self.summary_log_path,
            list(summary_row.keys()),
            summary_row
        )
        append_csv_row(
            self.global_summary_path,
            list(summary_row.keys()),
            summary_row
        )


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args(True)
    run_log_fp = None
    try:
        os.makedirs('logs', exist_ok=True)
        run_log_fp = open(os.path.join('logs', '{}_{}_run.log'.format(args.dataset, args.model_name or 'run')), 'a', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, run_log_fp)
        sys.stderr = Tee(sys.stderr, run_log_fp)
    except Exception:
        pass
    set_seed(args.seed)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['train_items'] = data_generator.train_items

    nonzero_idx = data_generator.nonzero_idx()
    nonzero_idx_img = data_generator.nonzero_idx_img()
    nonzero_idx_txt = data_generator.nonzero_idx_txt()
    config['nonzero_idx'] = nonzero_idx
    config['nonzero_idx_img'] = nonzero_idx_img
    config['nonzero_idx_txt'] = nonzero_idx_txt

    trainer = Trainer(config, args)
    trainer.train()
