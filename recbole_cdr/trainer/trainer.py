

r"""
recbole_cdr.trainer.trainer
################################
"""
import torch
import numpy as np
from time import time
from tqdm import tqdm
from recbole.trainer import Trainer
from recbole_cdr.utils import train_mode2state
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, WandbLogger
from recbole.data.dataloader import FullSortEvalDataLoader
from collections import OrderedDict

class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                if saved and self.start_epoch >= self.epochs:
                    self._save_checkpoint(-1, verbose=verbose)

                self.eval_collector.data_collect(train_data)
                if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
                    train_data.get_model(self.model)
                valid_step = 0

                for epoch_idx in range(self.start_epoch, self.epochs):
                    # train
                    training_start_time = time()
                    self.model.epoch_start()
                    train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                    self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                    training_end_time = time()
                    train_loss_output = \
                        self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                    if verbose:
                        self.logger.info(train_loss_output)
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                    self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

                    # eval
                    if self.eval_step <= 0 or not valid_data:
                        if saved:
                            self._save_checkpoint(epoch_idx, verbose=verbose)
                        continue
                    if (epoch_idx + 1) % self.eval_step == 0:
                        valid_start_time = time()
                        # Target
                        self.model.set_full_sort_func('Target')
                        self.model.set_predict_func('Target')
                        valid_score_target, valid_result_target = self._valid_epoch(valid_data[1], show_progress=show_progress)
                        # Source
                        self.model.set_full_sort_func('Source')
                        self.model.set_predict_func('Source')
                        valid_score_source, valid_result_source = self._valid_epoch(valid_data[0], show_progress=show_progress)
                        # All, addition by default
                        valid_score = valid_score_source + valid_score_target
                        valid_result = OrderedDict({key: value + valid_result_source[key] for key, value in valid_result_target.items()})

                        self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                            valid_score,
                            self.best_valid_score,
                            self.cur_step,
                            max_step=self.stopping_step,
                            bigger=self.valid_metric_bigger
                        )
                        valid_end_time = time()
                        valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                            + ": %.2fs, " + set_color("valid_score_source", 'blue') + ": %f]" + set_color("valid_score_target", 'blue') + ": %f]") % \
                                            (epoch_idx, valid_end_time - valid_start_time, valid_score_source, valid_score_target)
                        valid_result_output_source = set_color('valid result source', 'blue') + ': \n' + dict2str(valid_result_source)
                        valid_result_output_target = set_color('valid result target', 'blue') + ': \n' + dict2str(valid_result_target)
                        # valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                        #                     + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                        #                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                        # valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                        if verbose:
                            self.logger.info(valid_score_output)
                            self.logger.info(valid_result_output_source)
                            self.logger.info(valid_result_output_target)
                        self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                        self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                        if update_flag:
                            if saved:
                                self._save_checkpoint(epoch_idx, verbose=verbose)
                            self.best_valid_result = valid_result

                        if callback_fn:
                            callback_fn(epoch_idx, valid_score)

                        if stop_flag:
                            stop_output = 'Finished training, best eval result in epoch %d' % \
                                        (epoch_idx - self.cur_step * self.eval_step)
                            if verbose:
                                self.logger.info(stop_output)
                            break

                        valid_step+=1

                self._add_hparam_to_tensorboard(self.best_valid_score)
                return self.best_valid_score, self.best_valid_result
                

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, analyze=False, user_count=None):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head='eval')

        if analyze:
            evaled_user_count = {idx: user_count[user] for idx, user in enumerate(eval_data.uid_list.tolist())}
            user_id, count = torch.tensor(list(evaled_user_count.keys())), torch.tensor(list(evaled_user_count.values()))
            # user_id, count = torch.arange(struct['rec.topk'].shape[0]), struct['rec.topk'][:, -1]
            # group_point = [1, 2, 5, count.max() + 100] # epinions
            # group_point = [7, 15, 30, count.max() + 100] # douban
            group_point = [3, 5, 10, count.max() + 100] # amazon
            left = 0
            for group_idx, point in enumerate(group_point):
                right = point
                filtered = user_id[(count > left) & (count <= right)]
                # Recall
                pos_index, pos_len = torch.split(struct['rec.topk'][filtered], [20, 1], dim=1)
                pos_index = pos_index.cpu().numpy()
                pos_len = pos_len.cpu().numpy().squeeze()

                recall = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
                recall = recall.mean(axis=0)[-1]

                len_rank = np.full_like(pos_len, pos_index.shape[1])
                idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

                iranks = np.zeros_like(pos_index, dtype=float)
                iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
                idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
                for row, idx in enumerate(idcg_len):
                    idcg[row, idx:] = idcg[row, idx - 1]

                ranks = np.zeros_like(pos_index, dtype=float)
                ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
                dcg = 1.0 / np.log2(ranks + 1)
                dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)
                ndcg = (dcg / idcg).mean(axis=0)[-1]

                print('group {} len {}--- recall@20:{}   ndcg@20:{}'.format(group_idx, len(filtered), recall, ndcg))
                left = right

        return result


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result
