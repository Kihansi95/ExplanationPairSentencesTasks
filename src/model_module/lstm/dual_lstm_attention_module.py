import os
import warnings

from typing import Union

from os import path

import pytorch_lightning as pl
import torch

from torch import optim, nn
from torchtext.vocab.vectors import pretrained_aliases as pretrained

import torchmetrics as m

from data_module.constant import *

from model.lstm.dual_lstm_attention import DualPairLstmAttention
from modules.const import Mode
from modules.logger import log
from modules import metrics, rescale, INF
from modules.loss import IoU, KLDivLoss


class DualLSTMAttentionModule(pl.LightningModule):
	
	def __init__(self, cache_path, mode, vocab, pretrained_vectors: Union[str, torch.tensor]=None,
	             lambda_entropy:float = 0.,
	             lambda_lagrange:float = 0.,
	             lambda_supervise:float = 0.,
	             lambda_heuristic: float = 0.,
	             data='Unk data',
	             num_class=-1,
	             n_lstm=1,
	             **kwargs):
		super(DualLSTMAttentionModule, self).__init__()
		
		# log hyperparameters into hparams.yaml
		self.save_hyperparameters('data', 'n_lstm', 'lambda_entropy', 'lambda_supervise', 'lambda_lagrange', 'lambda_heuristic')
		self.data = data
		
		if pretrained_vectors is not None and isinstance(pretrained_vectors, str):
			vector_path = path.join(cache_path, '..', '.vector_cache')
			os.makedirs(vector_path, exist_ok=True)
			vectors = pretrained[pretrained_vectors](cache=vector_path)
			pretrained_vectors = [vectors[token] for token in vocab.get_itos()]
			pretrained_vectors = torch.stack(pretrained_vectors)

		self.model = DualPairLstmAttention(pretrained_embedding=pretrained_vectors,
		                                   vocab_size=len(vocab),
		                                   d_embedding=kwargs['d_embedding'],
		                                   padding_idx=vocab[PAD_TOK],
		                                   n_class=num_class,
		                                   n_lstm=n_lstm,
		                                   attention_raw=True)
		
		self.loss_fn = nn.CrossEntropyLoss()
		self.supervise_loss_fn = IoU()
		self.lagrange_loss_fn = nn.MSELoss()
		self.heuristic_loss_fn = KLDivLoss(reduction='batchmean', log_target=True)
		
		self.num_class = num_class
		self.vocab = vocab
		self._mode = mode
		self.lambda_entropy = lambda_entropy
		self.lambda_supervise = lambda_supervise
		self.lambda_lagrange = lambda_lagrange
		self.lambda_heuristic = lambda_heuristic
		
		template_y_metrics = m.MetricCollection({
			'y:accuracy': m.Accuracy(num_classes=num_class, multiclass=True),
			'y:fscore': m.F1Score(num_classes=num_class, multiclass=True)
		})
		
		with warnings.catch_warnings():
			template_attention_metrics = m.MetricCollection({
				'a:AUROC': m.AUROC(average='micro'),
				'a:AUPRC': m.AveragePrecision(average='micro'),
				'a:AURecall': metrics.AURecall(),
				'a:AUPrecision': metrics.AUPrecision(),
				'a:Jaccard': metrics.PowerJaccard(),
				'a:Specificity': m.Specificity(),
				'a:Dice': m.Dice(),
				'a:IoU': m.JaccardIndex(num_classes=2),
			})
			warnings.simplefilter("ignore")
		
		PHASES = ['TRAIN', 'VAL', 'TEST']
		self.y_metrics = nn.ModuleDict({
			phase: template_y_metrics.clone() for phase in PHASES
		})
		self.attention_metrics = nn.ModuleDict({
			phase: template_attention_metrics.clone() for phase in PHASES
		})
		self.entropy_metric = nn.ModuleDict({
			phase: metrics.Entropy(normalize=True) for phase in PHASES
		})
		self.reg_term_metric = nn.ModuleDict({
			phase: m.MeanMetric() for phase in PHASES
		})
		
	def forward(self, premise_ids, hypothesis_ids, premise_padding, hypothesis_padding):
		return self.model(
			premise_ids=premise_ids,
			hypothesis_ids=hypothesis_ids,
			premise_padding=premise_padding,
			hypothesis_padding=hypothesis_padding
		)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters())
		# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
		# return [optimizer], [lr_scheduler]
		return optimizer
	
	def training_step(self, batch, batch_idx, val=False):
		
		y_true = batch['y_true']
		padding_mask = batch['padding_mask']
		a_true = batch['a_true']
		
		y_hat, a_hat = self(
			premise_ids=batch['premise_ids'],
			hypothesis_ids=batch['hypothesis_ids'],
			premise_padding=padding_mask['premise'],
			hypothesis_padding=padding_mask['hypothesis']
		)
		
		loss_classif = self.loss_fn(y_hat, y_true)
		
		loss_entropy_pair = {s: metrics.entropy(a_hat[s], padding_mask[s], normalize=True, average='micro') for s in a_hat}
		loss_entropy = 0.5 * sum(loss_entropy_pair.values())
		
		# Sigmoid for IoU loss
		flat_a_hat = [self.flatten_attention(attention=a_hat[s], condition=y_true > 0, pad_mask=padding_mask[s], normalize='sigmoid') for s in a_hat]
		flat_a_true = [self.flatten_attention(attention=a_true[s], condition=y_true > 0, pad_mask=padding_mask[s]) for s in a_true]
		
		flat_a_hat = torch.cat(flat_a_hat)
		flat_a_true = torch.cat(flat_a_true)
		
		if flat_a_true is None:
			loss_supervise = torch.tensor(0.).type_as(loss_classif)
		else:
			loss_supervise = self.supervise_loss_fn(flat_a_hat, flat_a_true)
		
		loss = loss_classif + self.lambda_entropy * loss_entropy \
		       + self.lambda_supervise * loss_supervise
		
		return {'loss': loss,
		        'loss_entropy': loss_entropy,
		        'loss_supervise': loss_supervise,
		        'y_hat': y_hat, 'y_true': y_true, 'a_hat': a_hat, 'a_true': a_true, 'padding_mask': padding_mask}
	
	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx, True)
	
	def test_step(self, batch, batch_idx, dataloader_idx=None):

		y_true = batch['y_true']
		padding_mask = batch['padding_mask']
		y_hat, a_hat = self(
			premise_ids=batch['premise_ids'],
			hypothesis_ids=batch['hypothesis_ids'],
			premise_padding=padding_mask['premise'],
			hypothesis_padding=padding_mask['hypothesis'])
		
		a_hat = {s: a.detach() for s, a in a_hat.items()}
		
		return {'y_hat': y_hat,
		        'y_true': y_true,
		        'a_hat': a_hat,
		        'a_true': batch['a_true'],
		        'padding_mask': padding_mask}
	
	def predict_step(self, batch, batch_idx):

		padding_mask = batch['padding_mask']
		y_hat, a_hat = self(
			premise_ids=batch['premise_ids'],
			hypothesis_ids=batch['hypothesis_ids'],
			premise_padding=padding_mask['premise'],
			hypothesis_padding=padding_mask['hypothesis'])
		
		return {'y_hat': y_hat.argmax(axis=-1),
		        'a_hat': a_hat,
		        'padding_mask': padding_mask}
	
	def step_end(self, outputs, stage: str = 'TEST'):
		
		a_hat, a_true = outputs['a_hat'], outputs['a_true']
		y_hat, y_true = outputs['y_hat'], outputs['y_true']
		padding_mask = outputs['padding_mask']
		
		flat_a_hat = [self.flatten_attention(attention=a_hat[s], condition=y_true > 0, pad_mask=padding_mask[s], normalize='softmax_rescale') for s in a_hat]
		flat_a_true = [self.flatten_attention(attention=a_true[s], condition=y_true > 0, pad_mask=padding_mask[s]) for s in a_true]
		
		flat_a_hat = torch.cat(flat_a_hat)
		flat_a_true = torch.cat(flat_a_true)
		
		# log attentions metrics
		if flat_a_hat.size(0) > 0:
			metric_a = self.attention_metrics[stage](flat_a_hat, flat_a_true)
			metric_a['a:entropy'] = 0.5 * sum([metrics.entropy(a_hat[s], padding_mask[s], normalize=True, average='micro') for s in a_hat])
			metric_a = {f'{stage}/{k}': v.item() for k, v in metric_a.items()}  # put metrics within same stage under the same folder
			self.log_dict(metric_a, prog_bar=True)
		
		# log for classification metrics
		metric_y = self.y_metrics[stage](y_hat, y_true)
		metric_y = {f'{stage}/{k}': v for k, v in metric_y.items()}  # put metrics within same stage under the same folder
		self.log_dict(metric_y, prog_bar=True)
		
		if stage != 'TEST':
			# if not in test stage, log loss metrics
			loss_names = [k for k in outputs.keys() if 'loss' in k]
			for loss_metric in loss_names:
				self.log(f'{stage}/{loss_metric}', outputs[loss_metric], prog_bar=True)
	
	def training_step_end(self, outputs):
		return self.step_end(outputs, stage='TRAIN')
	
	def validation_step_end(self, outputs):
		return self.step_end(outputs, stage='VAL')
	
	def test_step_end(self, outputs):
		return self.step_end(outputs, stage='TEST')
	
	def flatten_attention(self, attention, pad_mask, condition=None, normalize:str =''):
		"""
		Filter attention
		Args:
		 a_hat ():
		 a_true ():
		 condition ():
		 pad_mask ():
		 y_hat ():
		 normalize (str): softmax, softmax_rescale, sigmoid

		Returns:

		"""
		if condition is None:
			condition = torch.ones(attention.size(0)).type(torch.bool)
		
		if (~condition).all():
			return torch.tensor([])
		
		# Filter by condition on y
		attention = attention[condition]
		pad_mask = pad_mask[condition]
		
		# Ir normalize specify:
		if normalize == 'sigmoid':
			attention = torch.sigmoid(attention)
		if 'softmax' in normalize:
			attention = torch.softmax(attention + pad_mask.float() * -INF, dim=1)
		if 'rescale' in normalize:
			attention = rescale(attention, pad_mask)
			
		# Filter by mask
		return attention[~pad_mask]
	
	def on_train_start(self):   
		init_hp_metrics = {f'TEST/{k}': 0 for k in self.y_metrics['TEST']}
		init_hp_metrics.update({f'TEST/{k}': 0 for k in self.attention_metrics['TEST']})
		init_hp_metrics.update({f'TEST/a:entropy': 0})
		self.logger.log_hyperparams(self.hparams, init_hp_metrics)
	
	def on_train_epoch_start(self):
		# Make new line for progress bar.
		# See: https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
		if self._mode == Mode.DEV:
			print()
	
	def epoch_end(self, stage):
		if self._mode == Mode.EXP:
			metric = self.y_metrics[stage].compute()
			try:
				metric.update(self.attention_metrics[stage].compute())
			except RuntimeError as e:
				log.error(e)
				
			metric.update({
				'a:entropy': self.entropy_metric[stage].compute()
			})
			metric = {k: round(v.item(), 3) for k, v in metric.items()}
			log.info(f'Epoch {self.current_epoch} {stage}:{metric}')
			
	def on_train_epoch_end(self):
		return self.epoch_end('TRAIN')
		
	def on_validation_epoch_end(self):
		return self.epoch_end('VAL')
		
	def __str__(self):
		return str(self.model)

