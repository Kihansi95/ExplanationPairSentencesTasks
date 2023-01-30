import os

from typing import Union

from torch import optim, nn

from torchtext.vocab.vectors import pretrained_aliases as pretrained

import torchmetrics as m

from data_module.yelp_hat import *
from modules.const import Mode

from modules.logger import log

from model.lstm.single_lstm import SingleLstm


class SingleLSTMModule(pl.LightningModule):
	
	def __init__(self, cache_path, mode, vocab, pretrained_vectors: Union[str, torch.tensor]=None,
	             data='Unk data',
	             num_class=-1,
	             n_context=1,
	             concat_context=False,
	             **kwargs):
		super(SingleLSTMModule, self).__init__()
		
		# log hyperparameters into hparams.yaml
		self.save_hyperparameters('data', 'n_context', 'd_embedding')
		self.data = data
		
		if pretrained_vectors is not None and isinstance(pretrained_vectors, str):
			vector_path = path.join(cache_path, '..', '.vector_cache')
			os.makedirs(vector_path, exist_ok=True)
			vectors = pretrained[pretrained_vectors](cache=vector_path)
			pretrained_vectors = [vectors[token] for token in vocab.get_itos()]
			pretrained_vectors = torch.stack(pretrained_vectors)

		self.model = SingleLstm(pretrained_embedding=pretrained_vectors,
		                                 vocab_size=len(vocab),
		                                 d_embedding=kwargs['d_embedding'],
		                                 padding_idx=vocab[SpecToken.PAD],
		                                 n_class=num_class,
		                                 n_lstm=n_context,
		                                 concat_context=concat_context)
		
		self.loss_fn = nn.CrossEntropyLoss()
		
		self.num_class = num_class
		self.vocab = vocab
		self._mode = mode
		
		self.caching_weight = None
		
		template_y_metrics = m.MetricCollection({
			'y:accuracy': m.Accuracy(num_classes=num_class, multiclass=True),
			'y:fscore': m.F1Score(num_classes=num_class, multiclass=True)
		})
		
		PHASES = ['TRAIN', 'VAL', 'TEST']
		self.y_metrics = nn.ModuleDict({
			phase: template_y_metrics.clone() for phase in PHASES
		})
		
	def forward(self, ids):
		return self.model(ids=ids)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters())
		return optimizer
	
	def training_step(self, batch, batch_idx, val=False):
		
		y_true = batch['y_true']
		y_hat = self(
			ids=batch['token_ids']
		)
		loss = self.loss_fn(y_hat, y_true)
		
		return {'loss': loss, 'y_hat': y_hat, 'y_true': y_true}
	
	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx, True)
	
	def test_step(self, batch, batch_idx, dataloader_idx=None):

		y_hat = self(
			ids=batch['token_ids']
		)
		
		return {'y_hat': y_hat,
		        'y_true': batch['y_true']}
	
	def predict_step(self, batch, batch_idx):
		y_hat = self.test_step(batch, batch_idx)
		
		return {'y_hat': y_hat.argmax(axis=-1)}
	
	def step_end(self, outputs, stage: str = 'TEST'):
		
		y_hat, y_true = outputs['y_hat'], outputs['y_true']
		
		# log for classification metrics
		metric_y = self.y_metrics[stage](y_hat, y_true)
		metric_y = {f'{stage}/{k}': v for k, v in metric_y.items()}  # put metrics within same stage under the same folder
		self.log_dict(metric_y, prog_bar=True)
		
		if stage != 'TEST':
			# if not in test stage, log loss metrics
			loss_names = [k for k in outputs.keys() if 'loss' in k]
			for loss_metric in loss_names:
				self.log(f'{stage}/{loss_metric}', outputs[loss_metric], prog_bar=True, sync_dist=True)
	
	def training_step_end(self, outputs):
		return self.step_end(outputs, stage='TRAIN')
	
	def validation_step_end(self, outputs):
		return self.step_end(outputs, stage='VAL')
	
	def test_step_end(self, outputs):
		return self.step_end(outputs, stage='TEST')
	
	def on_train_start(self):   
		init_hp_metrics = {f'TEST/{k}': 0 for k in self.y_metrics['TEST']}
		self.logger.log_hyperparams(self.hparams, init_hp_metrics)
	
	def on_train_epoch_start(self):
		# Make new line for progress bar.
		# See: https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
		if self._mode == Mode.DEV:
			print()
	
	def epoch_end(self, stage):
		if self._mode == Mode.EXP:
			metric = self.y_metrics[stage].compute()
			metric = {k: round(v.item(), 3) for k, v in metric.items()}
			log.info(f'Epoch {self.current_epoch} {stage}:{metric}')
			
	def on_train_epoch_end(self):
		return self.epoch_end('TRAIN')
		
	def on_validation_epoch_end(self):
		return self.epoch_end('VAL')
		
	def __str__(self):
		return str(self.model)

