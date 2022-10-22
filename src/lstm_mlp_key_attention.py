import os
import warnings

from argparse import ArgumentParser
from typing import Union
import json

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import callbacks as cb

from torch import optim, nn

from torchtext.vocab.vectors import pretrained_aliases as pretrained

import torchmetrics as m

from data_module.hatexplain import HateXPlainDM
from data_module.yelp_hat import *

from modules.logger import log, init_logging
from modules import metrics, env, rescale, INF

from model.lstm.lstm_mlp_key_attention import LstmMLPKeyAttention
from modules.loss import IoU


class LitModel(pl.LightningModule):
	
	def __init__(self, cache_path, mode, vocab, pretrained_vectors: Union[str, torch.tensor]=None,
	             lambda_entropy:float = 0.,
	             lambda_supervise:float = 0.,
	             lambda_lagrange:float = 0.,
	             data='Unk data',
	             num_class=-1,
	             n_lstm=1,
	             **kwargs):
		super(LitModel, self).__init__()
		
		# log hyperparameters into hparams.yaml
		self.save_hyperparameters('data', 'n_lstm', 'lambda_entropy', 'lambda_supervise', 'lambda_lagrange')
		self.data = data
		
		if pretrained_vectors is not None and isinstance(pretrained_vectors, str):
			vector_path = path.join(cache_path, '..', '.vector_cache')
			vectors = pretrained[pretrained_vectors](cache=vector_path)
			pretrained_vectors = [vectors[token] for token in vocab.get_itos()]
			pretrained_vectors = torch.stack(pretrained_vectors)

		self.model = LstmMLPKeyAttention(
			pretrained_embedding=pretrained_vectors,
			vocab_size=len(vocab),
			d_embedding=kwargs['d_embedding'],
			padding_idx=vocab[PAD_TOK],
			n_class=num_class,
			n_lstm=n_lstm,
			attention_raw=True)
		
		self.loss_fn = nn.CrossEntropyLoss()
		self.supervise_loss_fn = IoU()
		self.lagrange_loss_fn = nn.MSELoss()
		
		self.num_class = num_class
		self.vocab = vocab
		self._mode = mode
		self.lambda_entropy = lambda_entropy
		self.lambda_supervise = lambda_supervise
		self.lambda_lagrange = lambda_lagrange
		
		self.caching_weight = None
		
		template_y_metrics = m.MetricCollection({
			'y:accuracy': m.Accuracy(num_classes=num_class, multiclass=True),
			'y:fscore': m.F1Score(num_classes=num_class, multiclass=True)
		})
		
		with warnings.catch_warnings():
			template_attention_metrics = m.MetricCollection({
				'a:AUROC': m.AUROC(average='micro'),
				'a:AUPRC': m.AveragePrecision(average='micro'),
				'a:Jaccard': metrics.PowerJaccard(),
				'a:Jaccard2': metrics.PowerJaccard(power=2.),
				'a:Dice': m.Dice(),
				'a:IoU': m.JaccardIndex(num_classes=2),
				'a:Recall': metrics.AURecall(),
				'a:Precision': metrics.AUPrecision(),
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
		
	def forward(self, ids, mask):
		return self.model(ids=ids, mask=mask)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters())
		return optimizer
	
	def training_step(self, batch, batch_idx, val=False):
		
		y_true = batch['y_true']
		padding_mask = batch['padding_mask']
		a_true = batch['a_true']
		a_true_entropy = batch['a_true_entropy']
		y_hat, a_hat = self(ids=batch['token_ids'], mask=padding_mask)
		loss_classif = self.loss_fn(y_hat, y_true)
		a_hat_entropy = metrics.entropy(a_hat, padding_mask, normalize=True)
		loss_entropy = a_hat_entropy.mean()
		
		# Sigmoid for IoU loss
		flat_a_hat, flat_a_true = self.flatten_attention(a_hat=a_hat, a_true=a_true, condition=y_true > 0, pad_mask=padding_mask, normalize='sigmoid')
		
		if flat_a_true is None:
			loss_supervise = torch.tensor(0.).type_as(loss_classif)
		else:
			loss_supervise = self.supervise_loss_fn(flat_a_hat, flat_a_true)
			
		loss_lagrange = self.lagrange_loss_fn(a_hat_entropy, a_true_entropy)
		
		loss = loss_classif + self.lambda_entropy * loss_entropy + self.lambda_supervise * loss_supervise + self.lambda_lagrange * loss_lagrange
		
		return {'loss': loss, 'loss_entropy': loss_entropy, 'loss_supervise': loss_supervise, 'loss_lagrange': loss_lagrange,
		        'y_hat': y_hat, 'y_true': y_true, 'a_hat': a_hat, 'a_true': a_true, 'padding_mask': padding_mask}
	
	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx, True)
	
	def test_step(self, batch, batch_idx, dataloader_idx=None):

		y_hat, a_hat = self(ids=batch['token_ids'], mask=batch['padding_mask'])
		
		return {'y_hat': y_hat,
		        'y_true': batch['y_true'],
		        'a_hat': a_hat.detach(),
		        'a_true': batch['a_true'],
		        'padding_mask': batch['padding_mask']}
	
	def step_end(self, outputs, stage: str = 'TEST'):
		
		a_hat, a_true = outputs['a_hat'], outputs['a_true']
		y_hat, y_true = outputs['y_hat'], outputs['y_true']
		padding_mask = outputs['padding_mask']
		
		flat_a_hat, flat_a_true = self.flatten_attention(a_hat=a_hat, a_true=a_true, condition=y_true > 0, pad_mask=padding_mask, normalize='softmax_rescale')
		
		# log attentions metrics
		if flat_a_hat is not None and a_hat.size(0) > 0:
			metric_a = self.attention_metrics[stage](flat_a_hat, flat_a_true)
			metric_a['a:entropy'] = self.entropy_metric[stage](a_hat, padding_mask)
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
	
	def flatten_attention(self, a_hat, a_true, pad_mask, condition=None, normalize:str =''):
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
			condition = torch.ones(a_hat.size(0)).type(torch.bool)
		
		if (~condition).all():
			return None, None
		
		# Filter by condition on y
		a_hat = a_hat[condition]
		a_true = a_true[condition]
		pad_mask = pad_mask[condition]
		
		# Ir normalize specify:
		if normalize == 'sigmoid':
			a_hat = torch.sigmoid(a_hat)
		if 'softmax' in normalize:
			a_hat = torch.softmax(a_hat + pad_mask.float() * -INF, dim=1)
		if 'rescale' in normalize:
			a_hat = rescale(a_hat, pad_mask)
			
		# Filter by mask
		flat_a_hat = a_hat[~pad_mask]
		flat_a_true = a_true[~pad_mask]
		
		return flat_a_hat, flat_a_true
	
	
	def on_train_start(self):   
		init_hp_metrics = {f'TEST/{k}': 0 for k in self.y_metrics['TEST']}
		init_hp_metrics.update({f'TEST/{k}': 0 for k in self.attention_metrics['TEST']})
		init_hp_metrics.update({f'TEST/a:entropy': 0})
		self.logger.log_hyperparams(self.hparams, init_hp_metrics)
	
	def on_train_epoch_start(self):
		# Make new line for progress bar.
		# See: https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
		if self._mode == 'M_DEV':
			print()
	
	def epoch_end(self, stage):
		if self._mode == M_EXP:
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

def get_num_workers() -> int:
	"""
	Get maximum logical workers that a machine has
	Args:
		default (int): default value

	Returns:
		maximum workers number
	"""
	if hasattr(os, 'sched_getaffinity'):
		try:
			return len(os.sched_getaffinity(0))
		except Exception:
			pass
	
	num_workers = os.cpu_count()
	return num_workers if num_workers is not None else 0


def parse_argument(prog: str = __name__, description: str = 'Experimentation on NLP') -> dict:
	"""
	Parse arguments passed to the script.
	Args:
		prog (str): name of the programme (experimentation)
		description (str): What do we do to this script
	Returns:
		dictionary
	"""
	parser = ArgumentParser(prog=prog, description=description)
	
	# Optional stuff
	parser.add_argument('--disable_log_color', action='store_true', help='Activate for console does not support coloring')
	
	# Training params
	parser.add_argument('--cache', '-o', type=str, default=path.join(os.getcwd(), '..', '.cache'), help='Path to temporary directory to store output of training process')
	parser.add_argument('--mode', '-m', type=str, default='dev', help='Choose among f[dev, exp]. "exp" will disable the progressbar')
	parser.add_argument('--OAR_ID', type=int, help='Indicate whether we are in IGRIDA cluster mode')
	parser.add_argument('--num_workers', type=int, default=get_num_workers(), help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--accelerator', type=str, default='auto', help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--name', type=str, help='Experimentation name. If not given, use model name instead.')
	parser.add_argument('--version', type=str, help='Experimentation version')
	
	# For trainer setting
	parser.add_argument('--resume', '-r', action='store_true', help='Flag to resume the previous training process, detected by model name.')
	parser.add_argument('--epoch', '-e', type=int, default=1, help='Number training epoch. Default: 1')
	parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of data in batch. Default: 32')
	parser.add_argument('--strategy', '-s', type=str, help='')
	
	# Model configuration
	parser.add_argument('--vectors', type=str, help='Pretrained vectors. See more in torchtext Vocab, example: glove.840B.300d')
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--d_embedding', type=int, default=300, help='Embedding dimension, will be needed if vector is not precised')
	parser.add_argument('--d_hidden_lstm', type=int, default=-1)
	parser.add_argument('--n_lstm', type=int, default=1)
	
	# Data configuration
	parser.add_argument('--n_data', '-n', type=int, default=-1, help='Maximum data number for train+val+test, -1 if full dataset. Default: -1')
	parser.add_argument('--data', '-d', type=str, help='Choose dataset to train model')
	
	# Regularizer
	parser.add_argument('--lambda_entropy', type=float, default=0., help='multiplier for entropy')
	parser.add_argument('--lambda_supervise', type=float, default=0., help='multiplier for supervise')
	parser.add_argument('--lambda_lagrange', type=float, default=0., help='multiplier for relaxation of Lagrange (Supervision by entropy)')
	
	params = parser.parse_args()
	print('=== Parameters ===')
	print(json.dumps(vars(params), indent=4))
	
	# If data not provided, automatically get from '<cache>/dataset'
	params.mode = params.mode.lower()
	# params = {k: v for k, v in params.items() if v is not None}
	env.disable_tqdm = params.OAR_ID is not None
	return params

# const for mode
M_EXP = 'exp'
M_DEV = 'dev'

if __name__ == '__main__':
	
	args = parse_argument()
	
	DATA_CACHE = path.join(args.cache, 'dataset')
	MODEL_CACHE = path.join(args.cache, 'models')
	LOGS_CACHE = path.join(args.cache, 'logs')
	
	# init logging
	if args.mode == M_EXP:
		init_logging(cache_path=LOGS_CACHE, color=False, experiment=args.name, version=args.version)
	else:
		init_logging(color=True)
		
	dm_kwargs = dict(cache_path=DATA_CACHE,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			n_data=args.n_data)
	
	if args.data == 'hatexplain':
		dm = HateXPlainDM(**dm_kwargs)
	elif args.data == 'yelphat':
		dm = YelpHatDM(**dm_kwargs)
	elif args.data == 'yelphat50':
		dm = YelpHat50DM(**dm_kwargs)
	elif args.data == 'yelphat100':
		dm = YelpHat100DM(**dm_kwargs)
	elif args.data == 'yelphat200':
		dm = YelpHat200DM(**dm_kwargs)
	else:
		log.error(f'Unrecognized dataset: {args.data}')
		exit(1)
		
	# prepare data here before going to multiprocessing
	dm.prepare_data()
	model = LitModel(cache_path=MODEL_CACHE,
	                 mode=args.mode,
	                 vocab=dm.vocab,
	                 lambda_entropy=args.lambda_entropy,
	                 lambda_supervise=args.lambda_supervise,
	                 lambda_lagrange=args.lambda_lagrange,
	                 pretrained_vectors=args.vectors,
	                 n_lstm=args.n_lstm,
	                 d_hidden_lstm=args.d_hidden_lstm,
	                 d_embedding=args.d_embedding,
	                 data=args.data,
	                 num_class=dm.num_class
	                 )
	
	# call back
	early_stopping = cb.EarlyStopping('VAL/loss', patience=5, verbose=args.mode != M_EXP, mode='min')  # stop if no improvement withing 10 epochs
	model_checkpoint = cb.ModelCheckpoint(
		filename='best',
		monitor='VAL/loss', mode='min',  # save the minimum val_loss
	)
	
	# logger
	logger = TensorBoardLogger(
		save_dir=LOGS_CACHE,
		name=args.name,
		version=args.version,
		default_hp_metric=False # deactivate hp_metric on tensorboard visualization
	)
	
	trainer = pl.Trainer(
		max_epochs=args.epoch,
		accelerator=args.accelerator,  # auto use gpu
		enable_progress_bar=args.mode != M_EXP,  # not show progress bar when experimentation
		log_every_n_steps=1,
		default_root_dir=LOGS_CACHE,
		logger=logger,
		strategy=args.strategy,
		# fast_dev_run=True,
		callbacks=[early_stopping, model_checkpoint],
		# auto_scale_batch_size=True,
		#track_grad_norm=2,
		# detect_anomaly=True # deactivate on large scale experiemnt
	)
	
	trainer.fit(model, datamodule=dm)
	
	scores = trainer.test(
		ckpt_path='best',
	    datamodule=dm
	)
	
	# remove 'TEST/' from score dicts:
	scores = [
		{k.replace('TEST/', ''): v for k, v in s.items()} for s in scores
	]
	
	for idx, score in enumerate(scores):
		log.info(score)
		logger.log_metrics(score)
		
		score_path = path.join(logger.log_dir, f'score{"" if idx == 0 else "_" + str(idx)}.json')
		
		with open(score_path, 'w') as fp:
			json.dump(score, fp, indent='\t')
			log.info(f'Score is saved at {score_path}')
	
	print('Done')
	