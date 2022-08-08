import os
import pickle
import sys
import warnings
import spacy
from argparse import ArgumentParser
from os import path
from typing import Union
import json

from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vectors import pretrained_aliases as pretrained, GloVe
import torchtext.transforms as T
import torchmetrics as m

from data.hotpot_qa.dataset import download_format_dataset, HotpotNLIDataset
from data.transforms import GoldLabelTransform, LemmaLowerTokenizerTransform, HighlightTransform, HeuristicTransform
from modules import metrics, env
from modules.logger import init_logging, log
from model.pair_lstm_attention import SigmoidPairLstmAttention

INF = 1e30 # Infinity

def loss_heuristic(heuris_loss_fn, attention, heuristic, mask):
	
	assert attention.shape == heuristic.shape, f'Dimension mismatch: (attention) \n{attention.shape} vs (heuristic) {heuristic.shape}'
	assert mask.dtype == torch.bool, f'mask must be boolean'
	if attention is None:
		return 0
	
	attention = attention.masked_fill(mask, -INF)
	attention = torch.log_softmax(attention, dim=1)

	return heuris_loss_fn(attention, heuristic)
	

def rescale(attention: torch.Tensor, mask: torch.Tensor):
	v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
	v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
	v_min[v_min == v_max] = 0.
	rescale_attention = (attention - v_min)/(v_max - v_min)
	rescale_attention[mask] = 0.
	return rescale_attention

class LitModel(pl.LightningModule):
	
	def __init__(self, cache_path, mode, vocab, lamb=0., pretrained_vectors: Union[str, torch.tensor]=None, **kwargs):
		super(LitModel, self).__init__()
		
		# log hyperparameters into hparams.yaml
		self.save_hyperparameters('lamb', ignore=['vocab'])
		
		if pretrained_vectors is not None and isinstance(pretrained_vectors, str):
			vector_path = path.join(cache_path, '.vector_cache')
			vectors = pretrained[pretrained_vectors](cache=vector_path)
			pretrained_vectors = [vectors[token] for token in vocab.get_itos()]
			pretrained_vectors = torch.stack(pretrained_vectors)
		
		self.model = SigmoidPairLstmAttention(pretrained_embedding=pretrained_vectors, d_embedding=300, **kwargs)
		self.loss_fn = nn.CrossEntropyLoss()
		self.heuris_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)
		self.vocab = vocab
		self.lamb=lamb
		self._mode = mode
		
		template_y_metrics = m.MetricCollection({
			'accuracy': m.Accuracy(num_classes=3),
			'f1': m.F1Score(num_classes=3)
		})
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			template_attention_metrics = m.MetricCollection({
				'attention_auc_roc': m.AUROC(average='micro'),
				'attention_recall': m.Recall(),
				'attention_specificity': m.Specificity(),
				'attention_jaccard': m.JaccardIndex(num_classes=2),
			})
		
		PHASES = ['train_', 'val_', 'test_']
		self.y_metrics = nn.ModuleDict({
			phase: template_y_metrics.clone() for phase in PHASES
		})
		self.attention_metrics = nn.ModuleDict({
			phase: template_attention_metrics.clone() for phase in PHASES
		})
		self.entropy_metric = nn.ModuleDict({
			phase: m.MeanMetric() for phase in PHASES
		})
		self.reg_term_metric = nn.ModuleDict({
			phase: m.MeanMetric() for phase in PHASES
		})
		
	def forward(self, x):
		return self.model((x['premise'], x['hypothesis'], x['padding_mask_premise'], x['padding_mask_hypothesis']))

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters())
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
		return [optimizer], [lr_scheduler]
	
	def training_step(self, batch, batch_idx, val=False):
		
		x, y_true = batch
		padding_mask = [x['padding_mask_premise'], x['padding_mask_hypothesis']]
		highlights = (x['highlight_premise'], x['highlight_hypothesis'])
		heuristics = (x['heuristic_premise'], x['heuristic_hypothesis'])
		
		y_hat, attentions = self(x)
		loss = self.loss_fn(y_hat, y_true)
		
		reg_term = loss_heuristic(self.heuris_loss_fn, attentions[0], heuristics[0], padding_mask[0])
		reg_term += loss_heuristic(self.heuris_loss_fn, attentions[1], heuristics[1], padding_mask[1])
	
		loss += self.lamb * reg_term
		# loss = regularize_heuristic_attention(loss, self.lamb, reg_loss_fn=self.reg_loss_fn, attention=attentions, heuristic=heuristics)
		
		entropies = [metrics.entropy(a.detach(), m, normalize=True, average='micro') for a, m in zip(attentions, padding_mask)]
		entropies = sum(entropies) / len(entropies)
		
		flat_a_hat, flat_a_true = self.flatten_attention(
			a_hat=attentions,
			a_true=highlights,
			y_true=y_true,
			pad_mask=padding_mask,
			normalize='softmax_rescale')
		
		if flat_a_hat is not None: flat_a_hat = flat_a_hat.detach()
		return {'loss': loss, 'y_hat': y_hat, 'y_true': y_true, 'a_hat': flat_a_hat, 'a_true': flat_a_true, 'entropy': entropies, 'reg_term': reg_term.detach()}
	
	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx, True)
	
	def test_step(self, batch, batch_idx):
		x, y_true = batch
		padding_mask = [x['padding_mask_premise'], x['padding_mask_hypothesis']]
		highlights = (x['highlight_premise'], x['highlight_hypothesis'])
		heuristics = (x['heuristic_premise'], x['heuristic_hypothesis'])
		
		y_hat, attentions = self(x)
		
		reg_term = loss_heuristic(self.heuris_loss_fn, attentions[0], heuristics[0], padding_mask[0])
		reg_term += loss_heuristic(self.heuris_loss_fn, attentions[1], heuristics[1], padding_mask[1])
		
		flat_a_hat, flat_a_true = self.flatten_attention(
			a_hat=attentions,
			a_true=highlights,
			y_hat=y_hat,
			y_true=y_true,
			pad_mask=padding_mask,
			normalize='softmax_rescale')

		entropies = [metrics.entropy(a.detach(), m, normalize=True, average='micro') for a, m in zip(attentions, padding_mask)]
		entropies = sum(entropies) / len(entropies)
		
		return {'y_hat': y_hat, 'y_true': y_true, 'a_hat': flat_a_hat.detach(), 'a_true': flat_a_true, 'entropy': entropies, 'reg_term': reg_term}
	
	def step_end(self, outputs, stage: str = 'test_'):
		
		# log attentions metrics
		a_hat, a_true = outputs['a_hat'], outputs['a_true']
		metric_a = None
		if a_hat is not None and a_hat.size(0) > 0:
			metric_a = self.attention_metrics[stage](a_hat, a_true)
			metric_a['attn_entropy'] = self.entropy_metric[stage](outputs['entropy'])
			metric_a['reg_term'] = self.reg_term_metric[stage](outputs['reg_term'])
			metric_a = {f'{stage}/{k}': v.item() for k, v in metric_a.items()}  # put metrics within same stage under the same folder
			self.log_dict(metric_a, prog_bar=True)
		
		# log for classification metrics
		metric_y = self.y_metrics[stage](outputs['y_hat'], outputs['y_true'])
		metric_y = {f'{stage}/{k}': v for k, v in metric_y.items()}  # put metrics within same stage under the same folder
		self.log_dict(metric_y, prog_bar=True)
		
		if stage != 'test_':
			# if not in test stage, log for loss metrics
			self.log(f'{stage}/loss', outputs['loss'], prog_bar=True)
			
		else:
			# Log hyperparameters metrics if this is test
			if metric_a is not None:
				metric_a = { f'hp/{k.split("/")[-1]}': v for k, v in metric_a.items()}
				self.log_dict(metric_a, on_epoch=True, prog_bar=False)
			metric_y = {f'hp/{k.split("/")[-1]}': v for k, v in metric_y.items()}
			self.log_dict(metric_y, on_epoch=True, prog_bar=False)
	
	def training_step_end(self, outputs):
		return self.step_end(outputs, stage='train_')
	
	def validation_step_end(self, outputs):
		return self.step_end(outputs, stage='val_')
	
	def test_step_end(self, outputs):
		return self.step_end(outputs, stage='test_')
	
	def flatten_attention(self, a_hat, a_true, y_true, pad_mask, y_hat=None, normalize:str =None):
		"""
		Turns tuples of attentions filtered
		Args:
		 a_hat ():
		 a_true ():
		 y_true ():
		 pad_mask ():
		 y_hat ():
		 normalize (str): softmax, softmax_rescale, sigmoid

		Returns:

		"""
		if torch.all(y_true == 0): return None, None
		condition = y_true > 0
		if y_hat is not None:
			condition = torch.logical_and(condition, torch.argmax(y_hat, dim=1) == y_true)
		
		# filter by labels
		a_hat = [a[condition] for a in a_hat]
		a_true = [a[condition] for a in a_true]
		pad_mask = [m[condition] for m in pad_mask]
		
		# Ir normalize specify:
		if normalize is not None:
			if normalize == 'sigmoid':
				a_hat = [torch.sigmoid(a) for a in a_hat]
			if 'softmax' in normalize:
				a_hat = [torch.softmax(a + m.float() * -INF, dim=1) for a, m in zip(a_hat, pad_mask)]
			if 'rescale' in normalize:
				a_hat = [rescale(a, m) for a, m in zip(a_hat, pad_mask)]
		
		flat_attention = [a_[~pad_] for a_, pad_ in zip(a_hat, pad_mask)]
		flat_highlight = [h_[~pad_] for h_, pad_ in zip(a_true, pad_mask)]
		
		flat_attention = torch.cat(flat_attention)
		flat_highlight = torch.cat(flat_highlight)
		
		return flat_attention, flat_highlight
	
	def on_train_start(self):
		init_hp_metrics = {f'hp/{k}': 0 for k in self.y_metrics['test_']}
		init_hp_metrics.update({f'hp/{k}': 0 for k in self.attention_metrics['test_']})
		init_hp_metrics.update({f'hp/attn_entropy': 0})
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
			except RuntimeError:
				pass
			metric.update({
				'entropy': self.entropy_metric[stage].compute(),
				'reg_term': self.reg_term_metric[stage].compute()
			})
			metric = {k: round(v.item(), 4) for k, v in metric.items()}
			log.info(f'Epoch {self.current_epoch} {stage}:{metric}')
			
	def on_train_epoch_end(self):
		return self.epoch_end('train_')
		
	def on_validation_epoch_end(self):
		return self.epoch_end('val_')
		
	def __str__(self):
		return str(self.model)


class LitData(pl.LightningDataModule):
	
	def __init__(self, cache_path, batch_size=8, num_workers=0, spacy_model=None, n_data=-1):
		super().__init__()
		self.cache_path = cache_path
		self.batch_size = batch_size
		self.tokenizer = LemmaLowerTokenizerTransform(spacy_model=spacy_model if spacy_model is not None else spacy.load('en_core_web_sm'))
		self.vocab = None
		self.dataset = {'train': None, 'val': None, 'test': None}
		self.n_data = n_data
		self.num_workers = num_workers
		#self.text_transform = None
		#self.highlight_transform = None
		#self.class_transform = None
		#self.heuristic_transform = None
	
	def prepare_data(self):
		# called only on 1 GPU
		
		# download_dataset()
		dataset_path = HotpotNLIDataset.root(self.cache_path)
		vocab_path = path.join(dataset_path, f'vocab_{str(self.tokenizer)}.pt')
		
		for split in ['train', 'val', 'test']:
			download_format_dataset(dataset_path, split, n_data=self.n_data)
		
		# build_vocab()
		if not path.exists(vocab_path):
			
			# return a single list of tokens
			def tokenize_dataset(batch):
				tokens_batch = self.tokenizer(batch['premise'] + batch['hypothesis'])
				return [token for sentence in tokens_batch for token in sentence]
			train_set = HotpotNLIDataset(root=self.cache_path, split='train', n_data=self.n_data)
			
			# build vocab from train set
			dp = train_set.batch(self.batch_size).map(self.list2dict).map(tokenize_dataset)
			
			# Build vocabulary from iterator. We don't know yet how long does it take
			iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout, disable=env.disable_tqdm)
			if env.disable_tqdm: log.info(f'Building vocabulary')
			vocab = build_vocab_from_iterator(iter_tokens, specials=['[PAD]', '[UNK]'])
			vocab.set_default_index(vocab['[UNK]'])
			
			# Announce where we save the vocabulary
			torch.save(vocab, vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol to speed things up
			iter_tokens.set_postfix({'path': vocab_path})
			if env.disable_tqdm: log.info(f'Vocabulary is saved at {vocab_path}')
			iter_tokens.close()
			self.vocab = vocab
		else:
			log.info(f'Loading vocab at {vocab_path}')
			self.vocab = torch.load(vocab_path)
		
		# predefined processing mapper for setup
		self.text_transform = T.Sequential(
			self.tokenizer,
			T.VocabTransform(self.vocab),
			T.ToTensor(padding_value=self.vocab['[PAD]'])
		)
		
		self.fact_transform = T.Sequential(
			self.tokenizer,
			T.VocabTransform(self.vocab),
			T.ToTensor(padding_value=self.vocab['[PAD]'])
		)
		
		self.label_transform = T.Sequential(
			T.LabelToIndex(),
			T.ToTensor()
		)
		
		self.highlight_transform = T.Sequential(
			self.tokenizer,
			HighlightTransform(),
			T.ToTensor(padding_value=False)
		)
	
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			self.train_set = HotpotNLIDataset(split='train', **dataset_kwargs)
			self.val_set = HotpotNLIDataset(split='val', **dataset_kwargs)
		
		if stage == 'test' or stage is None:
			self.test_set = HotpotNLIDataset(split='test', **dataset_kwargs)
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate, num_workers=self.num_workers)
	
	def train_dataloader(self):
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
		
	def val_dataloader(self):
		return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	## ======= PRIVATE SECTIONS ======= ##
	def collate(self, batch):
		# prepare batch of data for dataloader
		batch = self.list2dict(batch)
		
		context_token = self.tokenizer(batch['context'])
		flatten_context = self.highlight_transform(batch['highlighted_facts'])
		
		x = {
			'question': self.text_transform(batch['question']),
			'context': self.text_transform(batch['context']),
			'support': self.label_transform(batch['support']),
			'facts' : self.fact_transform(facts=batch['facts'], context=batch['context'])
		}
		
		x['padding_mask_question'] = x['question'] == self.vocab['[PAD]']
		x['padding_mask_context'] = x['context'] == self.vocab['[PAD]']
		
		y = self.class_transform(batch['label'])
		return x, y
	
	def list2dict(self, batch):
		# convert list of dict to dict of list
		
		if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()} # handle case where no batch
		return {k: [row[k] for row in batch] for k in batch[0]}

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
	parser.add_argument('--data', '-d', type=str, help='Path to the root of dataset. Example: "-d $HOME/dataset/snli"')
	parser.add_argument('--cache', '-o', type=str, default=path.join(os.getcwd(),'_out'), help='Path to temporary directory to store output of training process')
	parser.add_argument('--n_data', '-n', type=int, default=-1, help='Maximum data number for train+val+test, -1 if full dataset. Default: -1')
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
	
	# Regularization
	parser.add_argument('--lambda', type=float, default=0.)
	
	params = vars(parser.parse_args())
	print('=== Parameters ===')
	print(json.dumps(params, indent=4))
	
	# If data not provided, automatically get from '<cache>/dataset'
	params['data'] = params.get('data', path.join(params['cache'], 'dataset'))
	params['mode'] = params['mode'].lower()
	params = {k: v for k, v in params.items() if v is not None}
	env.disable_tqdm = params.get('OAR_ID', None) is not None
	return params


# const for mode
M_EXP = 'exp'
M_DEV = 'dev'

if __name__ == '__main__':
	params = parse_argument()
	

	init_logging(cache_path=params['cache'], color=not params['disable_log_color'])
	
	dataset_cache = path.join(params['cache'], 'dataset')
	models_cache = path.join(params['cache'], 'models')
	
	dm = LitData(
		cache_path=dataset_cache,
		batch_size=params['batch_size'],
		num_workers=params['num_workers'],
		n_data=params['n_data']
	)
	
	# prepare data here before going to multiprocessing
	dm.prepare_data()
	model = LitModel(cache_path=models_cache,
	                 mode=params['mode'],
	                 vocab=dm.vocab,
	                 lamb=params['lambda'],
	                 pretrained_vectors=params['vectors'],
	                 )
	
	# call back
	early_stopping = cb.EarlyStopping('val_/loss', patience=5, verbose=params['mode'] != M_EXP, mode='min')  # stop if no improvement withing 10 epochs
	model_summary = cb.ModelSummary(max_depth=1)  # print the nested model summary
	model_checkpoint = cb.ModelCheckpoint(
		filename='best',
		monitor='val_loss', mode='min',  # save the minimum val_loss
	)
	
	# logger
	logger = TensorBoardLogger(
		save_dir=path.join(params['cache'], 'lightning_logs'),
		name=params.get('name', str(model)),
		version=params.get('version', None),
		default_hp_metric=False # deactivate hp_metric on tensorboard visualization
	)
	
	trainer = pl.Trainer(
		max_epochs=params['epoch'],
		accelerator=params['accelerator'],  # auto use gpu
		enable_progress_bar=params['mode'] != M_EXP,  # not show progress bar when experimentation
		log_every_n_steps=1,
		default_root_dir=params['cache'],
		logger=logger,
		strategy=params.get('strategy', None),
		# fast_dev_run=params['mode'] == M_DEV,
		callbacks=[early_stopping, model_summary],
		auto_scale_batch_size=True,
		detect_anomaly=True
	)
	
	dm.setup(stage='fit')
	trainer.fit(model, datamodule=dm)
	
	dm.setup(stage='test')
	performance = trainer.test(
		ckpt_path='best',
	    datamodule=dm
	)
	log.info(performance)
	logger.log_metrics(performance[0])
	
	print('Done')
	