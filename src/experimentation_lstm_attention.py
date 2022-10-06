import os

from argparse import ArgumentParser
import json

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import callbacks as cb

from data_module.esnli import ESNLIDM
from data_module.hatexplain import HateXPlainDM
from data_module.yelp_hat import *
from model_module import DualLSTMAttentionModule
from model_module.single_lstm_attention import SingleLSTMAttentionModule
from modules.const import InputType, Mode

from modules.logger import log, init_logging
from modules import env

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


def parse_argument(prog: str = __name__, description: str = 'Train LSTM-based attention') -> dict:
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
	parser.add_argument('--version', type=str, default='default_version', help='Experimentation version')
	
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
	
	params.mode = params.mode.lower()
	return params

if __name__ == '__main__':
	
	args = parse_argument()
	
	DATA_CACHE = path.join(args.cache, 'dataset')
	MODEL_CACHE = path.join(args.cache, 'models')
	LOGS_CACHE = path.join(args.cache, 'logs')
	
	# init logging
	if args.mode == Mode.EXP:
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
	elif args.data == 'esnli':
		dm = ESNLIDM(**dm_kwargs)
	else:
		log.error(f'Unrecognized dataset: {args.data}')
		exit(1)
		
	# prepare data here before going to multiprocessing
	dm.prepare_data()
	model_args = dict(
		cache_path=MODEL_CACHE,
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
	
	if dm.input_type == InputType.SINGLE:
		model = SingleLSTMAttentionModule(**model_args)
	elif dm.input_type == InputType.DUAL:
		model = DualLSTMAttentionModule(**model_args)
	else:
		msg = f'Unknown input type of dm {str(dm)}: {dm.input_type}'
		log.error(msg)
		raise ValueError(msg)
	
	# call back
	early_stopping = cb.EarlyStopping('VAL/loss', patience=5, verbose=args.mode != Mode.EXP, mode='min')  # stop if no improvement withing 10 epochs
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
		enable_progress_bar=args.mode != Mode.EXP,  # not show progress bar when experimentation
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
	scores = [ {k.replace('TEST/', ''): v for k, v in s.items()} for s in scores ]
	
	for idx, score in enumerate(scores):
		log.info(score)
		logger.log_metrics(score)
		
		score_path = path.join(logger.log_dir, f'score{"" if idx == 0 else "_" + str(idx)}.json')
		
		with open(score_path, 'w') as fp:
			json.dump(score, fp, indent='\t')
			log.info(f'Score is saved at {score_path}')
	
	print('Done')
	