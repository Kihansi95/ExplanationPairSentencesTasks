import os
from os import path
import json
from argparse import ArgumentParser
from codecarbon import EmissionsTracker

from modules import report_score
from modules.const import InputType, Mode
from modules.logger import init_logging
from modules.logger import log

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import callbacks as cb

from model_module import DualLSTMAttentionModule, SingleLSTMAttentionModule

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
	parser.add_argument('--OAR_ID', type=int, help='Get cluster ID to see error/output logs')
	
	# Trainer params
	parser.add_argument('--cache', '-o', type=str, default=path.join(os.getcwd(), '..', '.cache'), help='Path to temporary directory to store output of training process')
	parser.add_argument('--mode', '-m', type=str, default='dev', help='Choose among [dev, exp]. "exp" will disable the progressbar')
	parser.add_argument('--num_workers', type=int, default=get_num_workers(), help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--accelerator', type=str, default='auto', help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--name', type=str, help='Experimentation name. If not given, use model name instead.')
	parser.add_argument('--version', type=str, default='default_version', help='Experimentation version')
	parser.add_argument('--epoch', '-e', type=int, default=1, help='Number training epoch. Default: 1')
	parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of data in batch. Default: 32')
	parser.add_argument('--strategy', '-s', type=str, help='')
	parser.add_argument('--fast_dev_run', action='store_true')
	parser.add_argument('--detect_anomaly', action='store_true')
	parser.add_argument('--track_grad_norm', type=int, default=-1)
	
	# Model configuration
	parser.add_argument('--vectors', type=str, help='Pretrained vectors. See more in torchtext Vocab, example: glove.840B.300d')
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--d_embedding', type=int, default=300, help='Embedding dimension, will be needed if vector is not precised')
	parser.add_argument('--d_hidden_lstm', type=int, default=-1)
	parser.add_argument('--n_lstm', type=int, default=1)
	
	# Data configuration
	parser.add_argument('--n_data', '-n', type=int, default=-1, help='Maximum data number for train+val+test, -1 if full dataset. Default: -1')
	parser.add_argument('--data', '-d', type=str, help='Choose dataset to train model')
	
	# Fit configuration
	parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint if there is')
	
	# Predict configuration
	parser.add_argument('--test_path', type=str, help='Path to which model give output score')
	
	# Pipeline
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--predict', action='store_true')
	parser.add_argument('--morpho_filter', action='store_true')
	
	
	# Regularizer
	parser.add_argument('--lambda_entropy', type=float, default=0., help='multiplier for entropy')
	parser.add_argument('--lambda_supervise', type=float, default=0., help='multiplier for supervise loss')
	parser.add_argument('--lambda_heuristic', type=float, default=0., help='multiplier for heuristic loss')
	parser.add_argument('--lambda_lagrange', type=float, default=0., help='multiplier for relaxation of Lagrange (Supervision by entropy)')
	
	params = parser.parse_args()
	print('=== Parameters ===')
	print(json.dumps(vars(params), indent=4))
	
	params.mode = params.mode.lower()
	return params

if __name__ == '__main__':
	
	args = parse_argument()
	
	if not (args.train or args.test or args.predict):
		args.train = True
	
	DATA_CACHE = path.join(args.cache, 'dataset')
	MODEL_CACHE = path.join(args.cache, 'models')
	LOGS_CACHE = path.join(args.cache, 'logs')
	
	# init logging
	if args.mode == Mode.EXP:
		init_logging(cache_path=LOGS_CACHE, color=False, experiment=args.name, version=args.version)
	else:
		init_logging(color=True)
		
	if args.OAR_ID:
		log.info(f'OAR_ID = {args.OAR_ID}')
		
	log.info(f'experimentation_name = {args.name}')
	if args.resume:
		log.warn('Resume from previous training')
	
	# Carbon tracking
	dm_kwargs = dict(cache_path=DATA_CACHE,
	                 batch_size=args.batch_size,
	                 num_workers=args.num_workers,
	                 n_data=args.n_data)
	
	if args.data == 'hatexplain':
		from data_module.hatexplain import HateXPlainDM
		dm = HateXPlainDM(**dm_kwargs)
	elif args.data == 'yelphat':
		from data_module.yelp_hat import *
		dm = YelpHatDM(**dm_kwargs)
	elif args.data == 'yelphat50':
		from data_module.yelp_hat import *
		dm = YelpHat50DM(**dm_kwargs)
	elif args.data == 'yelphat100':
		from data_module.yelp_hat import *
		dm = YelpHat100DM(**dm_kwargs)
	elif args.data == 'yelphat200':
		from data_module.yelp_hat import *
		dm = YelpHat200DM(**dm_kwargs)
	elif args.data == 'esnli':
		from data_module.esnli import ESNLIDM
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
		lambda_heuristic=args.lambda_heuristic,
		pretrained_vectors=args.vectors,
		n_lstm=args.n_lstm,
		d_hidden_lstm=args.d_hidden_lstm,
		d_embedding=args.d_embedding,
		data=args.data,
		num_class=dm.num_class
	)
	
	if dm.input_type == InputType.SINGLE:
		ModelModule = SingleLSTMAttentionModule
	elif dm.input_type == InputType.DUAL:
		ModelModule = DualLSTMAttentionModule
	else:
		msg = f'Unknown input type of dm {str(dm)}: {dm.input_type}'
		log.error(msg)
		raise ValueError(msg)
	
	# call back
	early_stopping = cb.EarlyStopping('VAL/loss', patience=3, verbose=args.mode != Mode.EXP, mode='min')  # stop if no improvement withing 10 epochs
	model_checkpoint = cb.ModelCheckpoint(filename='best', monitor='VAL/loss', mode='min')  # save the minimum val_loss
	
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
		fast_dev_run=args.fast_dev_run,
		callbacks=[early_stopping, model_checkpoint],
		track_grad_norm=args.track_grad_norm, # track_grad_norm=2 for debugging
		detect_anomaly=args.detect_anomaly, # deactivate on large scale experiemnt
		benchmark=False,    # benchmark = False better time in NLP
	)
	
	# Set up output path
	ckpt_path = path.join(logger.log_dir, 'checkpoints', 'best.ckpt')
	hparams_path = path.join(logger.log_dir, 'hparams.yaml')
	
	if args.train:
		model = ModelModule(**model_args)
		trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path if args.resume else None)
	
	else:
		model = ModelModule.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, **model_args)
		
	# Carbon tracking
	if args.mode == Mode.EXP:
		tracker = EmissionsTracker(
			project_name=f'{args.name}/{args.version}',
			output_dir=logger.log_dir,
			log_level='critical'
		)
		tracker.start()
	
	if args.train or args.test:
		scores = trainer.test(model=model, datamodule=dm)
		report_score(scores, logger, args.test_path)
	
	if args.predict:
		# TODO complete: make a new parquet file to save predictions along dataset
		predictions = trainer.predict(
			model=model,
			datamodule=dm
		)
		
		log.warn('Prediction incompleted')
		predict_path = path.join(logger.log_dir, f'predict.txt')
		
		with open(predict_path, 'w') as fp:
			fp.write(predictions)
			log.info(f'Predictions are saved at {predict_path}')
	
	if args.mode == Mode.EXP:
		emission = tracker.stop()
		emission_str = f'Total emission in experiment trial: {emission} kgs'
		log.info(emission_str)
			
	if args.morpho_filter:
		raise Exception('Not yet implemented')
		