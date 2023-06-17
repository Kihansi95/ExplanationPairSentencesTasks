import os
from os import path
import json
from argparse import ArgumentParser
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from model_module import DualEkeyLqueryModule, SingleEkeyLqueryModule
from modules import report_score
from modules.const import InputType, Mode, TrackCarbon
from modules.inferences.prediction_writer import ParquetPredictionWriter
from modules.logger import init_logging
from modules.logger import log

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import callbacks as cb

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


def get_carbon_tracker(args) -> EmissionsTracker:
	if args.track_carbon is None:
		return None
	
	if args.track_carbon == TrackCarbon.ONLINE:
		tracker = EmissionsTracker(
			project_name=f'{args.name}/{args.version}',
			output_dir=LOG_DIR,
			log_level='critical'
		)
	elif args.track_carbon == TrackCarbon.OFFLINE:
		tracker = OfflineEmissionsTracker(
			project_name=f'{args.name}/{args.version}',
			output_dir=LOG_DIR,
			log_level='critical',
			country_iso_code='FRA'
		)
	
	tracker.start()
	return tracker

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
	parser.add_argument('--track_carbon', type=str, help='If precised will track down carbon')
	
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
	parser.add_argument('--devices', type=int, help='Precise number of GPU available if the environment allows')
	parser.add_argument('--num_nodes', type=int, help='Precise number of node if the environment allows')
	
	# Model configuration
	parser.add_argument('--vectors', type=str, help='Pretrained vectors. See more in torchtext Vocab, example: glove.840B.300d')
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--d_embedding', type=int, default=300, help='Embedding dimension, will be needed if vector is not precised')
	parser.add_argument('--n_context', type=int, default=1)
	parser.add_argument('--concat_context', action='store_true')
	
	# Data configuration
	parser.add_argument('--n_data', '-n', type=int, default=-1, help='Maximum data number for train+val+test, -1 if full dataset. Default: -1')
	parser.add_argument('--data', '-d', type=str, help='Choose dataset to train model')
	parser.add_argument('--shuffle_off', action='store_true', help='Turn off shuffle in training cycle. Used for debug large dataset.')
	
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
	params.shuffle = not params.shuffle_off
	print('=== Parameters ===')
	print(json.dumps(vars(params), indent=4))
	
	# Customized arguments
	params.mode = params.mode.lower()
	if params.strategy == 'ddp_find_off':
		from pytorch_lightning.strategies import DDPStrategy
		params.strategy = DDPStrategy(find_unused_parameters=False)
	elif params.strategy == 'ddp_spawn_find_off':
		from pytorch_lightning.strategies import DDPSpawnStrategy
		params.strategy = DDPSpawnStrategy(find_unused_parameters=False)
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
	
	dm_kwargs = dict(cache_path=DATA_CACHE,
	                 batch_size=args.batch_size,
	                 num_workers=args.num_workers,
	                 n_data=args.n_data,
	                 shuffle=args.shuffle)
	
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
		n_context=args.n_context,
		concat_context=args.concat_context,
		d_embedding=args.d_embedding,
		data=args.data,
		num_class=dm.num_class
	)
	
	if dm.input_type == InputType.SINGLE:
		ModelModule = SingleEkeyLqueryModule
	elif dm.input_type == InputType.DUAL:
		ModelModule = DualEkeyLqueryModule
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
	
	LOG_DIR = logger.log_dir
	
	pred_writer = ParquetPredictionWriter(output_dir=path.join(LOG_DIR, 'predictions'), write_interval='batch')
	
	trainer = pl.Trainer(
		max_epochs=args.epoch,
		accelerator=args.accelerator,  # auto use gpu
		enable_progress_bar=args.mode != Mode.EXP,  # not show progress bar when experimentation
		log_every_n_steps=1,
		default_root_dir=LOGS_CACHE,
		logger=logger,
		strategy=args.strategy,
		fast_dev_run=args.fast_dev_run,
		callbacks=[early_stopping, model_checkpoint, pred_writer],
		track_grad_norm=args.track_grad_norm, # track_grad_norm=2 for debugging
		detect_anomaly=args.detect_anomaly, # deactivate on large scale experiemnt
		benchmark=False,    # benchmark = False better time in NLP
		devices=args.devices,
		num_nodes=args.num_nodes,
	)
	
	# Set up output fpath
	ckpt_path = path.join(LOG_DIR, 'checkpoints', 'best.ckpt')
	hparams_path = path.join(LOG_DIR, 'hparams.yaml')
	
	if args.train:
		model = ModelModule(**model_args)
		trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path if args.resume else None)
	
	else:
		model = ModelModule.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, **model_args)
		
	# Carbon tracking
	tracker = get_carbon_tracker(args)
	
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
		predict_path = path.join(LOG_DIR, f'predict.txt')
		
		with open(predict_path, 'w') as fp:
			fp.write(predictions)
			log.info(f'Predictions are saved at {predict_path}')
	
	if tracker is not None:
		emission = tracker.stop()
		emission_str = f'Total emission in experiment trial: {emission} kgs'
		log.info(emission_str)
	
		