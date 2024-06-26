import json
import os
from argparse import ArgumentParser
from datetime import timedelta
from os import path

import pytorch_lightning as pl
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from pytorch_lightning import callbacks as cb
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.states import RunningStage

from model_module.lstm.archival_lstm_module import ArchivalLstmModule
from modules import report_score
from modules.const import *
from modules.inferences import get_prediction_writer
from modules.logger import init_logging, log


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


def get_carbon_tracker(args, output_dir) -> EmissionsTracker:
	"""Get carbon tracker based on argument

	Parameters
	----------
	args : Any

	Returns
	-------
	tracker : EmissionsTracker
		Either online, offline tracker if we precise --carbon_tracker. None if this argument is absent.
	"""
	
	if args.track_carbon is None:
		return None
	
	if args.track_carbon == TrackCarbon.ONLINE:
		tracker = EmissionsTracker(
			project_name=f'{args.name}/{args.version}',
			output_dir=output_dir,
			log_level='critical'
		)
	elif args.track_carbon == TrackCarbon.OFFLINE:
		tracker = OfflineEmissionsTracker(
			project_name=f'{args.name}/{args.version}',
			output_dir=output_dir,
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
	parser.add_argument('--disable_log_color', action='store_true',
						help='Activate for console does not support coloring')
	parser.add_argument('--server_message', type=str, help='Get cluster ID to see error/output logs')
	
	# For reports
	parser.add_argument('--track_carbon', type=str, help='If precised will track down carbon')
	parser.add_argument('--track_grad_norm', type=int, default=-1, help='Log gradient norm in tensorboard log')
	parser.add_argument('--track_time', action='store_true', help='Enable time tracking')
	
	# Trainer params
	parser.add_argument('--cache', '-o', type=str, default=path.join(os.getcwd(), '..', '.cache'),
						help='Path to temporary directory to store output of training process')
	parser.add_argument('--mode', '-m', type=str, default='dev',
						help='Choose among [dev, exp]. "exp" will disable the progressbar')
	parser.add_argument('--num_workers', type=int, default=get_num_workers(),
						help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--accelerator', type=str, default='auto',
						help='Indicate whether we are in IGRIDA cluster mode. Default: Use all cpu cores.')
	parser.add_argument('--name', type=str, help='Experimentation name. If not given, use model name instead.')
	parser.add_argument('--version', type=str, default='default_version', help='Experimentation version')
	parser.add_argument('--epoch', '-e', type=int, default=1, help='Number training epoch. Default: 1')
	parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of data in batch. Default: 32')
	parser.add_argument('--strategy', '-s', type=str, help='')
	parser.add_argument('--fast_dev_run', action='store_true')
	parser.add_argument('--detect_anomaly', action='store_true')
	parser.add_argument('--devices', type=int)
	parser.add_argument('--num_nodes', type=int)
	
	# Model configuration
	parser.add_argument('--vectors', type=str,
						help='Pretrained vectors. See more in torchtext Vocab, example: fasttext.fr.300d')
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--d_embedding', type=int, default=300, help='Embedding dimension, will be needed if vector is not precised')
	parser.add_argument('--d_hidden_lstm', type=int, default=-1)
	parser.add_argument('--n_context', type=int, default=1)
	parser.add_argument('--concat_context', action='store_true')
	
	# Data configuration
	# parser.add_argument('--raw_path', type=str, default=path.join(os.getcwd(), '.cache', 'archival_nli.parquet'), help='Where to find the generated archival_nli.parquet file from notebook.')
	parser.add_argument('--n_data', '-n', type=int, default=-1, help='Maximum data number for train+val+test, -1 if full dataset. Default: -1')
	parser.add_argument('--data', '-d', choices=Data.list(), type=str, help=f'Choose dataset to train model, possible choice: archival_nli, xnli')
	parser.add_argument('--shuffle_off', action='store_true', help='Turn off shuffle in training cycle. Used for debug large dataset.')
	parser.add_argument('--data_version', type=str, help='data version')
	
	# Fit configuration
	parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint if there is')
	
	# Test configuration
	parser.add_argument('--test_path', type=str, help='Path to which model give output score')
	
	# Predict configuration
	parser.add_argument('--predict_path', type=str, help='Path to which model give predict')
	parser.add_argument('--writer', type=str, choices=Writer.list(), default=Writer.JSON.value,
						help='Writer to write prediction. Default: json')
	
	# Pipeline
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--predict', action='store_true')
	
	# Regularizer
	parser.add_argument('--lambda_entropy', type=float, default=0., help='multiplier for entropy')
	parser.add_argument('--lambda_heuristic', type=float, default=0., help='multiplier for heuristic loss')
	
	params = parser.parse_args()
	params.shuffle = not params.shuffle_off
	print('=== Parameters ===')
	print(json.dumps(vars(params), indent=4))
	
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
	LOGS_CACHE = path.join(args.cache, 'logs')
	
	# init logging
	if args.mode == Mode.EXP:
		init_logging(cache_path=LOGS_CACHE, color=False, experiment=args.name, version=args.version)
	else:
		init_logging(color=True)
	
	if args.server_message:
		log.debug('Server message:')
		log.info(f'{args.server_message}')
	
	log.info(f'experimentation_name = {args.name}')
	
	# Init data module
	dm_kwargs = dict(cache_path=DATA_CACHE,
					 batch_size=args.batch_size,
					 num_workers=args.num_workers,
					 n_data=args.n_data,
					 shuffle=args.shuffle,
					 )
	
	if args.data == Data.ARCHIVAL_NLI:
		from data_module.archival_module import ArchivalNLIDM
		
		dm_kwargs['version'] = args.data_version
		if args.predict_path is not None: dm_kwargs['predict_path'] = args.predict_path
		dm = ArchivalNLIDM(**dm_kwargs)
	elif args.data == Data.XNLI:
		from data_module.fr_xnli_module import FrXNLIDM
		
		dm = FrXNLIDM(**dm_kwargs)
	else:
		log.error(f'Unrecognized dataset: {args.data}')
		exit(1)
	
	# prepare data here before going to multiprocessing
	dm.prepare_data()
	model_args = dict(
		cache_path=args.cache,
		mode=args.mode,
		vocab=dm.vocab,
		lambda_entropy=args.lambda_entropy,
		lambda_heuristic=args.lambda_heuristic,
		pretrained_vectors=args.vectors,
		concat_context=args.concat_context,
		n_context=args.n_context,
		d_hidden_lstm=args.d_hidden_lstm,
		d_embedding=args.d_embedding,
		data=args.data,
		num_class=dm.num_class
	)
	
	if dm.input_type == InputType.DUAL:
		ModelModule = ArchivalLstmModule
	else:
		msg = f'Unknown input type of dm {str(dm)}: {dm.input_type}'
		log.error(msg)
		raise ValueError(msg)
	
	# logger
	logger = TensorBoardLogger(
		save_dir=LOGS_CACHE,
		name=args.name,
		version=args.version,
		default_hp_metric=False  # deactivate hp_metric on tensorboard visualization
	)
	
	LOG_DIR = logger.log_dir
	PREDICT_PATH = path.join(LOG_DIR, 'predictions')
	
	fname = 'inference'
	if args.predict_path is not None:
		fname = path.splitext(path.basename(args.predict_path))[0]
	
	# call back
	early_stopping = cb.EarlyStopping('VAL/loss', patience=3, verbose=args.mode != Mode.EXP,
									  mode='min')  # stop if no improvement withing 10 epochs
	model_checkpoint = cb.ModelCheckpoint(filename='best', monitor='VAL/loss', mode='min')  # save the minimum val_loss
	pred_writer = get_prediction_writer(writer=args.writer, dm=dm, output_dir=PREDICT_PATH,
										write_interval='batch')  # write output during epoch
	callbacks = [early_stopping, model_checkpoint, pred_writer]
	
	# Time tracking
	if args.track_time:
		timer = Timer(verbose=False)
		callbacks.append(timer)
	
	# Carbon tracking
	tracker = get_carbon_tracker(args, output_dir=LOG_DIR)
	
	trainer = pl.Trainer(
		max_epochs=args.epoch,
		accelerator=args.accelerator,  # auto use gpu
		enable_progress_bar=args.mode != Mode.EXP,  # not show progress bar when experimentation
		log_every_n_steps=1,
		default_root_dir=LOGS_CACHE,
		logger=logger,
		strategy=args.strategy,
		fast_dev_run=args.fast_dev_run,
		callbacks=callbacks,
		track_grad_norm=args.track_grad_norm,  # track_grad_norm=2 for debugging
		detect_anomaly=args.detect_anomaly,  # deactivate on large scale experiemnt
		benchmark=False,  # benchmark = False better time in NLP
		devices=args.devices,
		num_nodes=args.num_nodes
	)
	
	# Set up output fpath
	ckpt_path = path.join(LOG_DIR, 'checkpoints', 'best.ckpt')
	hparams_path = path.join(LOG_DIR, 'hparams.yaml')
	
	###########################################################################
	## Train : either train model from scratch or by resuming previous training
	###########################################################################
	
	if args.train:
		log.info(f'Training...')
		model = ModelModule(**model_args)
		
		if args.resume:
			try:
				log.debug(f'Model resume training from {ckpt_path}')
				trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
			except FileNotFoundError:
				log.error(f'{ckpt_path} does not exist, run a new model instead')
				trainer.fit(model, datamodule=dm)
		else:
			log.debug(f'Train new model from scratch')
			trainer.fit(model, datamodule=dm)
	
	else:
		log.debug(f'Loading trained model')
		log.debug(f'\tCheckpoint file : {ckpt_path}')
		log.debug(f'\tHyperparameters file: {hparams_path}')
		model = ModelModule.load_from_checkpoint(
			checkpoint_path=ckpt_path,
			hparams_file=hparams_path,
			**model_args
		)
	
	##################################
	## Test: report score on test set
	##################################
	
	if args.train or args.test:
		log.info(f'Testing...')
		scores = trainer.test(model=model, datamodule=dm)
		report_score(scores, logger, args.test_path)
	
	##################################################
	## Predict: predict on test set or custom dataset
	##################################################
	if args.predict:
		log.info(f'Predicting...')
		log.info(f'Prediction saved in {PREDICT_PATH}')
		os.makedirs(PREDICT_PATH, exist_ok=True)
		trainer.predict(
			model=model,
			datamodule=dm,
			return_predictions=False
		)
		
		pred_writer.assemble_batch()
	
	#############################################
	## Report : Additional information for papers
	#############################################
	log.info(f'Reporting...')
	
	if args.track_time:
		for stage in [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING]:
			duration = timedelta(seconds=int(timer.time_elapsed(stage)))
			log.info(f'Duration of stage {stage} : {duration}')
	
	if args.track_carbon:
		emission = tracker.stop()
		emission_str = f'Total emission in experiment trial: {emission} kgs'
		log.info(emission_str)
	
	log.info(f'Finished')
