import json
import os
import sys
from functools import reduce
from os import path
from argparse import ArgumentParser

import pandas as pd
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from yaml import SafeLoader

from modules.logger import log, init_logging

_FIGURES = 'figures'
_TABLE = 'table'

def common_element(*list_):
	# Get common elemnent from list of list
	return list(reduce(lambda i, j: i & j, (set(x) for x in list_)))

def parse_argument(prog: str = __name__, description: str = 'Summarize results produced by experiments') -> dict:
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
	
	# Summary arguments
	parser.add_argument('--summary', action='store_true', help='Force to regenerate summary cache')
	parser.add_argument('--experiment', type=str, nargs='+', help='Experiment name in log_dir. '
	                                                                'If not given, try to summary all experiments'
	                                                                'If more then 2 values, try to generate appropriate scaling '
	                                                                'between different experiments')
	
	# Figure arguments
	parser.add_argument('--figure', action='store_true', help='Regenerate figures')
	parser.add_argument('--ylim', type=float, nargs=2, help='Scale for y-axis in figure')
	
	# Table arguments
	parser.add_argument('--table', action='store_true', help='Force to regenerate summary table')
	parser.add_argument('--round', type=int, help='Rounding table in latex')
	
	# General arguments
	parser.add_argument('--log_dir', '-l', type=str, required=True, help='Path where logs is saved, contains different '
	                                                                     'experimentations set up.')
	parser.add_argument('--out_dir', '-o', type=str, required=True, help='Output directory')
	return parser.parse_args()

if __name__ == '__main__':
	
	args = parse_argument()
	init_logging(color=True)
	
	log_dir = args.log_dir
	experiments = None
	
	if isinstance(args.experiment, list) and len(args.experiment) > 0:
		for e in args.experiment:
			if not path.exists(path.join(log_dir, e)): raise ValueError(f'Experiment {e} does not exist in {log_dir}')
		experiments = args.experiment
	else:
		experiments = [e for e in os.listdir(log_dir) if '.DS_Store' not in e]
	
	# individual summary results
	exp_summary = list()
	exp_config = list()
	
	for experiment in experiments:

		experiment_path = path.join(log_dir, experiment)
		out_path = path.join(args.out_dir, 'summary', experiment)
		parquet_path = path.join(out_path, 'summary.parquet') # save metrics results and hparams
		config_path = path.join(out_path, 'config.json') # save hparams and metric names
		
		results = list()
		runs = list(os.listdir(experiment_path))
		runs = [r for r in runs if '.DS_Store' not in r]
		runs = [r for r in runs
		        if path.exists(path.join(experiment_path, r, 'hparams.yaml'))
		        and path.exists(path.join(experiment_path, r, 'score.json'))] # get runs that has hparams and score
		
		if len(runs) == 0:
			log.warn(f'Skip {experiment}: No run found')
			continue
		
		if args.summary or not path.exists(parquet_path):
		
			for run in tqdm(runs, total=len(runs), desc=f'Summarize {experiment}', file=sys.stdout):
				
				with open(path.join(experiment_path, run, 'hparams.yaml')) as f:
					hparam = yaml.load(f, Loader=SafeLoader)
				
				with open(path.join(experiment_path, run, 'score.json')) as f:
					score = json.load(f)
					score = {k.replace('TEST/', ''): v for k, v in score.items()}
				
				score_row = {**hparam, **score}
				results.append(score_row)
			
			summary = pd.DataFrame(results, index=runs)
			config = {
				'hparams': list(hparam.keys()),
				'metrics': list(score.keys())
			}
			
			# Save to summary
			## Cache dataframe
			os.makedirs(out_path)
			summary.to_parquet(parquet_path)
			with open(config_path, "w") as f:
				json.dump(config, f)
			
		else:
			summary = pd.read_parquet(parquet_path)
			#log.debug(f'config_path={config_path}')
			with open(config_path, 'r') as f:
				config = json.load(f)
				
		exp_summary.append(summary)
		exp_config.append(config)
		
		if not args.figure and not args.table: continue
		
		figure_path = path.join(out_path, _FIGURES)
		table_path = path.join(out_path, _TABLE)
		
		## Save figure
		objectives = [p for p in config['hparams'] if 'lambda' in p and summary[p].nunique() > 1] # ["lambda_entropy", "lambda_supervise"]
		groups = [p for p in config['hparams'] if 'lambda' not in p and summary[p].nunique() > 1] # ["n_lstm", "n_attention", "n_cnn"]
		sns.set(font_scale=2)
		
		os.makedirs(figure_path, exist_ok=True)
		os.makedirs(table_path, exist_ok=True)
		
		for m in config['metrics']:
			for lambda_value in objectives:
				for g in groups:
					
					if args.figure:
						fig_path = path.join(figure_path, f'x={lambda_value}_y={m}_color={g}.png')
						fig = plt.figure(figsize=(20, 15), clear=True)
						graphic = sns.pointplot(data=summary, x=lambda_value, y=m, hue=g, errwidth=2, capsize=0.1, dodge=True, palette='husl')
						if args.ylim: graphic.set_ylim(tuple(args.ylim))
						plt.savefig(fig_path, bbox_inches="tight")
						plt.close(fig)
			
					if args.table:
						mean = summary.groupby([g, lambda_value]).mean()
						std = summary.groupby([g, lambda_value]).std()
						
						if args.round:
							mean = mean.round(args.round)
							std = std.round(args.round)
						
						df_quantify = mean.astype(str) + u"\u00B1" + std.astype(str)
						latex_path = path.join(table_path, f'{e}_{g}_{lambda_value}.tex')
						
						with open(latex_path, 'w') as f:
							f.write(df_quantify.style.to_latex())
						
						html_path = path.join(table_path, f'{e}_{g}_{lambda_value}.html')
						with open(html_path, "w") as f:
							f.write(df_quantify.to_html(justify='center'))
		
		
	# sub summary results
	if args.experiment is not None and len(args.experiment) > 1:
		exp_str = '.'.join(experiments)
		compare_path = path.join(args.out_dir, 'summary', f'comparing.{exp_str}')
		os.makedirs(compare_path,exist_ok=True)
		log.debug(f'Comparing figure in {compare_path}')
		
		hparams_names = [c['hparams'] for c in exp_config]
		hparam = common_element(*hparams_names)
		metrics_names = [c['metrics'] for c in exp_config]
		metrics = common_element(*metrics_names)
		
		# Once found common ylim, plot subfigure by 2 columns
		n_exp = len(exp_summary)
		for m in metrics:
			
			all_y_min = [summary[m].min() for summary in exp_summary]
			all_y_max = [summary[m].max() for summary in exp_summary]
			y_min = min(all_y_min)
			y_max = max(all_y_max)
			margin = 0.1*(y_max-y_min)
			y_lim = (y_min - margin, y_max + margin)
			
			fig, axes = plt.subplots((n_exp // 2 + n_exp % 2), 2, figsize=(40, 20), sharey=True)
			for axe, summary in zip(axes, exp_summary):
				lambdas = [p for p in config['hparams'] if 'lambda' in p and summary[p].nunique() > 1]
				groups = [p for p in config['hparams'] if 'lambda' not in p and summary[p].nunique() > 1]
				graphic = sns.pointplot(ax=axe, data=summary, x=lambdas[0], y=m, hue=groups[0])
				# graphic.set_ylim(y_lim)
			
			fig_path = path.join(compare_path, f'y={m}.png')
			plt.savefig(fig_path, bbox_inches="tight")
			plt.close(fig)
	