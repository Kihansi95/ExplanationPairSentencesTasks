import json
import os
from os import path

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import pandas as pd
import pytorch_lightning as pl

from modules import log, recursive_list2dict

class ParquetPredictionWriter(BasePredictionWriter):
    
    def __init__(self,
                 output_dir,
                 dm: pl.LightningDataModule = None,
                 fname='inference',
                 **kwargs):
        # def __init__(self, output_dir, label_itos:dict=None, label_column:Union[str,list]=None, **kwargs):
        """Write prediction result in parquet. The writer is called only during prediction phrase

        Parameters
        ----------
        output_dir : str
            Where to store resulted file
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.fname = fname
        self.dm = dm
    
    def write_on_batch_end(
            self,
            trainer: pl.Trainer,
            model_module: pl.LightningModule,
            prediction,
            batch_indices,
            batch,
            batch_idx: int,
            dataloader_idx: int):
        
        # flatten dictionary
        prediction = pd.json_normalize(prediction).to_dict(orient='records')[0]
        batch = pd.json_normalize(batch).to_dict(orient='records')[0]
        
        # convert into list from tensor columns
        result = {**prediction, **batch}
        result = {k: v.tolist() if torch.is_tensor(v) else v for k, v in result.items()}
        
        df = pd.DataFrame(result, index=None if len(batch_indices) <= 0 else batch_indices)
        if self.dm is not None: df = self.dm.format_predict(df)
        os.makedirs(path.join(self.output_dir, 'batches'), exist_ok=True)
        df.to_parquet(path.join(self.output_dir, 'batches', f'{self.fname}_{batch_idx}_{trainer.global_rank}.parquet'))
    
    def assemble_batch(self):
        """Assemble batch predictions into single file

        Returns
        -------

        """
        
        if not path.exists(path.join(self.output_dir, 'batches')):
            return
        
        batch_folder = path.join(self.output_dir, 'batches')
        files = [path.join(batch_folder, f) for f in os.listdir(batch_folder) if '.parquet' in f]
        predictions = [pd.read_parquet(f) for f in files]
        df = pd.concat(predictions)
        
        inference_path = path.join(self.output_dir, f'{self.fname}.parquet')
        df.to_parquet(inference_path)
        log.info(f'Finished assembling inference files from {batch_folder}.')
        log.info(f'Inferences are stored in {inference_path}')
    
    def write_on_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            predictions, batch_indices):
        
        # predictions = [loader_0, loader_1, ...]
        # loader_x = [batch_0, batch_1, ...]
        # batch_x = [prediction_dict]
        predictions = [batch for loader_x in predictions for batch in loader_x]  # flatten batch from loaders
        predictions = recursive_list2dict(predictions)  # flatten batched dict / batched list
        predictions = pd.json_normalize(predictions).to_dict(orient='records')[0]  # flatten nested dictionary
        
        for k in predictions:
            if not isinstance(predictions[k], list):
                continue
            if torch.is_tensor(predictions[k][0]):
                predictions[k] = [p for p_batch in predictions[k] for p in
                                  p_batch.tolist()]  # flatten tensor and transform into list
        
        df = pd.DataFrame(predictions)
        
        if self.dm is not None:
            df = pd.concat([df, self.dm.test_set.data], axis=1)
            df = self.dm.format_predict(df)
        df.to_parquet(path.join(self.output_dir, f'{self.fname}.parquet'))