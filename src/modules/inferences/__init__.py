from .json_prediction_writer import JsonPredictionWriter
from .parquet_prediction_writer import ParquetPredictionWriter
from .html_prediction_writer import HtmlPredictionWriter
from ..const import Writer

__all__ = [
    'JsonPredictionWriter',
    'ParquetPredictionWriter',
    'HtmlPredictionWriter'
]


def get_prediction_writer(writer: Writer, *args, **kwargs):
    """Factory method to create prediction writer
    
    Parameters
    ----------
    writer : str|Writer
        Writer type
        
    Returns
    -------
        Prediction writer
    """
    
    if writer == Writer.JSON:
        return JsonPredictionWriter(*args, **kwargs)
    elif writer == Writer.PARQUET:
        return ParquetPredictionWriter(*args, **kwargs)
    elif writer == Writer.HTML:
        return HtmlPredictionWriter(*args, **kwargs)
    else:
        raise ValueError(f'Unknown writer type: {writer}')
    