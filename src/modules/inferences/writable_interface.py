import abc
from typing import Union


class WritableInterface(metaclass=abc.ABCMeta):
    """
    To be implemented in data modules to allow for proper writing tokens to a file.
    """
    
    @abc.abstractmethod
    def writing_tokens(self, datarow) -> Union[list, dict]:
        raise NotImplementedError
    