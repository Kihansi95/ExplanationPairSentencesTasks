from enum import Enum

# Experiment training mode
class Mode(str, Enum):
	EXP = 'exp'
	DEV = 'dev'

# Data configuration constant
class InputType(Enum):
	SINGLE = 1
	DUAL = 2
	
class SpecToken(str, Enum):
	PAD = '<pad>'
	UNK = '<unk>'

class Normalization(str, Enum):
	NONE = None
	STANDARD = 'std'
	LOG_STANDARD = 'log_std'
	SOFTMAX = 'softmax'
	LOG_SOFTMAX = 'log_softmax'

# Model configuration constant


