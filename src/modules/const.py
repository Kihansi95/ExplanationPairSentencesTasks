from enum import Enum

class ExtendedEnum(Enum):

	@classmethod
	def list(cls):
		return list(map(lambda c: c.value, cls))

# Experiment training mode
class Mode(str, ExtendedEnum):
	EXP = 'exp'
	DEV = 'dev'

# Data configuration constant
class InputType(ExtendedEnum):
	SINGLE = 1
	DUAL = 2
	
class SpecToken(str, ExtendedEnum):
	PAD = '<pad>'
	UNK = '<unk>'
	CLS = '<cls>'

class Normalization(str, ExtendedEnum):
	NONE = None
	STANDARD = 'std'
	LOG_STANDARD = 'log_std'
	SOFTMAX = 'softmax'
	LOG_SOFTMAX = 'log_softmax'

# Model configuration constant
class ContextType(str, ExtendedEnum):
	LSTM='lstm'
	CNN='cnn'
	ATTENTION='attention'
