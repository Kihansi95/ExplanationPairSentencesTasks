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

# Model configuration constant

