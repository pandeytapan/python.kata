# Example of a Pythonic card deck
from collections import namedtuple

# A named tuple can be used to create a tuple like class
Card = namedtuple('Card', ['rank', 'suit'])
