import os
from util import interactivePrompt
from submission import extractWordFeatures

"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""


def main():
    """
    This function shows the sentiment of the input review
    by users on Console.
    """
    weights = openWeights()
    interactivePrompt(extractWordFeatures, weights)


def openWeights():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_path, "weights")
    weights = {}
    with open(file_path, "r") as file:
        content = file.read()
        for line in content.split("\n"):
            if line:
                key, value = line.split()
                weights[key] = float(value)
    return weights


if __name__ == "__main__":
    main()
