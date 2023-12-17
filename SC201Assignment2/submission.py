#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # FeatureVector = {}
    # for word in x.split():
    #     if word in FeatureVector:
    #         FeatureVector[word] += 1
    #     else:
    #         FeatureVector[word] = 1
    # return FeatureVector
    featureVector = defaultdict(int)

    for word in x.split():
        featureVector[word] += 1
    # for word in featureVector:
    #     featureVector[word] /= len(x.split())
    return dict(featureVector)


############################################################
# Milestone 4: Sentiment Classification


def learnPredictor(
    trainExamples: List[Tuple[Any, int]],
    validationExamples: List[Tuple[Any, int]],
    featureExtractor: Callable[[str], FeatureVector],
    numEpochs: int,
    alpha: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the
    identity function may be used as the featureExtractor function during testing.
    """
    weights = defaultdict(float)  # Initialize weights

    def predictor(x):
        featureVector = featureExtractor(x)
        score = dotProduct(featureVector, weights)
        return 1 if score >= 0 else -1

    for epoch in range(numEpochs):
        cost = 0
        for x, y in trainExamples:
            y = max(y, 0)
            # phi(x) = featureExtractor(x)
            featureVector = featureExtractor(x)
            k = dotProduct(featureVector, weights)
            h = sigmoid(k)
            l = -(y * math.log(h) + (1 - y) * math.log(1 - h))
            cost += l
            for feature in featureVector:
                weights[feature] -= alpha * (h - y) * featureVector[feature]
            # gradient = {
            #     feature: (h - y) * value for feature, value in featureVector.items()
            # }
            # increment(weights, -alpha, gradient)
        # cost /= len(trainExamples)
        trainError = evaluatePredictor(trainExamples, predictor)
        validationError = evaluatePredictor(validationExamples, predictor)
        print(f"Training Error: ({epoch} epoch): {trainError}")
        print(f"Validation Error: ({epoch} epoch): {validationError}")

    return dict(weights)


############################################################
# Milestone 5a: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        phi = {}  # feature vector for x
        selected_features = random.sample(
            weights.keys(), k=random.randint(1, len(weights))
        )
        for feature in selected_features:
            phi[feature] = random.randint(1, 10)

        dot_product = sum(phi[feature] * weights[feature] for feature in phi)
        y = 1 if dot_product >= 0 else -1

        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> FeatureVector:
        x = x.replace(" ", "")
        featureVector = defaultdict(int)

        for i in range(len(x) - n + 1):
            ngram = x[i : i + n]
            featureVector[ngram] += 1
        return featureVector

    return extract


############################################################
# Problem 3f:
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )
