#!/usr/bin/env python3
# figures
# Script to generate figures for the PyCon 2017 Yellowbrick Poster.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Wed May 10 17:26:09 2017 -0400
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: figures.py [] benjamin@bengfort.com $

"""
Script to generate figures for the PyCon 2017 Yellowbrick Poster.
"""

##########################################################################
## Imports
##########################################################################

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yellowbrick as yb
import matplotlib.pyplot as plt

from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Default place to store figures directory
BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, "data")

# Listing of datasets and their paths
datasets = {
    "bikeshare": os.path.join(DATADIR, "bikeshare", "bikeshare.csv"),
    "concrete": os.path.join(DATADIR, "concrete", "concrete.csv"),
    "credit": os.path.join(DATADIR, "credit", "credit.csv"),
    "energy": os.path.join(DATADIR, "energy", "energy.csv"),
    "game": os.path.join(DATADIR, "game", "game.csv"),
    "hobbies": os.path.join(DATADIR, "hobbies"),
    "mushroom": os.path.join(DATADIR, "mushroom", "mushroom.csv"),
    "occupancy": os.path.join(DATADIR, "occupancy", "occupancy.csv"),
}

##########################################################################
## Helper Functions
##########################################################################

def get_figures_path(name, ext=".pdf", root=BASEDIR):
    """
    Returns the path to store a figure with ``viz.poof(outpath=path)``,
    creating a figures directory if necessary.
    """
    # Location to store figures in
    figdir = os.path.join(root, "figures")

    # Create the figures directory if it does not exist
    if not os.path.exists(figdir):
        os.mkdir(figdir)

    # Determine if we need to add the default extension
    base, oext = os.path.splitext(name)
    if not oext:
        name = base + ext

    # Return the path to the figure
    return os.path.join(figdir, name)


def savefig(name, visualizer, png=False):
    """
    Saves the visualizer to the figures directory with the given name.
    """
    outpdf = get_figures_path(name)
    visualizer.poof(outpath=outpdf)

    if png:
        outpng = get_figures_path(name, ext=".png")
        visualizer.poof(outpath=outpng)


##########################################################################
## Data Loader
##########################################################################

# TODO: Add memoization and caching to prevent reload of data.
def load_data(name, cols=None, target=None, tts=False, text=False):
    # Get the path from the datasets
    path = datasets[name]

    # TODO: handle the NLP datasets
    if text:
        return load_text(name, categories=cols, tts=tts)

    # Load the data frame
    data = pd.read_csv(path)

    # Get X and y data sets
    X = data[cols].as_matrix() if cols else data
    y = data[target].as_matrix() if target else None

    if tts:
        return train_test_split(X, y, test_size=0.2)
    return X,y


def load_text(name, categories=None, tts=False):
    # Get the path from the datasets
    path = datasets[name]

    # Do a listdir to get the categories if not supplied
    if not categories:
        categories = [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]

    # Get the paths to files on disk as input
    X, y = [], []
    for category in categories:
        for name in os.listdir(os.path.join(path, category)):
            X.append(os.path.join(path, category, name))
            y.append(category)

    if tts:
        return train_test_split(X, y, test_size=0.2)
    return X,y


##########################################################################
## Transformers
##########################################################################

class MultiColumnEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.encoders = [
            LabelEncoder().fit(column)
            for column in X.T
        ]
        return self

    def transform(self, X):
        return np.array([
            self.encoders[idx].transform(column)
            for idx, column in enumerate(X.T)
        ]).T


class ToDense(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


##########################################################################
## Visualizer Functions
##########################################################################

def radviz(ax):

    from yellowbrick.features import RadViz

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    target   = "occupancy"
    classes  = ['unoccupied', 'occupied']

    # Load the data
    X, y = load_data('occupancy', cols=features, target=target)

    # Instantiate and fit the visualizer
    visualizer = RadViz(ax=ax, classes=classes, features=features)
    visualizer.title = "RadViz of Features to Predict Room Occupancy"
    visualizer.fit(X, y)
    visualizer.transform(X)
    return visualizer


def pcoords(ax):
    from yellowbrick.features import ParallelCoordinates

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    target   = "occupancy"
    classes  = ['unoccupied', 'occupied']

    # Load the data
    X, y = load_data('occupancy', cols=features, target=target)

    # Instantiate and fit the visualizer
    visualizer = ParallelCoordinates(ax=ax, classes=classes, features=features)
    visualizer.title = "Parallel Coordinates of Features to Predict Room Occupancy"
    visualizer.fit(X, y)
    visualizer.transform(X)
    return visualizer


def rank2d(ax, algorithm='pearson'):
    from yellowbrick.features import Rank2D

    # Specify the features of interest
    features = [
        'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
        'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
        'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
        'jul_pay', 'aug_pay', 'sep_pay',
    ]

    # Load the data
    X, y = load_data('credit', cols=features, target='default')

    # Instantiate and fit the visualizer
    visualizer = Rank2D(features=features, algorithm=algorithm)
    visualizer.title = "2D Ranking of Pairs of Features by {}".format(algorithm.title())
    visualizer.fit(X, y)
    visualizer.transform(X)
    return visualizer


def residuals(ax):
    from sklearn.linear_model import RidgeCV
    from yellowbrick.regressor import ResidualsPlot

    features = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']

    splits = load_data('concrete', cols=features, target='strength', tts=True)
    X_train, X_test, y_train, y_test = splits

    estimator = RidgeCV()
    visualizer = ResidualsPlot(estimator, ax=ax)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def perror(ax):
    from sklearn.linear_model import LassoCV
    from yellowbrick.regressor import PredictionError

    features = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']

    splits = load_data('concrete', cols=features, target='strength', tts=True)
    X_train, X_test, y_train, y_test = splits

    estimator = LassoCV()
    visualizer = PredictionError(estimator, ax=ax)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def classification_report(ax):
    from sklearn.naive_bayes import GaussianNB
    from yellowbrick.classifier import ClassificationReport

    features = [
        "a1", "a2", "a3", "a4", "a5", "a6",
        "b1", "b2", "b3", "b4", "b5", "b6",
        "c1", "c2", "c3", "c4", "c5", "c6",
        "d1", "d2", "d3", "d4", "d5", "d6",
        "e1", "e2", "e3", "e4", "e5", "e6",
        "f1", "f2", "f3", "f4", "f5", "f6",
        "g1", "g2", "g3", "g4", "g5", "g6",
    ]

    classes = ['win', 'loss', 'draw']

    splits = load_data('game', cols=features, target='outcome', tts=True)
    X_train, X_test, y_train, y_test = splits

    labels = LabelEncoder()

    estimator = Pipeline([
        ('encoder', MultiColumnEncoder()),
        ('onehot', OneHotEncoder()),
        ('dense', ToDense()),
        ('nbayes', GaussianNB()),
    ])
    visualizer = ClassificationReport(estimator, ax=ax)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def confusion_matrix(ax):
    from sklearn.linear_model import LogisticRegression
    from yellowbrick.classifier import ConfusionMatrix

    features = [
        "a1", "a2", "a3", "a4", "a5", "a6",
        "b1", "b2", "b3", "b4", "b5", "b6",
        "c1", "c2", "c3", "c4", "c5", "c6",
        "d1", "d2", "d3", "d4", "d5", "d6",
        "e1", "e2", "e3", "e4", "e5", "e6",
        "f1", "f2", "f3", "f4", "f5", "f6",
        "g1", "g2", "g3", "g4", "g5", "g6",
    ]

    classes = ['win', 'loss', 'draw']

    splits = load_data('game', cols=features, target='outcome', tts=True)
    X_train, X_test, y_train, y_test = splits

    labels = LabelEncoder()

    estimator = Pipeline([
        ('encoder', MultiColumnEncoder()),
        ('onehot', OneHotEncoder()),
        ('dense', ToDense()),
        ('maxent', LogisticRegression()),
    ])
    visualizer = ConfusionMatrix(estimator, ax=ax)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def rocauc(ax):
    from yellowbrick.classifier import ROCAUC
    from sklearn.ensemble import RandomForestClassifier

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    target   = "occupancy"
    classes  = ['unoccupied', 'occupied']

    # Load the data
    splits = load_data('occupancy', cols=features, target=target, tts=True)
    X_train, X_test, y_train, y_test = splits

    estimator = RandomForestClassifier()
    visualizer = ROCAUC(estimator, ax=ax)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def class_balance(ax):
    from yellowbrick.classifier import ClassBalance
    from sklearn.ensemble import RandomForestClassifier

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    target   = "occupancy"
    classes  = ['unoccupied', 'occupied']

    # Load the data
    splits = load_data('occupancy', cols=features, target=target, tts=True)
    X_train, X_test, y_train, y_test = splits

    estimator = RandomForestClassifier()
    visualizer = ClassBalance(estimator, ax=ax, classes=classes)
    visualizer.title = "Class Balance of Room Occupancy Target"
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    return visualizer


def elbow(ax):
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.datasets import make_blobs

    kws = {
        'centers': 8,
        'n_samples': 1000,
        'n_features': 12,
        'shuffle': True,
    }

    X = make_blobs()
    X, y = make_blobs(centers=8)
    visualizer = KElbowVisualizer(KMeans(), ax=ax, k=(2,12))
    visualizer.title = "Silhouette Ranked Elbow Curve for K-Means on 8 Blob Dataset"
    visualizer.fit(X)
    return visualizer


def silhouette(ax):
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import SilhouetteVisualizer
    from sklearn.datasets import make_blobs

    kws = {
        'centers': 8,
        'n_samples': 1000,
        'n_features': 12,
        'shuffle': True,
    }

    X = make_blobs()
    X, y = make_blobs(centers=8)
    visualizer = SilhouetteVisualizer(KMeans(6), ax=ax)
    visualizer.title = "Silhouette Clusters for K-Means (k=6) on an 8 Blob Dataset"
    visualizer.fit(X)
    return visualizer


def alphas(ax):
    from sklearn.linear_model import RidgeCV
    from yellowbrick.regressor import AlphaSelection

    features = [
        "relative compactness", "surface area", "wall area", "roof area",
        "overall height", "orientation", "glazing area",
        "glazing area distribution"
    ]
    target = "heating load"
    # target = "cooling load"

    X, y = load_data("energy", cols=features, target=target)

    estimator = RidgeCV(scoring="neg_mean_squared_error")
    visualizer = AlphaSelection(estimator, ax=ax)
    visualizer.title = ""
    visualizer.fit(X, y)
    return visualizer


def freqdist(ax, stopwords=None):
    from sklearn.feature_extraction.text import CountVectorizer
    from yellowbrick.text import FreqDistVisualizer

    X, y = load_data("hobbies", text=True)

    freq = CountVectorizer(input='filename', stop_words=stopwords)
    X = freq.fit_transform(X)

    title = "Frequency Distribution of Top 50 Tokens in a Corpus"
    if stopwords:
        title += " (Without Stopwords)"

    visualizer = FreqDistVisualizer(ax=ax)
    visualizer.title = title
    visualizer.fit(X, freq.get_feature_names())
    return visualizer


def tsne(ax):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from yellowbrick.text import TSNEVisualizer

    X, y = load_data("hobbies", text=True)

    freq = TfidfVectorizer(input='filename', stop_words='english')
    X = freq.fit_transform(X)

    visualizer = TSNEVisualizer(ax=ax)
    visualizer.title = "t-SNE Projection of the Hobbies Corpus"
    visualizer.fit(X, y)
    return visualizer


def postag(ax, text="nursery"):

    from postag_texts import nursery_rhyme
    from postag_texts import algebra
    from postag_texts import french_silk
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag, word_tokenize
    from yellowbrick.text import PosTagVisualizer

    title = "Highligthed Parts of Speech Tags of the {} Text".format(text.title())

    text = {
        'nursery': nursery_rhyme,
        'algebra': algebra,
        'recipe': french_silk,
    }[text]

    text = pos_tag(word_tokenize(text))
    visualizer = PosTagVisualizer(ax=ax)
    visualizer.title = title
    visualizer.fit_transform(text)
    return visualizer


##########################################################################
## Main Method
##########################################################################

# Listing of all the figures to generate
FIGURES = {
    # "occupancy_radviz": radviz,
    # "occupancy_parallel_coordinates": pcoords,
    "credit_default_covariance_rank2d": partial(rank2d, algorithm='covariance'),
    "credit_default_pearson_rank2d": partial(rank2d, algorithm='pearson'),
    # "concrete_ridgecv_residuals": residuals,
    # "concrete_lassocv_prediction_error": perror,
    # "game_nbayes_classification_report": classification_report,
    # "game_maxent_confusion_matrix": confusion_matrix,
    # "occupancy_random_forest_rocauc": rocauc,
    # "occupancy_random_forest_class_balance": class_balance,
    # "eight_blobs_kmeans_elbow_curve": elbow,
    # "eight_blobs_kmenas_silhouette": silhouette,
    # "energy_ridgecv_alphas": alphas,
    "hobbies_freq_dist": partial(freqdist, stopwords='english'),
    "hobbies_freq_dist_stopwords": partial(freqdist, stopwords=None),
    # "hobbies_tnse": tsne,
    # "nursery_nltk_pos_tag": partial(postag, text="nursery"),
    # "algebra_nltk_pos_tag": partial(postag, text="algebra"),
    # "recipe_nltk_pos_tag": partial(postag, text="recipe"),
}


if __name__ == '__main__':
    # Render all uncommented figures
    for name, method in FIGURES.items():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        visualizer = method(ax)
        savefig(name, visualizer, png=True)
