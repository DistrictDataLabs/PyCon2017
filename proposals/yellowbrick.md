# Title
Machine Learning with Yellowbrick: Extending the Scikit-Learn API with Visual Diagnostics

# Presenters
Rebecca Bilbro and Benjamin Bengfort

# Description
In machine learning, model selection is much more nuanced than just picking the 'right' or 'wrong' algorithm. In practice, the workflow includes (1) selecting and/or engineering the smallest and most predictive feature set, (2) choosing a set of algorithms from a model family, and (3) tuning the algorithm hyperparameters to optimize performance. Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications, but in practice, human intuition and guidance are much more effective than exhaustive search and comparison. This poster presents a new option, the [Yellowbrick](https://pypi.python.org/pypi/yellowbrick/0.3.1) library, which enables machine learning practitioners to visualize the model selection process, steer towards interpretable models, and avoid common pitfalls and traps.

Yellowbrick is an open source, pure Python project aimed at providing a visualization and diagnosis platform for machine learning to help steer the model selection process. It provides visual analysis and diagnostic tools built with custom Matplotlib and designed to facilitate machine learning with Scikit-Learn. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models, and assist in diagnosing problems throughout the machine learning workflow.

In this poster, we'll show not only what you can do with Yellowbrick, but how it works under the hood (since we're always looking for new contributors!). We'll illustrate how Yellowbrick extends the Scikit-Learn and Matplotlib APIs with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the Scikit-Learn Pipeline process - providing iterative visual diagnostics throughout the transformation of high dimensional data.


# Additional Notes
_Anything else you'd like the program committee to know when making their selection: your past speaking experience, open source community experience, etc._

At PyCon 2016, Ben and Rebecca presented two posters, one on [An Architecture for Machine Learning in Django](http://pycon.districtdatalabs.com/posters/machine-learning/horizontal/ddl-machine-learning-print.pdf) and one on [Evolutionary Design of Particle Swarms](http://pycon.districtdatalabs.com/posters/python-for-science/horizontal/ddl-python-for-science-horz-print.pdf). Rebecca delivered a [talk](https://www.youtube.com/watch?v=c5DaaGZWQqY) about visual diagnostics for machine learning that mapped out the foundations of the Yellowbrick project, which she also shared at [PyData Carolinas 2016](https://www.youtube.com/watch?v=cgtNPx7fJUM) and [PyData DC 2016](https://www.youtube.com/watch?v=xJYerGy8SzY). Ben also presented a PyCon tutorial in 2016 on [Natural Language Processing with NLTK and Gensim](https://www.youtube.com/watch?v=itKNpCPHq3I&feature=youtu.be). All their PyCon 2016 materials are published and freely available at http://pycon.districtdatalabs.com/.

Ben and Rebecca are active contributors to the open source community, and this poster is based on Yellowbrick, a project they've been building together with the team at District Data Labs, an open source collaborative in Washington, DC. They are also co-authors of the forthcoming O'Reilly book, __Applied Text Analysis with Python__ and organizers for Data Community DC - a not-for-profit organization of 9 meetups that organizes free monthly events and lectures for the local data community in Washington, DC.
