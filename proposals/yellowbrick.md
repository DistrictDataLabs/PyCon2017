# Title
Yellowbrick: Steering Scikit-Learn with Visual Transformers

## Presenters
Rebecca Bilbro and Benjamin Bengfort

## Description
In machine learning, model selection is a bit more nuanced than simply picking the 'right' or 'wrong' algorithm. In practice, the workflow includes (1) selecting and/or engineering the smallest and most predictive feature set, (2) choosing a set of algorithms from a model family, and (3) tuning the algorithm hyperparameters to optimize performance. Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications. In practice, however, human intuition and guidance can more effectively hone in on quality models than exhaustive search.

This poster presents a new Python library, [Yellowbrick](https://pypi.python.org/pypi/yellowbrick/0.3.1), which extends the Scikit-Learn API with a visual transfomer (visualizer) that can incorporate visualizations of the model selection process into pipelines and modeling workflow. Visualizers enable machine learning practitioners to visually interpret the model selection process, steer workflows toward more predictive models, and avoid common pitfalls and traps.

Yellowbrick is an open source, pure Python project that extends Scikit-Learn with visual analysis and diagnostic tools. The Yellowbrick API also wraps matplotlib to create publication-ready figures and interactive data explorations while still allowing developers fine-grain control of figures. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models, and assist in diagnosing problems throughout the machine learning workflow.

In this poster, we'll show not only what you can do with Yellowbrick, but how it works under the hood (since we're always looking for new contributors!). We'll illustrate how Yellowbrick extends the Scikit-Learn and Matplotlib APIs with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the Scikit-Learn Pipeline process - providing iterative visual diagnostics throughout the transformation of high dimensional data.


## Additional Notes
_Anything else you'd like the program committee to know when making their selection: your past speaking experience, open source community experience, etc._

At PyCon 2016, Ben and Rebecca presented two posters, one on [An Architecture for Machine Learning in Django](http://pycon.districtdatalabs.com/posters/machine-learning/horizontal/ddl-machine-learning-print.pdf) and one on [Evolutionary Design of Particle Swarms](http://pycon.districtdatalabs.com/posters/python-for-science/horizontal/ddl-python-for-science-horz-print.pdf). Rebecca delivered a [talk](https://www.youtube.com/watch?v=c5DaaGZWQqY) about visual diagnostics for machine learning that mapped out the foundations of the Yellowbrick project, which she also shared at [PyData Carolinas 2016](https://www.youtube.com/watch?v=cgtNPx7fJUM) and [PyData DC 2016](https://www.youtube.com/watch?v=xJYerGy8SzY). Ben also presented a PyCon tutorial in 2016 on [Natural Language Processing with NLTK and Gensim](https://www.youtube.com/watch?v=itKNpCPHq3I&feature=youtu.be). All their PyCon 2016 materials are published and freely available at http://pycon.districtdatalabs.com/.

Ben and Rebecca are active contributors to the open source community, and this poster is based on Yellowbrick, a project they've been building together with the team at District Data Labs, an open source collaborative in Washington, DC. They are also co-authors of the forthcoming O'Reilly book, __Applied Text Analysis with Python__ and organizers for Data Community DC - a not-for-profit organization of 9 meetups that organizes free monthly events and lectures for the local data community in Washington, DC.
