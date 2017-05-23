# PyCon2017

**Resources and materials related to PyCon 2017.**

## Poster Visualizations

The `poster` directory generates the figures for the Yellowbrick poster. To run, make sure that you have Yellowbrick 0.4 or later installed as well as Pandas and NLTK (with the NLTK corpora downloaded). Then run the following:

    $ python download.py
    $ python figures.py

The first script downloads the datasets and the second generates the figures. They can be found in PDF in the `figures` directory. Note that all directories are relative to the `poster` directory.


## Tutorial

The `tutorial` directory is a copy of the [github repo](https://github.com/nd1/pycon_2017) for the tutorial [Fantastic Data and Where To Find Them: An introduction to APIs, RSS, and Scraping](https://us.pycon.org/2017/schedule/presentation/177/). The presentation slides were created using [reveal.js](https://github.com/hakimel/reveal.js/) and [jupyter notebook](http://jupyter.org/). They are [hosted online](https://nd1.github.io/pycon_2017/#/) at the project repo, or you can view them locally by running the following in the directory where `slides.ipynb` is located:

    $ jupyter-nbconvert --to slides slides.ipynb --reveal-prefix=reveal.js --post serve
