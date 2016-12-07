# TITLE
Building A Gigaword Corpus: Lessons on Data Ingestion, Management, and Processing for NLP

# CATEGORY
Best Practices

# DURATION
I prefer a 30 minute slot

# DESCRIPTION
_If your talk is accepted this will be made public and printed in the program. Should be one paragraph, maximum 400 characters._

As the applications we build are increasingly driven by text, doing data ingestion, management, loading, and preprocessing in a robust, organized, parallel, and memory-safe way can get tricky. This talk walks through the highs (a custom billion-word corpus!), the lows (segfaults, 400 errors, pesky mp3s), and the new Python libraries we built to ingest and preprocess text for machine learning.

# AUDIENCE
_Who is the intended audience for your talk? (Be specific; "Python programmers" is not a good answer to this question.)_

Application developers who want to integrate text analytics features into their software, and Python programmers who have tinkered with NLP and machine learning and are interested in leveraging these tools with a custom corpus.


# PYTHON LEVEL
Intermediate


# OBJECTIVES
 1. Discover why building your own corpus is important for language-aware data products.
 2. Understand the problems you're likely to encounter when building your own corpus and hear about our mistakes along the way.
 3. Learn how to leverage Python packages and other best practices for corpus ingestion, management, loading, and preprocessing.


# DETAILED ABSTRACT
_Detailed description. Will be made public if your talk is accepted._

As the applications we build are increasingly driven by text, doing data ingestion, management, loading, and preprocessing in a robust, organized, parallel, and memory-safe way can get tricky. This talk walks through the highs (a custom billion-word corpus!), the lows (segfaults, 400 errors, pesky mp3s), and the new Python libraries we built to ingest and preprocess text for machine learning.

While applications like Siri, Cortana, and Alexa may still seem like novelties, language-aware applications are rapidly becoming the new norm. Under the hood, these applications take in text data as input, parse it into composite parts, compute upon those composites, and then recombine them to deliver a meaningful and tailored end result. The best applications use language models trained on _domain-specific corpora_ (collections of related documents containing natural language) that reduce ambiguity and prediction space to make results more intelligible. Here's the catch: these corpora are huge, generally consisting of at least hundreds of gigabytes of data inside of thousands of documents, and often more!

In this talk, we'll see how working with text data is substantially different from working with numeric data, and show that ingesting a raw text corpus in a form that will support the construction of a data product is no trivial task. For instance, when dealing with a text corpus, you have to consider not only how the data comes in (e.g. respecting rate limits, terms of use, etc.), but also where to store the data and how to keep it organized. Because the data comes from the web, it's often unpredictable, containing not only text but audio files, ads, videos, and other kinds of web detritus. Since the datasets are large, you need to anticipate potential performance problems and ensure memory safety through streaming data loading and multiprocessing. Finally, in anticipation of the machine learning components, you have to establish a standardized method of transforming your raw ingested text into a corpus that's ready for computation and modeling.

In this talk, we'll explore many of the challenges we experienced along the way and introduce two Python packages that make this work a bit easier: [Baleen](https://pypi.python.org/pypi/baleen/0.3.3) and [Minke](https://github.com/bbengfort/minke). Baleen is a package for ingesting formal natural language data from the discourse of professional and amateur writers, like bloggers and news outlets, in a categorized fashion. Minke extends Baleen with a library that performs parallel data loading, preprocessing, normalization, and keyphrase extraction to support machine learning on a large-scale custom corpus.

# OUTLINE

## Introduction
_(5 minutes)_
 - Why we wanted to build a custom corpus and you should too.

## Things we learned along the way

### Data Ingestion and Management
_(5 minutes)_
 - The good, the bad, and the ugly: APIs, RSS and webscraping
 - Raw corpus vs. preprocessed corpus
 - Corpus disk structure

### Data Loading
_(5 minutes)_
 - Creating a custom corpus reader
 - Planning for streaming access
 - Parallelizing computation for speed

### Preprocessing and Wrangling
_(5 minutes)_
 - A list of lists of lists of tuples (content extraction, paragraph blocking, sentence segmentation, word tokenization, and part-of-speech tagging)
 - Spoilers! A word on cross-validation and vectorization

## Python Tools for Production-Grade NLP
_(5 minutes)_
 - `Baleen`: Build your own domain-specific dataset with a production-grade RSS corpus builder.
 - `Minke`: Load, preprocess, normalize, and preprocess your custom corpus to prepare it for machine learning.

## A Case Study
_(5 minutes)_
 - `Partisan Discourse`: An interactive NLP web application that identifies party in political discourse and an example of operationalized machine learning.

## Additional Notes
_Anything else you'd like the program committee to know when making their selection: your past speaking experience, open source community experience, etc._

I am an active contributor to the open source community, and this proposed talk is based on three open source Python projects I have been working on with the team at District Data Labs, an open source collaborative in Washington, DC:
 - [Baleen](https://pypi.python.org/pypi/baleen/0.3.3) is an automated RSS ingestion service designed to construct a production-grade text corpus for NLP research and machine learning applications.
 - [Minke](https://github.com/bbengfort/minke) extends Baleen with a library to perform extraction, preprocessing, keyphrase extraction, and modeling on the exported corpora.
 - [Partisan Discourse](https://github.com/DistrictDataLabs/partisan-discourse/) is an interactive NLP web application that identifies party in political discourse and an example of operationalized machine learning.

The ideas have been met with a great deal of interest and enthusiasm so far, and I am excited to share them and to help Python programmers integrate more NLP into their applications!

I was honored to speak at [PyCon 2016](https://www.youtube.com/watch?v=c5DaaGZWQqY), [PyData Carolinas 2016](https://www.youtube.com/watch?v=cgtNPx7fJUM), and [PyData DC 2016](https://www.youtube.com/watch?v=xJYerGy8SzY), and am a co-author of the forthcoming O'Reilly book, __Applied Text Analysis with Python__. I am also an organizer for Data Community DC - a not-for-profit organization of 9 meetups that organizes free monthly events and lectures for the local data community in Washington, DC.
