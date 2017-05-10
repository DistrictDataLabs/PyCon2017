## Title: Fantastic Data and Where To Find Them: An introduction to APIs, RSS, and Scraping


## Description:


Whether you’re building a custom web application, getting started in machine learning, or just want to try something new, everyone needs data. And while the web offers a seemingly boundless source for custom data sets, the collection of that data can present a whole host of obstacles. From ever-changing APIs to rate-limiting woes, from nightmarishly nested XML to convoluted DOM trees, working with APIs and web scraping are challenging but critically useful skills for application developers and data scientists alike. In this tutorial, we’ll introduce RESTful APIs, RSS feeds, and web scraping in order to see how different ingestion techniques impact application development. We’ll explore how and when to use Python libraries such as `feedparser`, `requests`, `beautifulsoup`, and `urllib`. And finally we will present common data collection problems and how to overcome them. 


We’ll take a hands-on, directed exercise approach combined with short presentations to engage a range of different APIs (with and without authentication), explore examples of how and why you might web scrape, and learn the ethical and legal considerations for both. To prepare attendees to create their own data ingestion scripts, the tutorial will walk through a set of examples for robust and responsible data collection and ingestion. This tutorial will conclude with a case study of [Baleen](https://pypi.python.org/pypi/baleen/0.3.3), an automated RSS ingestion service designed to construct a production-grade text corpus for NLP research and machine learning applications. Exercises will be presented both as Jupyter Notebooks and Python scripts.


## Audience:
This is an intermediate Python tutorial for anyone interested in data collection. Attendees will be expected to have beyond beginner knowledge of Python, but no experience at accessing APIs or web scraping. Users will be introduced to libraries for APIs and web scraping including, but not limited to: `feedparser`, `urllib`, `requests`, `beautifulsoup`. In particular, students will be required to have the following knowledge & preparations in advance of the course:
 
- Python 3 installation (Anacondas is fine)
- Knowledge of how to run and execute Jupyter Notebooks
- Knowledge of how to write and execute Python scripts
- Understanding of how to use the command line
- Necessary libraries installed
- Understand the client-server model of HTTP Requests


At the conclusion of this tutorial, attendees will be able to:
- Collect data from APIs, with and without authentication
- Script the process of API data collection
- Navigate the ethical and legal issues involved with web scraping
- Perform basic web scraping on HTML sites with Python libraries
- Understand the potential of operationalization of data collection with an example


## Outline:


Intro
Who you are, who are we
Session Overview
Time: 0:10, Total Time: 0:10


APIs, an introduction (lecture)
What are APIs?
Where are APIs?
How do I access APIs?
What is the API giving me?
Time: 0:15, Total Time: 0:25


APIs, hands-on (workshop)
RSS (notebook)
REST APIs with Authentication (notebook)
Scripting API Calls (script)
Time: 0:35, Total Time: 1:00 


APIs, review (lecture)
Review what we did
Time: 0:05, Total Time: 1:05


Web scraping, an introduction (lecture)
What is web scraping?
When should I web scrape?
How do I web scrape?
Time: 0:25, Total Time: 1:30


Break
Time: 0:20, Total Time: 1:50


Web scraping, hands-on (workshop)
BeautifulSoup, an overview (notebook)
Web scraping, downloading data example (notebook)
Web scraping, try it out (notebook/ script)
Time: 0:55, Total Time: 2:45


Web scraping, review (lecture)
Review what we did
Time: 0:05, Total Time: 2:50


Baleen, a case study (lecture)
How can we operationalization data collection?
Time: 0:20, Total Time: 3:10


Conclusion (lecture)
Time: 0:10, Total Time: 3:20








## Additional Notes:
This proposal is based on a workshop given at [Tech Lady Hackathon + Training Day DC #4](http://techladyhackathon.org/)  ([github repo](https://github.com/nd1/tlh4_workshop)), no video available) and has been expanded to assume attendees will have Python experience and an environment available to work through exercises in notebooks and scripts. 


Nicole Donnelly is currently the teaching assistant for the [Data Science professional certificate](http://scs.georgetown.edu/programs/375/data-science/) at [The Georgetown University Center for Continuing & Professional Education](http://scs.georgetown.edu/departments/5/center-for-continuing-and-professional-education/). In this role, she has supplemented existing materials and created new ones to lead students through hands-on exploration of lecture topics. She has spoken at [PyData DC](http://pydata.org/dc2016/schedule/presentation/35/) ([video](https://www.youtube.com/watch?v=1dKonIT-Yak)) and [Women Data Scientists DC](https://www.meetup.com/WomenDataScientistsDC/events/235267675/) ([video](https://www.youtube.com/watch?v=Y1h7BgLA1Zc)) since embarking on a career change in January 2016. In her prior role as a computer forensics and electronic discovery consultant, Nicole mentored and trained junior employees in positions that required a significant amount of on the job learning, and spoke at professional conferences including Techno Forensics (now [Techno Security & Digital Forensics Conference](http://www.technosecurity.us/) and the Computer and Enterprise Investigations Conference (now [enfuse](https://www.guidancesoftware.com/enfuse-conference/about)). 


Nicole started using APIs and web scraping in July while building a prototype project to replicate (Chicago's Food Inspection Forecasting)[https://chicago.github.io/food-inspections-evaluation/] in Python for restaurant inspections in Washington, DC. She is now working on building out a sustainable scraper at [Code for DC](http://codefordc.org/index.html) so the data set can be built out until such a time when DC converts the inspection information to an [open data set](http://opendata.dc.gov/).


Will Voorhees is a software development engineer that specializes in writing enterprise security tools. For the past five years he's been working on products for protecting communication between services in a service oriented architecture. These products run on hundreds of thousands of servers all over the world. In his spare time he spends far too much time on YouTube and enjoys playing video games.


Will has spoken at [PyData DC](http://pydata.org/dc2016/schedule/presentation/50/) ([video](https://www.youtube.com/watch?v=_xa9R50e4v4), helped with tutorials at [Strata](http://conferences.oreilly.com/strata/stratany2013/public/schedule/detail/30806), and as a grad student, taught rambuncous college freshmen the value of Computer Science.
Will is a developer on Baleen and an ardent Python supporter.


[Tony Ojeda](https://www.linkedin.com/in/tonyojeda) is a data scientist, author, and entrepreneur with expertise in streamlining business processes and over a decade of experience creating innovative data products. He is the Founder of District Data Labs, where he pursues his passion for advancing the field of data science and the abilities of those who practice it. Tony has an MS in Finance from Florida International University and an MBA in Strategy and Entrepreneurship from DePaul University. He is a Co-founder and former President of Data Community DC, a non-profit organization that promotes the work of data scientists through community-driven events, and a co-author of the Practical Data Science Cookbook (published in 2014) and Applied Text Analytics with Python (coming in 2017). 




The case study presented in this tutorial, [Baleen](http://baleen-ingest.readthedocs.io/en/latest/), is an automated ingestion service for blogs to construct a corpus for NLP research built and maintained as an open source project by [District Data Labs](https://www.districtdatalabs.com/#research-lab), a data science research and education company based in the Washington, DC area. Baleen was built to operationalization the data collection process. 




## Additional Speakers: Will, Tony




