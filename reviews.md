---
layout: page
title: Data Science Book Reviews
permalink: /books/
---

In my quest to further my knowledge of Data Science, I have read several books. If you are considering buying any of these, this list may be of help! I have included a subjective ranking as to how complex the material in the book is so you can decide if a book is suited to your level of experience.



###Currently reading: 

###Advanced Analytics with Spark by [Sandy Ryza, Uri Laserson, Sean Owen, and Josh Wills](http://www.amazon.com/Advanced-Analytics-Spark-Patterns-Learning/dp/1491912766/ref=sr_1_1?ie=UTF8&qid=1433013688&sr=8-1&keywords=advanced+analytics+with+spark) <font color = 'orange'>(moderate)</font>

<img src = "/images/AA_spark.gif" align = right>

This book comes from the top data scientists at Cloudera and shows you how to use Spark for typical data science applications. Although Spark is available to use with a Python library (PySpark), the authors prefer to demonstrate the book in Scala for the most part, since Spark is written in this. So far, the book is excellent, allowing you to learn both Spark and Scala at the same time! There are some really great examples of how to use Spark properly in here. I have learned more about Spark so far reading just the first few chapters of the book than I have in my own study from the official Spark docs or several local Meetups I have been to. A little surprising, the book assumes very limited prior knowledge of machine learning, so you may be able to skip some of the material explaining how various machine learning algorithms work. Regardless, even though the book seems a tad thin, it really is incredibly useful. One caveat, however, is that the book was written with Spark 1.2 in mind, so the newest major changes (such as data frames from 1.3) are not discussed. That being said, if you plan to use Spark on any of your projects, read this first. 

###Already read:


###Hadoop: The Definitive Guide (3rd edition) by [Tom White](http://www.amazon.com/Hadoop-Definitive-Guide-Tom-White/dp/1449311520) <font color = 'orange'>(moderate)</font>

<img src = "/images/Hadoop_book.jpg" align = right>

This review is for the older 3rd edition of the book. The newer, [4th edition](http://www.amazon.com/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1433014408&sr=8-1&keywords=hadoop+the+definitive+guide+4th+edition) includes Spark (which the 3rd edition does not, sadly, because Spark hadn't been released yet) along with several other updates. In my opinion, the book provides a lot of great case examples for when the HDFS should be used and how to optimize it properly. Because of the very fast pace of development in the Hadoop ecosystem, a lot of the 3rd edition now feels behind (hence the new 4th edition). That being said, the chapters on Pig and Hive (which aren't going anywhere) were very helpful, along with information about actual use cases at several companies such as Facebook, Infochimps, and last.fm. The book is probably a better fit for a data engineer than a data scientist, but data scientists still need to understand some of the Hadoop system's strengths and weaknesses. If you have some question about almost anything involving Hadoop, go here first if Google/StackOverflow doesn't get you an answer.  

###Predictive Analytics by [Eric Siegel](http://www.amazon.com/Predictive-Analytics-Power-Predict-Click-ebook/dp/B00BGC2WGQ) <font color = 'green'>(beginner)</font> 

<img src = "/images/PredAn_Book.jpg" align = right width = 128 height = 168>

This book is more to understand why predictive analytics in data science is necessary for many industries. It gives some prominent examples, such as Target predicting pregnancy of a girl before her own father knew about it based on her purchasing habits. This is fairly high-level with few technical details, but it is a great peak inside the world of data science and what is possible.
The author runs a conference series called "Predictive Analytics World" where anyone attending can get a free copy of the book.

###Big Data for Dummies by [Judith Hurwitz, Alan Nugent, Dr. Fern Halper, and Marcia Kaufman](http://www.amazon.com/Big-Data-Dummies-Alan-Nugent-ebook/dp/B01C7A89OA/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869670&sr=1-1&keywords=big+data+for+dummies) <font color = 'green'>(beginner)</font>

<img src = "/images/DataDumb_Book.jpg" align = right width = 128 height = 168>

If you want to understand all of the IT jargon necessary in big data, this is a decent place to start. It emphasizes the hardware more than the software in big data, and gives advice to businesses on how to scale their existing infrastructure. It also talks about issues businesses should be concerned about, such as scalability and security (Target or Sony anyone??). It seems more meant for management or a business wanting to take their first steps into the world of big data. Interesting note: Dr. Fern Halper, one of the authors, also got her Ph.D. at Texas A&M.

###The Signal and The Noise by [Nate Silver](http://www.amazon.com/Signal-Noise-Many-Predictions-Fail-but-ebook/dp/B007V65R54/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869711&sr=1-1&keywords=signal+and+the+noise) <font color = 'green'>(beginner)</font>

<img src = "/images/Signal_Book.jpg" align = right width = 128 height = 168>

You aren't going to find many technical details here, but Nate Silver has done so many interesting things with data (especially in baseball and politics) it's more than worth getting some insight into how he thinks. There are some great examples of data being used effectively here. I especially appreciate his compliments towards the atmospheric science and meteorology community on improved weather forecast accuracy! I didn't agree entirely, however, with his chapter regarding global warming, even though he seemed to take great pains to be impartial. That being said, the book was truly fascinating and helped increase my desire to become a data scientist.

###Data Science for Business by [Foster Provost and Tom Fawcett](http://www.amazon.com/Data-Science-Business-data-analytic-thinking-ebook/dp/B00E6EQ3X4/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869759&sr=1-1&keywords=data+science+for+business) <font color = 'orange'>(moderate)</font>

<img src = "/images/DSBus_Book.jpg" align = right width = 128 height = 168>

This book seems to be a good fit for MBA types with a somewhat technical background. Now things are starting to get a little more difficult in this book. It's very practical and applied, and it includes some interesting cases such as classification of whiskey. It also talks about some of the basics a data scientist must understand such as ROC curves. There is also discussion of the simpler machine learning algorithms and how they are used. If you are a manager wanting to understand better what your data scientists are doing, this is the best book I have read so far.

###Python for Data Analysis by [Wes McKinney](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython-ebook/dp/B009NLMB8Q/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869820&sr=1-1&keywords=python+for+data+analysis) <font color = 'orange'>(moderate)</font>

<img src = "/images/PDA_Book.jpg" align = right width = 128 height = 168>

For those that know Python well already, a better title would be "The Official Guide to Pandas." If you want to understand the Pandas library from the perspective of the guy who actually created it, this book is incredibly useful. Some data scientists prefer to use R only and haven't made the jump into Python just yet. Scared? Read this book first. Pandas is based on data frames too! However, the syntax to manipulate them is extremely different than the syntax in R (more object-oriented), so this book should help you get up to speed quickly. A great reference to keep handy if googling Stack Overflow questions or reading the offical pandas documentation just isn't cutting it.

###Practical Data Science with R by [Nina Zumel and John Mount](http://www.amazon.com/Practical-Data-Science-Nina-Zumel/dp/1617291560/ref=asap_bc?ie=UTF8) <font color = 'orange'>(moderate)</font>

<img src = "/images/PDS_Book.jpg" align = right width = 128 height = 168>

If you want to see everything R can do for a data scientist (well, almost everything) this is your book. From visualization in ggplot, to manipulation of data frames, to using the right apply function, read this to learn how to use R in your work. The book guides you through every step of the data science process and gives advice on the entire workflow, from data discovery and exploration to implementation of your model. The authors even give advice on how to present your results to different audiences. I used this book to help myself learn R and its capabilities.

###Developing Analytic Talent: Becoming a Data Scientist by [Vincent Granville](http://www.amazon.com/Developing-Analytic-Talent-Becoming-Scientist/dp/1118810082/ref=sr_1_1?s=books&ie=UTF8&qid=1425869936&sr=1-1&keywords=developing+analytic+talent+becoming+a+data+scienist) <font color = 'red'>(difficult)</font>

<img src = "/images/DAT_Book.jpg" align = right width = 128 height = 168>

I decided to give this book a try because the website he is associated with (datasciencecentral.com) used to be one of my favorites for news and information about data science. However, as a book, it seems poorly organized. What Dr. Granville basically did is collate a selection of articles he had already written on the website and put them together in one place. Granville also seems to have his own way of doing things that tend to be different from most practitioners, so keep that in mind as you read. My recommendation is to just stick to the website if you are interested in what he has to say.

###Java: A Beginner's Guide (Sixth Edition) by [Herbert Schildt](http://www.amazon.com/Java-Beginners-Guide-Herbert-Schildt/dp/0071809252/ref=sr_1_1?s=books&ie=UTF8&qid=1425870089&sr=1-1&keywords=java%3A+a+beginner%27s+guide) <font color = 'red'>(difficult)</font>

<img src = "/images/Java_Book.jpg" align = right width = 128 height = 168>

While many data scientists don't necessarily need Java, the vast majority of the Hadoop framework is based upon it! More job ads are requesting Java than I had originally expected. Why? Because if you want to create any sort of custom UDF (user defined function) while using many components of Hadoop, you need to know how Java works. Java could also come in handy when a model you designed needs to be integrated with a company's existing system. It is a compiled language (pretty fast) that runs on almost anything (including your household electronics!) This book helped me truly understand the purpose of the object-oriented programming paradigm and why Java can be very useful. It's too difficult a book for someone entirely new to programming, but if you have several languages already under your belt, I highly recommend it.


##Online Books (free!)

I have found a couple of books that you can get for free. I highly recommend both.

###The Field Guide to Data Science from [Booz Allen Hamilton](http://www.boozallen.com/insights/2013/11/data-science-field-guide) <font color = 'red'>(difficult)</font> 

<img src = "/images/FGuide_Book.jpg" align = right width = 128 height = 168>

This gives advice from the data science team at BAH on how they like to approach problems. It is very colorful and well organized. The contents span a variety of subjects, such as hiring data science teams, what algorithms work best in certain situations, and case studies from problems they actually had to solve. It gives a great insight into how data scientists think.

###Elements of Statistical Learning by [Hastie, Tibshirani, and Friedman](http://statweb.stanford.edu/~tibs/ElemStatLearn/) <font color = 'purple'>(brace yourself . . .)</font>

<img src = "/images/ESL_Book.jpg" align = right width = 128 height = 168>

Anyone who is a really hard-core data scientist should have read this book. It describes all of the mathematics and minute details for almost every major supervised and unsupervised machine learning or statistical algorithm there is. It's not enough to just use these algorithms in R, Mahout, or scikit-learn and go on your merry way. To use them properly, it is important to understand how they really work and what the internals look like. Yes, it is a challenging book to read. There is a lot of advanced mathematics in it, and it is very long (over 700 pages!). If you don't have a strong math background, it will be intimidating to you. Tough. Read it anyway, all of it! If it makes you feel any better, the authors themselves (who are brilliant!) have realized the book is very technical and have added a little icon of a yellow man with hands in the air to mark sections they feel are really difficult. The authors claim you can skip these sections if you wish, but I would try to read them regardless. It is truly considered a classic for many data scientists and machine learning experts.


