---
layout: page
title: Data Science Book Reviews
permalink: /books/
---

In my quest to further my knowledge of Data Science, I have read several books. If you are considering buying any of these, this list may be of help! I have included a subjective ranking as to how complex the material in the book is so you can decide if a book is suited to your level of experience.



### Scala for Data Science by [Pascal Bugnion](https://www.amazon.com/Scala-Data-Science-Pascal-Bugnion/dp/1785281372) <font color = 'orange'>(moderate)</font>

![](/images/Scala_DS.jpg){:align="right" width="128px" height="168px"}

I have been waiting for a book like this one for quite some time. While other books focus more on Spark specifically, this book focuses primarily on other libraries available in Scala that could prove useful for your workflow. While not much attention is paid to machine learning (there is some in Chapter 12), the book focuses more on Scala’s strengths such as immutability, futures, and type safety. There is also a lot of focus on how to interact with databases. The author essentially argues that Scala’s best use is for permanent programs instead of exploration and experimentation. This book will definitely come in handy if you want to translate your model into production code that requires scale.

### Amazon Web Services in Action by [Andreas Wittig and Michael Wittig](https://www.amazon.com/Amazon-Services-Action-Andreas-Wittig/dp/1617292885) <font color = 'orange'>(moderate)</font>

![](/images/AWS_action.jpg){:align="right" width="128px" height="168px"}

The myriad of possible features available from Amazon Web Services (AWS) can be a tad intimidating to understand. I decided what I really wanted was a guide to teach me how to use these services from experienced professionals. Fortunately, we now have this book! Geared more towards DevOps Engineers than Data Scientists, the book still was a welcome help for me. All of the major aspects of AWS are covered (although Redshift is not even mentioned in the book, which I found quite surprising!) along with examples of how to configure them in great detail. Web applications, security, load balancing, databases, and flexible computing resources are all covered. I feel much more confident that I could use AWS with greater ease than before. The Wittig brothers really know their stuff. If you want to start using AWS more in your work, definitely keep this book handy.


### Doing Data Science by [Cathy O'Neill & Rachel Schutt](http://www.amazon.com/Doing-Data-Science-Straight-Frontline/dp/1449358659) <font color = 'green'>(beginner)</font>

![](/images/DDS_Book.gif){:align="right" width="128px" height="168px"}

This book is a great survey of the field. It covers almost every aspect of data science, but it doesn't get into details as much as other books do. This is a great book to start with if you are just getting introduced to data science for the first time. Chapter 8 on recommendation engines was exceptionally good, especially since most books don't discuss them at all or simply gloss over them. The chapters also have information from experts in several domains, including data engineering, data visualization, machine learning at Kaggle, and many more. 


### Learning Spark by [Holden Karau, Andy Konwinski, Patrick Wendell & Matei Zaharia](http://www.amazon.com/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624) <font color = 'orange'>(moderate)</font>

![](/images/Learn_Spark.jpg){:align="right" width="128px" height="168px"}
Written by core developers of Spark (including Matei Zaharia himself), this book is a nice complement to Advanced Analytics with Spark. While the other book focuses on applications for an analytical context, this book focuses more on the internals of Spark itself. This includes how Spark works, useful tips for debugging, available settings to use, performance optimization, and scenarios where Spark may be a good solution. This is a good reference book when Spark has you stuck. Python, Scala, and Java code samples are shown throughout the book, so don't worry about your personal language preference (although the Scala API is probably the one you should end up using eventually). 


### Data Science at The Command Line by [Jeroen Janssens](http://www.amazon.com/Data-Science-Command-Line-Time-Tested/dp/1491947853) <font color = 'orange'>(moderate)</font>

![](/images/DSAtCL.png){:align="right" width="128px" height="168px"}

This book has long been on my to-do list after using the author's [data science toolbox](http://datasciencetoolbox.org/). If you know of anyone that uses the command line all the time, this book will show you how to leverage it properly. The book goes through several very useful tools I had no idea were available. The one that especially interested me was the GNU Parallel tool, which was new to me. With this, you can run command line inputs in parallel very simply utilizing all of the cores available on your machine! The author argues for doing a very large portion of your work on the command line, which I don't necessarily agree is the most practical answer. It will certainly come in handy, however, for data preprocessing. I recommend reading this just to simply broaden your horizons as to what tools are available. You can then pick which command line tools best support your workflow.

### Interactive Data Visualization by [Scott Murray](http://www.amazon.com/Interactive-Data-Visualization-Scott-Murray/dp/1449339735) <font color = 'green'>(beginner)</font>

![](/images/IDVis.jpg){:align="right" width="128px" height="168px"}

Written by design professor Scott Murray, this book is an excellent introduction to d3.js. It's written to be friendly even to people less experienced with programming, such as journalists who want to add d3's capabilities to their articles. There are several great tutorials in here that help explain what d3 is doing and also provide an easy-to-follow high-level introduction to javascript. The author is not a software engineer, however, so some of the syntax he chooses may seem a little different than what a more experienced engineer may have written. By the end of this book, you should have enough knowledge to make your own simple d3 plots. It's not meant to be an advanced book, so for more complicated d3 plots it may be best to look at the [gallery](https://github.com/mbostock/d3/wiki/Gallery) from Mike Bostock, who helped create d3. 


### Mining the Social Web (2nd edition) by [Matthew A. Russell](http://www.amazon.com/Mining-Social-Web-Facebook-LinkedIn/dp/1449367615) <font color = 'orange'>(moderate)</font>
<!---
<img src = "/images/Mining_Soc_Web.gif" align = right>
-->
![](/images/Mining_Soc_Web.gif){:align="right" width="128px" height="168px"}

The book focuses on how to use social media (along with the unstructured data inside of it) to gain insights that can help your business or just answer interesting sociological questions. The book mainly focuses on how to extract data from a variety of social networking sites, such as Facebook, Twitter, LinkedIn (although LinkedIn, unfortunately,  have tightened up a lot on your ability to do this now) and Github. It also offers some explanation of Natural Language Processing (NLP) in Chapter 5. I was disappointed the book only stopped at gathering data for the most part and didn't do much else with it (no recommendation engines or machine learning applications). The book is also starting to show some age, unfortunately (which isn't the author's fault of course). It is written entirely in Python, but it uses Python 2 instead of 3. The best chapter is probably the last one, called the "Twitter Cookbook", which gives a bunch of useful code samples for extracting data using Twitter's API, although there is a chance some of it is out of date. I wanted to like this book but I felt like it was missing something. If you need to extract a lot of social network data for an upcoming project, this book might be worth checking out.

### Advanced Analytics with Spark by [Sandy Ryza, Uri Laserson, Sean Owen, and Josh Wills](http://www.amazon.com/Advanced-Analytics-Spark-Patterns-Learning/dp/1491912766/ref=sr_1_1?ie=UTF8&qid=1433013688&sr=8-1&keywords=advanced+analytics+with+spark) <font color = 'orange'>(moderate)</font>

<!---
<img src = "/images/AA_spark.gif" align = right>
-->
![](/images/AA_spark.gif){:align="right" width="128px" height="168px"}
This book comes from the top data scientists at Cloudera and shows you how to use Spark for typical data science applications. Although Spark is available to use with a Python library (PySpark), the authors prefer to demonstrate the book in Scala for the most part, since Spark is written in this. The book is excellent, allowing you to learn both Spark and Scala at the same time! I agree with the authors that you do not need to read every chapter (if you only can read one, Chapter 2 is definitely the best!), as the book is structured across use cases from multiple industries. I have learned more about Spark from this book than I have in my own study from the official Spark docs or several local Meetups I have been to. A little surprising, the book assumes very limited prior knowledge of machine learning, so you may be able to skip some of the material explaining how various machine learning algorithms work. Regardless, even though the book seems a tad thin, it really is incredibly useful. One caveat, however, is that the book was written with Spark 1.2 in mind, so the newest major changes (such as data frames from 1.3) are not discussed. That being said, if you plan to use Spark on any of your projects, read this first. 



### Hadoop: The Definitive Guide (3rd edition) by [Tom White](http://www.amazon.com/Hadoop-Definitive-Guide-Tom-White/dp/1449311520) <font color = 'orange'>(moderate)</font>
<!---
<img src = "/images/Hadoop_book.jpg" align = right>
-->
![](/images/Hadoop_book.jpg){:align="right" width="128px" height="168px"}
This review is for the older 3rd edition of the book. The newer, [4th edition](http://www.amazon.com/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1433014408&sr=8-1&keywords=hadoop+the+definitive+guide+4th+edition) includes Spark (which the 3rd edition does not, sadly, because Spark hadn't been released yet) along with several other updates. In my opinion, the book provides a lot of great case examples for when the HDFS should be used and how to optimize it properly. Because of the very fast pace of development in the Hadoop ecosystem, a lot of the 3rd edition now feels behind (hence the new 4th edition). That being said, the chapters on Pig and Hive (which aren't going anywhere) were very helpful, along with information about actual use cases at several companies such as Facebook, Infochimps, and last.fm. The book is probably a better fit for a data engineer than a data scientist, but data scientists still need to understand some of the Hadoop system's strengths and weaknesses. If you have some question about almost anything involving Hadoop, go here first if Google/StackOverflow doesn't get you an answer.  

### Predictive Analytics by [Eric Siegel](http://www.amazon.com/Predictive-Analytics-Power-Predict-Click-ebook/dp/B00BGC2WGQ) <font color = 'green'>(beginner)</font> 
<!---
<img src = "/images/PredAn_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/PredAn_Book.jpg){:align="right" width="128px" height="168px"}
This book is more to understand why predictive analytics in data science is necessary for many industries. It gives some prominent examples, such as Target predicting pregnancy of a girl before her own father knew about it based on her purchasing habits. This is fairly high-level with few technical details, but it is a great peak inside the world of data science and what is possible.
The author runs a conference series called "Predictive Analytics World" where anyone attending can get a free copy of the book.

### Big Data for Dummies by [Judith Hurwitz, Alan Nugent, Dr. Fern Halper, and Marcia Kaufman](http://www.amazon.com/Big-Data-Dummies-Alan-Nugent-ebook/dp/B01C7A89OA/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869670&sr=1-1&keywords=big+data+for+dummies) <font color = 'green'>(beginner)</font>
<!---
<img src = "/images/DataDumb_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/DataDumb_Book.jpg){:align="right" width="128px" height="168px"}
If you want to understand all of the IT jargon necessary in big data, this is a decent place to start. It emphasizes the hardware more than the software in big data, and gives advice to businesses on how to scale their existing infrastructure. It also talks about issues businesses should be concerned about, such as scalability and security (Target or Sony anyone??). It seems more meant for management or a business wanting to take their first steps into the world of big data. Interesting note: Dr. Fern Halper, one of the authors, also got her Ph.D. at Texas A&M.

### The Signal and The Noise by [Nate Silver](http://www.amazon.com/Signal-Noise-Many-Predictions-Fail-but-ebook/dp/B007V65R54/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869711&sr=1-1&keywords=signal+and+the+noise) <font color = 'green'>(beginner)</font>
<!---
<img src = "/images/Signal_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/Signal_Book.jpg){:align="right" width="128px" height="168px"}
You aren't going to find many technical details here, but Nate Silver has done so many interesting things with data (especially in baseball and politics) it's more than worth getting some insight into how he thinks. There are some great examples of data being used effectively here. I especially appreciate his compliments towards the atmospheric science and meteorology community on improved weather forecast accuracy! I didn't agree entirely, however, with his chapter regarding global warming, even though he seemed to take great pains to be impartial. That being said, the book was truly fascinating and helped increase my desire to become a data scientist.

### Data Science for Business by [Foster Provost and Tom Fawcett](http://www.amazon.com/Data-Science-Business-data-analytic-thinking-ebook/dp/B00E6EQ3X4/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869759&sr=1-1&keywords=data+science+for+business) <font color = 'orange'>(moderate)</font>
<!---
<img src = "/images/DSBus_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/DSBus_Book.jpg){:align="right" width="128px" height="168px"}

This book seems to be a good fit for MBA types with a somewhat technical background. Now things are starting to get a little more difficult in this book. It's very practical and applied, and it includes some interesting cases such as classification of whiskey. It also talks about some of the basics a data scientist must understand such as ROC curves. There is also discussion of the simpler machine learning algorithms and how they are used. If you are a manager wanting to understand better what your data scientists are doing, this is the best book I have read so far.

### Python for Data Analysis by [Wes McKinney](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython-ebook/dp/B009NLMB8Q/ref=sr_1_1?s=digital-text&ie=UTF8&qid=1425869820&sr=1-1&keywords=python+for+data+analysis) <font color = 'orange'>(moderate)</font>
<!---
<img src = "/images/PDA_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/PDA_Book.jpg){:align="right" width="128px" height="168px"}
For those that know Python well already, a better title would be "The Official Guide to Pandas." If you want to understand the Pandas library from the perspective of the guy who actually created it, this book is incredibly useful. Some data scientists prefer to use R only and haven't made the jump into Python just yet. Scared? Read this book first. Pandas is based on data frames too! However, the syntax to manipulate them is extremely different than the syntax in R (more object-oriented), so this book should help you get up to speed quickly. A great reference to keep handy if googling Stack Overflow questions or reading the offical pandas documentation just isn't cutting it.

### Practical Data Science with R by [Nina Zumel and John Mount](http://www.amazon.com/Practical-Data-Science-Nina-Zumel/dp/1617291560/ref=asap_bc?ie=UTF8) <font color = 'orange'>(moderate)</font>
<!---
<img src = "/images/PDS_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/PDS_Book.jpg){:align="right" width="128px" height="168px"}
If you want to see everything R can do for a data scientist (well, almost everything) this is your book. From visualization in ggplot, to manipulation of data frames, to using the right apply function, read this to learn how to use R in your work. The book guides you through every step of the data science process and gives advice on the entire workflow, from data discovery and exploration to implementation of your model. The authors even give advice on how to present your results to different audiences. I used this book to help myself learn R and its capabilities.

### Developing Analytic Talent: Becoming a Data Scientist by [Vincent Granville](http://www.amazon.com/Developing-Analytic-Talent-Becoming-Scientist/dp/1118810082/ref=sr_1_1?s=books&ie=UTF8&qid=1425869936&sr=1-1&keywords=developing+analytic+talent+becoming+a+data+scienist) <font color = 'red'>(difficult)</font>
<!---
<img src = "/images/DAT_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/DAT_Book.jpg){:align="right" width="128px" height="168px"}
I decided to give this book a try because the website he is associated with (datasciencecentral.com) used to be one of my favorites for news and information about data science. However, as a book, it seems poorly organized. What Dr. Granville basically did is collate a selection of articles he had already written on the website and put them together in one place. Granville also seems to have his own way of doing things that tend to be different from most practitioners, so keep that in mind as you read. My recommendation is to just stick to the website if you are interested in what he has to say.

### Java: A Beginner's Guide (Sixth Edition) by [Herbert Schildt](http://www.amazon.com/Java-Beginners-Guide-Herbert-Schildt/dp/0071809252/ref=sr_1_1?s=books&ie=UTF8&qid=1425870089&sr=1-1&keywords=java%3A+a+beginner%27s+guide) <font color = 'red'>(difficult)</font>
<!---
<img src = "/images/Java_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/Java_Book.jpg){:align="right" width="128px" height="168px"}
While many data scientists don't necessarily need Java, the vast majority of the Hadoop framework is based upon it! More job ads are requesting Java than I had originally expected. Why? Because if you want to create any sort of custom UDF (user defined function) while using many components of Hadoop, you need to know how Java works. Java could also come in handy when a model you designed needs to be integrated with a company's existing system. It is a compiled language (pretty fast) that runs on almost anything (including your household electronics!) This book helped me truly understand the purpose of the object-oriented programming paradigm and why Java can be very useful. It's too difficult a book for someone entirely new to programming, but if you have several languages already under your belt, I highly recommend it.


## Online Books (free!)

I have found a couple of books that you can get for free. I highly recommend both.

### The Field Guide to Data Science from [Booz Allen Hamilton](http://www.boozallen.com/insights/2013/11/data-science-field-guide) <font color = 'red'>(difficult)</font> 

<!---
<img src = "/images/FGuide_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/FGuide_Book.jpg){:align="right" width="128px" height="168px"}
This gives advice from the data science team at BAH on how they like to approach problems. It is very colorful and well organized. The contents span a variety of subjects, such as hiring data science teams, what algorithms work best in certain situations, and case studies from problems they actually had to solve. It gives a great insight into how data scientists think.

### Elements of Statistical Learning by [Hastie, Tibshirani, and Friedman](http://statweb.stanford.edu/~tibs/ElemStatLearn/)
<font color = 'purple'>(brace yourself . . .)</font>
<!---
<img src = "/images/ESL_Book.jpg" align = right width = 128 height = 168>
-->
![](/images/ESL_Book.jpg){:align="right" width="128px" height="168px"}
Anyone who is a really hard-core data scientist should have read this book. It describes all of the mathematics and minute details for almost every major supervised and unsupervised machine learning or statistical algorithm there is. It's not enough to just use these algorithms in R, Mahout, or scikit-learn and go on your merry way. To use them properly, it is important to understand how they really work and what the internals look like. Yes, it is a challenging book to read. There is a lot of advanced mathematics in it, and it is very long (over 700 pages!). If you don't have a strong math background, it will be intimidating to you. Tough. Read it anyway, all of it! If it makes you feel any better, the authors themselves (who are brilliant!) have realized the book is very technical and have added a little icon of a yellow man with hands in the air to mark sections they feel are really difficult. The authors claim you can skip these sections if you wish, but I would try to read them regardless. It is truly considered a classic for many data scientists and machine learning experts.


