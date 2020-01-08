# Contents
1. Baseline.ipynb - This python notebook contains the preminilary analysis of our data and our baseline model that we would be using for the rest of the project. 
2. Requirements.txt - This file contains the libraries that are required to run our project and also mentions other requirements needed to keep in mind before running the project. 
3. MovieLens Data Recommendation System - Report.pdf - This file contains our entire report.
4. KNN.ipynb - This python notebook contains the item-item and user-user based recommendation system that uses kNN algorithm to find the nearest neighbors. 
5. ALS.ipynb - This python notebook contains the model based approach to build the recommendation system. The model we have used is ALS and have implemented it using Spark. 

## Introduction 
In today’s world, where data has become the new oil, recommendation systems has taken a huge leap in terms of research and real time implementation.
With the growing technology industry, the amount of competition between these companies to retain users has grown tremendously. Everyone is behind personalizing their experiences based on the user so that the user actually comes back to their side.
What is stopping a user from simply switching tabs and going for movie recommendations to Netflix or Imdb? What does it take to actually understand the users, keeping in mind his/her preferences, and recommending them the right movie? Keeping in mind one of the most important business goals, wherein, the company needs to make some profits out of it. If the company doesn’t make profits by implementing the recommendation systems, what was the point of doing it in the first place?
So let’s try and understand what we are trying to do here.
We have been given this data where we have ratings of over x users for y movies. Using this rating dataset we will be trying to recommend movies to users.

## Data Analysis 
After doing some market research, we have come to the conclusion that the most active users (that is the users that have rated more than z number of movies) have a lot of influence on the recommendation of movies. Their ratings are more important according to us, as they have rated a lot of movies. Also, the influence of the users who haven’t rated many movies would be very less as they wouldn’t be generalizing the trends that we need to consider while suggesting a movie to one user.
The movies also follow a similar trend as the popular ones would be having high frequency of being rated. We only take into consideration such movies which have high frequency of getting rated.

## Business Rule
Let's take a minute here to understand how our recommendation systems are actually recommending movies to the user, that is the business rule we have considered here.
After each of the implementations of the recommendation system, we will have the predicted rating values of the movies that the user would have given to the movie had he seen it. This predicted value is found based on our implementations explained above. The predicted values are for all movies the user hasn’t seen before. From these predicted rating values, we will find the top 10 movies that we would recommend based on these values. If this value is greater than or equal to 3.5, then we assume that the user would like the movie. We only recommend the movies that the user likes (that is the ones with ratings 3.5 or greater).
We have selected 3.5 as the threshold value here as that is the mean movie rating of our dataset. Keeping this in mind, let's move forward.
Next, let's talk about movies. We want to recommend the movies that our users will actually like, so that they come back to our website for more recommendations.
To simply solve this problem, let's assume we recommend the top 10 most popular movies.
You might ask, how do we define “popular” in this case? For this problem, we have defined popular as the movies that have been recommended more than a number of times and then we sort these movies based on their mean rating. The top 10 movies that we get in the end, are the top 10 most popular movies.
Now, we think that just recommending the top 10 most popular movies is not a good thing to do, as we are not considering the users preferences in this case and also that we are letting our data science skills go in vain.
So what we try to do here, is that we build 2 recommendation systems based on our knowledge of collaborative filtering.
The first one is a memory based recommendation system where we find similar items/users using k-NN (k nearest algorithms) in which similarity is used to determine the nearest neighbours .
The second one is a model based approach where we are using ALS (alternating least squares) model to get the ratings and use those eventually to recommend movies.

## Objective
As we have explained above,
1. We care more about the movies that have been rated more (as these are the movies that have been seen more than others, so the chances of a user liking them would be higher).
2. We care about the users that have rated more than a certain number of movies (as these are our active users).
We are trying to make sure that the users get the recommendations that they will like, keeping in mind the popular movies (so that we decrease the risk of recommending the less popular movies).
Next, to know if we have built a good recommendation system, we first created a baseline and noted its accuracy. We calculate the baseline for the two evaluation metrics: RMSE and R-Squared.
For the baseline for RMSE, we take the mean rating for each of the four samples as the predicted values and calculate the baseline RMSE for the corresponding samples.
For the baseline for R-squared, similar to RMSE we take the mean rating for each of the four samples as the predicted values and hence the R-Squared values is zero. Therefore, the baseline is the horizontal line i.e (y = 0) in this case.

## Recommendation Systems
Now let's talk about the systems that we have built -
1. Memory based recommendation system using KNN  - For this case we use different kinds of similarity metrics to calculate the similarity and hence, find the “k” nearest neighbours for the user/item that are into consideration. Based on the average of those neighbours’ ratings we predict the ratings for a user. We built both user-user similarity model as well as item-item similarity model.
As expected the item-item model performed better than the user-user one and is appropriate for the case we have in hand as adding items to the database makes more sense than adding a new user and computing similarity amongst each of the existing ones. The items being lesser in magnitude makes sense to be compared with each other for computing the similarity.
2. Model based recommendation system using ALS  - In this case we are using the spark MlLib library used for collaborative filtering that is ALS. Using the API given, we are trying to recommend 10 movies to each of the users.
First we run our model on the sample dataset. Based on the rmse values (we will explain this in just a minute) we are trying to tune the parameters of the ALS model.
The different parameters that we consider here are - Number of iterations (numIters), Rank and Regression parameter (lambda)
After finding the best values of the parameters using cross validation, we use different sample sizes
to increase the size of our datasets and answer some important questions. In the real-world, the data we have would be much more than this, so we need to make sure that we have made our model good enough to work on the large dataset.

## Evaluation Metrics
As we promised above, before understanding anything else, lets see how we have evaluated the accuracy of our systems.
To know if we have built a good recommendation system, we first created a baseline and noted its accuracy. You must be wondering, how we have defined accuracy here.
We have used to evaluation metrics to define the accuracy -
1. RMSE (Root mean squared error) - Primary
2. R2 (R - squared) - Secondary

## Hyperparameter Tuning
After we have understood the coverage of our systems, we will go on to tune our parameters for the different samples.
We have 4 samples into consideration here -
1. Take the ratings of the top 500 movies, from users that have rated more than 1500 movies
2. Take the ratings of the top 1000 movies, from users that have rated more than 1500 movies 3. Take the ratings of the top 1500 movies, from users that have rated more than 1500 movies 4. Take the ratings of the top 2000 movies, from users that have rated more than 1500 movies

## Coverage
Because of our business rule (explained in the beginning), we might end up not recommending some movies and we also might end up not recommending 10 movies for some users. Here, we introduce the coverage concept, where we try and understand how much of the sample dataset are we actually covering.
There are two types of coverage that we have found -
1. User coverage = The number of users that we are recommending 10 movies / The total number of users in our sample.
2. Catalogue coverage(items) = The number of unique movies recommended to all users / The total number of movies in our sample.
