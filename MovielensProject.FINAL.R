# Capstone Edx - Movielens Project
# Author: Francielle Mina 


# This is a part Data Science Professional Certificate - Movielens Projects
# Here is only the code for the project. 

# 3.1 Explore dataset Edx.

# MovieLens 10M dataset:
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(RColorBrewer)
#https://grouplens.org/datasets/movielens/10m/
#http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# 3.1 EDA - Exploring data set
str(edx)

head(edx)

dim(edx)

summary(edx)

# 3.1.2 Data analysis Edx data


# 3.1.2.1 Distribuition the Movies by ratings 


library(RColorBrewer)
edx %>% group_by(movieId) %>% summarise(n = n()) %>% ggplot(aes(n)) + geom_histogram(color = "black", bins = 20, fill = "#66CCFF") + ggtitle("Distribuition the Movies by rating") +
  scale_x_log10() + xlab("Number of Rating") + ylab("Number of Movies") 

n_distinct(edx$movieId)

edx %>% group_by(movieId) %>% summarise(n = n()) %>% head()

# 3.1.2.2 Analysis of Distribuition by Users


edx %>% group_by(userId) %>% summarise(n = n()) %>% ggplot(aes(n)) + geom_histogram(color = "black", bins = 20, fill = "#66CCFF") + ggtitle("Distribuiton of Users") + scale_x_log10() + xlab("Number of Ratings") + ylab("Number of Users")

n_distinct(edx$userId)

edx %>% group_by(userId) %>% summarise(n = n()) %>% head()


# 3.1.2.3 Analysis of Ratings 

edx %>% group_by(rating) %>% summarise(n = n()) %>% ggplot(aes(rating, n)) + geom_line(color = "#66CCFF") + geom_point() + scale_x_log10() + ggtitle("Rating Distribuition") + 
  xlab("Rating") + ylab("Count")

n_distinct(edx$rating)

edx %>% group_by(rating) %>% summarise(n = n()) %>% head()

# 3.2.1 Explore datset on Validation 


str(validation)

head(validation)

dim(validation)

summary(validation)

# 3.2.2 Data analysis on Validation dataset

# 3.2.3  Analysis of Distribuition by Movies 


validation %>% group_by(movieId) %>% summarise(n = n()) %>% ggplot(aes(n)) + geom_histogram(color = "black", bins = 20, fill = "#009999") + ggtitle("Distribuition the Movies by rating", subtitle = "Validation dataset") +
  scale_x_log10() + xlab("Number of Rating") + ylab("Number of Movies")

n_distinct(validation$movieId)

# 3.2.4  Analysis of Distribuition by Users


validation %>% group_by(userId) %>% summarise(n = n()) %>% ggplot(aes(n)) + geom_histogram(color = "black", bins = 20, fill = "#009999") + ggtitle("Distribuition the Users", subtitle = "Validation dataset") +
  scale_x_log10() + xlab("Number of Rating") + ylab("Number of Movies")

n_distinct(validation$userId)

# 3.2.5 Analysis of Ratings 


validation %>% group_by(rating) %>% summarise(n = n()) %>% ggplot(aes(rating, n)) + geom_line(color = "#009999") + geom_point() + scale_x_log10() + ggtitle("Rating Distribuition") + 
  xlab("Rating") + ylab("Count")

n_distinct(validation$rating)

validation %>% group_by(rating) %>% summarise(n = n()) %>% head()


# 4.0 Results

# For this step, we are split the edx data set into train and test. The train data set has 10%, and the test has 90% of the original data. Also, it is an important method that evaluates the accuracy of the dataset. As explained before, train the part of data to allow the algorithm to predict the outcome. 

# 4.1 Recommendations system


set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# make sure userId and movieId in validation set are also in edx set 
test_set <- temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, test_set)

train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# This next step is an initial preparation. Let's check the initial RMSE. 

mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

rmse_results <- tibble(method = "Naive RMSE", RMSE = naive_rmse)
rmse_results

# 4.1.2 Effect on Movie (b_i)

mu <- mean(train_set$rating) 
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))


b_i %>% ggplot(aes(b_i)) + geom_histogram(bins = 20, fill = "#3300CC", color = "black") + ggtitle("Distribuition of Movie effect") + xlab("Distribuition of Movie effect") +
  ylab("Count")


# 4.1.3 predict moveis on test set 


predicted_b_i <- mu + test_set %>% 
  left_join(b_i, by='movieId') %>%
  pull(b_i)
rmse_1 <- RMSE(predicted_b_i, test_set$rating)
rmse_1 

rmse_results1 <-tibble(method = "Movie effect model on test set", RMSE = rmse_1)
rmse_results1
rmse_results1 %>% knitr::kable()

# 4.1.3 User effects on b_u

b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

b_u %>% ggplot(aes(b_u)) + geom_histogram(bins = 20, fill = "#3300CC", color = "black") + ggtitle("User effect on distribuition") +
  xlab("Users") + ylab("Count")


# 4.1.4 predict values on test set (movie and user)


predicted_b_u <- test_set %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_2 <- RMSE(predicted_b_u, test_set$rating)
rmse_2


rmse_results2 <- tibble(method="Movie + User Effects Model", RMSE = rmse_2)
rmse_results2 %>% knitr::kable()


# 5.0 Regularisation 

# Before run regularisation, let's check the b_i effect on the movie. 


# 5.1 Effect of movie. 

titles <- train_set %>% 
  select(movieId, title) %>% 
  distinct()
titles

b_i %>% 
  inner_join(titles, by = "movieId") %>% 
  arrange(b_i) %>% 
  select(title) %>%
  slice(1:10) 

# Here effect on b_i on movies.  List 10 worst movies. 



b_i %>% 
  inner_join(titles, by = "movieId") %>% 
  arrange(-b_i) %>% 
  select(title) %>%
  slice(1:10)

# Here Effect on b_i on 10 betters movies. 

# 5.2 Regularisation and Lambda

# Below is the regularization function to choose the best value that minimizes the RMSE.


lambda <- seq(0, 10, 0.25)

regularisation <- sapply(lambda, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- train_set %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% .$pred
  return(RMSE(train_set$rating, predicted_ratings))
})

regularisation


qplot(lambda, regularisation)

# This graph allows visualising the lambda. The minimum of lambda is 3.0. 

lambdas <- lambda[which.min(regularisation)]
lambdas


# 6.0 Build the third methodo with Regularisation 

# Compute movie effect with regularisation on the train set. The lambda chosen is 3.0, and the next step will see the effect on bi and bu. 
# bi(movie+regularisation) 
# bu(user+regularisation)


# 6.1 Movie (bi) + Regularisation


bi <- train_set %>% group_by(movieId) %>% summarise(bi = sum(rating - mu)/(n()+lambdas))
bi

bi %>% ggplot(aes(bi)) + geom_histogram(bins = 20, fill = "#336699", color = "black") + ggtitle("Effect of regularisation on Movies distribution") +
  xlab("Movies") + ylab("Count")


# 6.2 Compute user effect with regularisation on trainset 

bu <- train_set %>% left_join(bi, by = "movieId") %>% group_by(userId) %>% summarise(bu = sum(rating - bi - mu)/(n()+lambdas))
bu

bu %>% ggplot(aes(bu)) + geom_histogram(bins = 20, fill = "#336699", color = "black") + ggtitle("Effect of regularisation on Users distribution") +
  xlab("Users") + ylab("Count")


# 6.3 Compute predicted value on teste set 


predict_bi_bu <- test_set %>% left_join(bi, by = "movieId") %>% left_join(bu, by = "userId") %>%
  mutate(pred = mu + bi + bu) %>% .$pred


rmse_3 <- RMSE(test_set$rating, predict_bi_bu)
rmse_3

rmse_results_3 <- tibble(method = "Regularisation movie and user effect model on test set", RMSE = rmse_3)
rmse_results_3

# 7.0 Final Validation 


mu_edx <- mean(edx$rating)
mu_edx


# 7.1 Movie effect (bi) 


bi_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu_edx)/(n()+lambdas)) 


# 7.2 User effect (bu)

bu_edx <- edx %>% 
  left_join(bi_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - bi - mu_edx)/(n()+lambdas))  


# 7.3 Prediction on validation set 


prediction_edx <- validation %>% 
  left_join(bi_edx, by = "movieId") %>% 
  left_join(bu_edx, by = "userId") %>% 
  mutate(pred = mu_edx + bi + bu) %>%.$pred

rmse_4 <- RMSE(validation$rating, prediction_edx)
rmse_4

rmse_results4 <- tibble(method = "Final regularisation, edx vs validation", RMSE = rmse_4)
rmse_results4

