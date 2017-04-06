library(readr)
test <- read_csv("~/ML2017/hw2/test.csv")
train <- read_csv("~/ML2017/hw2/train.csv")
X_train <- read_csv("~/ML2017/hw2/X_train.csv")
Y_train <- read_csv("~/ML2017/hw2/Y_train.csv", col_names = FALSE)
X_test <- read_csv("~/ML2017/hw2/X_test.csv")

X_train <- X_train[,c('capital_gain','Preschool','Married-civ-spouse','Husband','Own-child')]
X_test <- X_test[,c('capital_gain','Preschool','Married-civ-spouse','Husband','Own-child')]

X_train_mat <- as.matrix(X_train)
X_test_mat <- as.matrix(X_test)

the_train_mean <- colMeans(X_train_mat)
the_train_std <- apply(X_train_mat, MARGIN=2, FUN=sd)

mean_mat <- matrix(the_train_mean, ncol=ncol(X_train_mat), nrow=nrow(X_train_mat), byrow = TRUE)
std_mat <- matrix(the_train_std, ncol=ncol(X_train_mat), nrow=nrow(X_train_mat), byrow = TRUE)

the_train <- X_train_mat - mean_mat
the_train <- the_train/std_mat

the_train <- cbind(the_train, Y_train)
names(the_train)[ncol(the_train)] = 'label'



model <- glm(label ~.,family=binomial(link='logit'),data=the_train)
train_res <- predict(model,the_train,type='response')

acc <- c(Y_train$X1) - c(round(train_res))
acc <- acc^2
acc <- 1 - sum(acc)/length(acc)
print(acc)

res <- predict(model,X_test,type='response')
res <- round(res)
res[res>1] = 1
res1 <- res
res_df <- data.frame(id = 1:16281, label = res)
# write.csv(res_df, file='res.csv', row.names = FALSE)



y = matrix(the_train$label, ncol=1)
xmat = cbind(rep(1,nrow(the_train)), the_train[,1:ncol(the_train)-1])
names(xmat)[1] = 'coeff'
xmat = as.matrix(xmat)
bhead = solve(t(xmat) %*% xmat) %*% t(xmat) %*% y

train_res <- round(xmat %*% bhead)
acc <- c(y) - c(round(train_res))
acc <- acc^2
acc <- 1 - sum(acc)/length(acc)
print(acc)


mean_mat <- matrix(the_train_mean, ncol=ncol(X_train_mat), nrow=nrow(X_test_mat), byrow = TRUE)
std_mat <- matrix(the_train_std, ncol=ncol(X_train_mat), nrow=nrow(X_test_mat), byrow = TRUE)

the_test <- X_test_mat - mean_mat
the_test <- the_test/std_mat


test_matrix = as.matrix(cbind(rep(1,nrow(the_test)), the_test))
res <- test_matrix %*% bhead
res <- round(res)
res[res>1] = 1
res2 <- res
res_df <- data.frame(id = 1:16281, label = res)
write.csv(res_df, file='res.csv', row.names = FALSE)


print(sum((res1 -res2)^2)/length(res1))
