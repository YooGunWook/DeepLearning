# 1. 


# (a)
library(car)

N=150
P=50
X=matrix(NA,nrow=N,ncol=P)
dim(X)
covmat=matrix(rnorm(P^2,sd=2),nrow=P)
covmat=covmat+t(covmat)
U=eigen(covmat)$vectors
D=diag(rexp(P,rate=10))
covmat=U%*%D%*%t(U)
covmat

library(mvtnorm)
for(i in 1:N){
  X[i,]=rmvnorm(1,mean=rep(0,P),sigma=covmat)
}
X=data.frame(X)
head(X)
dim(X)

betas.true=c(1,2,3,4,5,-1,-2,-3,-4,-5,rep(0,P-10))
betas.true

sigma=15.7
X=as.matrix(X)
y=X%*%betas.true+rnorm(N,mean=0,sd=sigma)
y

alldata=data.frame(cbind(y,X))
names(alldata)[1] <- "y"
head(alldata)
train=alldata[1:100,]
test=alldata[101:150,]
dim(train)
dim(test)

fit=lm(y~.,data=train)
summary(fit)
vif(fit)

betas.lm=coef(fit)
betas.lm

yhat.lm=predict(fit,newdata=test)
mspe.lm=mean((test$y-yhat.lm)^2)
mspe.lm
# vif가 10 이상이면 다중공선성이다. 몇몇 변수가 10 넘는 것을 보면 다중공선성이 있는 데이터라는 것을 알 수 있다.

# (b)
library(glmnet)
## alpha=0 gives ridge regression
## alpha=1 gives lasso regression

## fit ridge (trying 100 different lambda values)
rr=glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0,nlambda=100)
plot(rr,xvar="lambda",main="Ridge Regression Betas for Different Values of the Tuning Parameter")

## use 10-fold crossvalidation to find the best lambda
cv.rr=cv.glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0,nfolds=10,nlambda=100)
# k-fold validation을 써서 최적의 람다를 구한다. 
cv.rr

## getting cvmspe from best value of lambda
cvmspe.rr=min(cv.rr$cvm)
cvmspe.rr


## get lambda and best rr fit
lambda.rr=cv.rr$lambda.min
lambda.rr
log(lambda.rr)

## some plots
par(mfrow=c(1,2))
plot(cv.rr)
abline(v=log(lambda.rr))
plot(rr,xvar="lambda",main="Ridge Regression Betas for Different Values of the Tuning Parameter")
abline(v=log(lambda.rr))

## beta estimates for best lambda
betas.rr=coef(cv.rr,s="lambda.min")
betas.rr
betas.lm
# coefficient가 0인 것은 없다. 

plot(betas.rr,betas.lm,xlim=c(-6,6),ylim=c(-6,6))
abline(0,1)
# penalty를 넣어주니 smaller value가 된다. 

yhat.rr=predict(cv.rr,s="lambda.min",newx=as.matrix(test[,-1]))
mspe.rr=mean((test$y-yhat.rr)^2)
mspe.rr


# (C)

## alpha=0 gives ridge regression
## alpha=1 gives lasso regression

## fit lasso (trying 100 different lambda values)
lasso=glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=1,nlambda=100)
plot(lasso,xvar="lambda",main="Lasso Regression Betas for Different Values of the Tuning Parameter")
plot(rr,xvar="lambda",main="Ridge Regression Betas for Different Values of the Tuning Parameter")

## use 10-fold crossvalidation to find the best lambda
cv.lasso=cv.glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=1,nfolds=10)
cv.lasso

## get lambda and best lasso fit
lambda.lasso=cv.lasso$lambda.min
lambda.lasso
log(lambda.lasso)

## getting cvmspe from best value of lambda
cvmspe.lasso=min(cv.lasso$cvm)

## some plots
par(mfrow=c(1,2))
plot(cv.lasso)
abline(v=log(lambda.lasso))
plot(lasso,xvar="lambda")
abline(v=log(lambda.lasso))

## beta estimates for best lambda
betas.lasso=coef(cv.lasso,s="lambda.min")
betas.lasso
betas.lm

yhat.lasso=predict(cv.lasso,newx=as.matrix(test[,-1]),s="lambda.min")
mspe.lasso=mean((test$y-yhat.lasso)^2)
mspe.lasso

# (d)

mspe.lm
mspe.rr
mspe.lasso

# (e)
# lasso는 correlation이 높은 데이터를 없애주는 특징이 있다. 수업에서 사용한 데이터는 correlation이 없는 데이터였기 때문에 beta가 0이 아닌 값들이 많이 남은 것을 확인할 수 있었다.
# 그러나 이번에는 multicolinearity가 강하게 있는 데이터였기 때문에 vif가 높은 데이터들이 많이 삭제된 것을 볼 수 있다. 

# (f)

N=150
P=50

X=matrix(NA,nrow=N,ncol=P)
dim(X)
covmat=matrix(rnorm(P^2,sd=2),nrow=P)
covmat=covmat+t(covmat)
U=eigen(covmat)$vectors
D=diag(rexp(P,rate=10))
covmat=U%*%D%*%t(U)
covmat

library(mvtnorm)
for(i in 1:N){
  X[i,]=rmvnorm(1,mean=rep(0,P),sigma=covmat)
}
X=data.frame(X)
head(X)
dim(X)


## true betas
betas.true=c(rep(2,10),rep(0,P-10))

## simulating "y"
X=as.matrix(X)


# poisson으로 나온다.
y=as.integer(X%*%betas.true+rpois(N, lambda = 10))
y

## lasso
lasso=glmnet(x=X,y=y, family="poisson", alpha=1,nlambda=100)
## use 10-fold crossvalidation to find the best lambda
cv.lasso=cv.glmnet(x=X,y=y,alpha=1,nfolds=10)

## get lambda and best lasso fit
lambda.lasso=cv.lasso$lambda.1se
log(lambda.lasso)

## some plots
par(mfrow=c(1,2))
plot(cv.lasso)
abline(v=log(lambda.lasso))
plot(lasso,xvar="lambda")
abline(v=log(lambda.lasso))

## beta estimates for best lambda
betas.lasso=coef(cv.lasso)
betas.lasso
rm(list = ls())


# 2. Consider the wines data that we covered in the lecture. 
# Conduct Kohonen’s SOM by varying dimension of output (2 × 2, 4 × 4, 6 × 6, 10 × 10).
# Explain how it changes with increasing number of grids and what do you think is the appropriate number of grids? 
# Can you fit SOM for 14 × 14 grids for this example? If not why? (20 point)
library("kohonen")
data("wines")
str(wines)
head(wines)
View (wines)

# scale 하는 것에 다양한 방법이 있다. 
# scale은 자신이 원하는 방법으로 하면 된다. 

# 2 X 2
set.seed(1)
som.wines = som(scale(wines), grid = somgrid(2, 2, "hexagonal"))
som.wines
dim(getCodes(som.wines))
par(mfrow = c(1, 2))
plot(som.wines, main = "Wine data Kohonen SOM")
plot(som.wines, type = "changes", main = "Wine data: SOM")

# 4 X 4
set.seed(1)
som.wines = som(scale(wines), grid = somgrid(4, 4, "hexagonal"))
som.wines
dim(getCodes(som.wines))
par(mfrow = c(1, 2))
plot(som.wines, main = "Wine data Kohonen SOM")
plot(som.wines, type = "changes", main = "Wine data: SOM")

# 6 X 6
set.seed(1)
som.wines = som(scale(wines), grid = somgrid(6, 6, "hexagonal"))
som.wines
dim(getCodes(som.wines))
par(mfrow = c(1, 2))
plot(som.wines, main = "Wine data Kohonen SOM")
plot(som.wines, type = "changes", main = "Wine data: SOM")

# 10 X 10
set.seed(1)
som.wines = som(scale(wines), grid = somgrid(10, 10, "hexagonal"))
som.wines
dim(getCodes(som.wines))
par(mfrow = c(1, 2))
plot(som.wines, main = "Wine data Kohonen SOM")
plot(som.wines, type = "changes", main = "Wine data: SOM")



## 과제와는 연관 없음.
training = sample(nrow(wines), 150)
Xtraining = scale(wines[training, ])
Xtest = scale(wines[-training, ],
              center = attr(Xtraining, "scaled:center"),
              scale = attr(Xtraining, "scaled:scale"))
trainingdata = list(measurements = Xtraining, vintages = vintages[training])
testdata = list(measurements = Xtest, vintages = vintages[-training])

mygrid = somgrid(5, 5, "hexagonal")
# multiple level일 때 쓰면 좋다.
som.wines = supersom(trainingdata, grid = mygrid)
som.prediction = predict(som.wines, newdata = testdata)

table(vintages[-training], som.prediction$predictions[["vintages"]])







