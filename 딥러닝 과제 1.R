setwd('/Volumes/GoogleDrive/내 드라이브/학교 수업/20-1학기/딥러닝')
df = read.csv('stock.csv',sep=",",header =TRUE)
# 1.
#### a. ####
df$Jan = as.numeric(df$Month == 1)
df$Feb = as.numeric(df$Month == 2)
df$Mar = as.numeric(df$Month == 3)
df$Apr = as.numeric(df$Month == 4)
df$May = as.numeric(df$Month == 5)
df$Jun = as.numeric(df$Month == 6)
df$July = as.numeric(df$Month == 7)
df$Aug = as.numeric(df$Month == 8)
df$Sep = as.numeric(df$Month == 9)
df$Oct = as.numeric(df$Month == 10)
df$Nov = as.numeric(df$Month == 11)
x = cbind(df$Jan,df$Feb,df$Mar,df$Apr,df$May,df$Jun,df$July,df$Aug,df$Sep,df$Oct,df$Nov)
x

#### b. ####
X = cbind(1, df$Interest, df$Unemployment,df$Jan,df$Feb,df$Mar,df$Apr,df$May,df$Jun,df$July,df$Aug,df$Sep,df$Oct,df$Nov)
X
y = Stock

#### c. ####
beta.hat <- solve(t(X)%*%X)%*%t(X)%*%y 
beta.hat

sigmasq.hat <- as.numeric( t(y-X%*%beta.hat)%*%(y-X%*%beta.hat)/(24-14) ) # (n-p-1)
sqrt(  diag(solve(t(X)%*%X))*sigmasq.hat  ) # standard error of reg

#### d. ####
model <- lm(Stock ~ Interest + Unemployment + Jan + Feb + Mar + Apr + May + Jun + July + Aug + Sep + Oct + Nov )
summary(model)

## result is quite same. 

# 2. 
detach(df)
admissions <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
head(admissions)

#### a. ####
fit=glm(admit~gpa,family='binomial', data = admissions)
summary(fit)

#### b. ####
xvals=runif(200, min=2, max=4)
xvals = sort(xvals)
xvals
newdata=data.frame(gpa=xvals)
newdata
eta=predict(fit,newdata=newdata,type="link")
eta

#### c. ####
mu=predict(fit,newdata=newdata,type="response")
mu

#### d. ####
par(mfrow=c(1,2))
plot(xvals,eta,main="Linear Predictor",xlab="gpa",ylab=expression(eta),type="l")
plot(xvals,mu,main="Mean Response as a Function of the Predictor",xlab="gpa",ylab=expression(mu),ylim=c(0,1),type="l",lwd=3) 
points(jitter(admissions$gpa),admissions$admit)

# we can see that B is positve, so B and y value has a positve relationship. As in the frist plot, it shows that when gpa is increaed, linear predictor also increased. 
# In the second plot, circle is a real value about y, as we can see that when gpa is increased, expectation value y given x also increased. 


