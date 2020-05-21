# 1. 우선 sigmoid 함수를 통해 output을 구해본다. 

### define functions
sigmoid = function(z){ 
  return( 1/(1+exp(-z)) )
}


# forwardPropagation code
forwardProp = function(input, w, b){
  # input to hidden layer
  neth1 = w[1]*input[1] + w[2]*input[2] + b[1]
  neth2 = w[3]*input[1] + w[4]*input[2] + b[1]
  outh1 = sigmoid(neth1)
  outh2 = sigmoid(neth2)
  
  # hidden layer to output layer
  neto1 = w[5]*outh1 + w[6]*outh2 + b[2]
  neto2 = w[7]*outh1 + w[8]*outh2 + b[2]
  outo1 = sigmoid(neto1)
  outo2 = sigmoid(neto2)
  
  res = c(outh1, outh2, outo1, outo2)
  return(res)
}


backProp = function(res, out, input, gamma){
  outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
  
  
  # 에러에 대해서 outo1으로 편미분 한 후 w5와 w6에 대해서 BackProb를 해준다. 
  dE_douto1 = -( out[1] - outo1 )
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1*douto1_dneto1*dneto1_dw5
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1*douto1_dneto1*dneto1_dw6
  
  # 에러에 대해서 outo2으로 편미분 한 후 w7과 w8에 대해서 BackProb를 해준다. 
  dE_douto2 = -( out[2] - outo2 )
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2*douto2_dneto2*dneto2_dw7
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2*douto2_dneto2*dneto2_dw8
  
  # Bias를 구하는 식 -> 과제에서는 따로 계산 안해도 된다고 했다. 
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  
  # neto1을 outh1으로 편미분하면 w5가 나오고, net02를 outh1으로 편미분하면 w7이 된다.
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  # 우선 에러를 outh1으로 편미분 한 값을 구해준다. 
  dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
  
  # neto1을 outh2으로 편미분하면 w6가 나오고, net02를 outh2으로 편미분하면 w8이 된다.
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  # 우선 에러를 outh2로 편미분 한 값을 구해준다. 
  dE_douth2 = dE_douto1*douto1_dneto1*dneto1_douth2 + dE_douto2*douto2_dneto2*dneto2_douth2 
  
  
  # outh1에 대해서 neth1으로 편미분 해준다.
  douth1_dneth1 = outh1*(1-outh1)
  # neth1을 w1으로 편미분하면 input값이 나온다.
  dneth1_dw1 = input[1]
  # 이를 통해 w1을 backProb 해준다.
  dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
  
  # 위와 비슷하게 backProb 해준다. 
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1*douth1_dneth1*dneth1_dw2
  
  # outh2를 neth2로 편미분 해준다. 
  douth2_dneth2 = outh2*(1-outh2)
  # heth2를 w3로 편미분하면 input값이 나온다.
  dneth2_dw3 = input[1] 
  # 이를 통해 w3로 편미분 해준다. 
  dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
  
  # 위와 비슷하게 backProb 해준다. 
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2*douth2_dneth2*dneth2_dw4  
  
  # b1에 대한 편미분 값. 여기서는 고려하지 않아도 된다. 
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  # weight를 업데이트 해준다.
  w1 = w1 - gamma*dE_dw1
  w2 = w2 - gamma*dE_dw2
  w3 = w3 - gamma*dE_dw3
  w4 = w4 - gamma*dE_dw4
  w5 = w5 - gamma*dE_dw5
  w6 = w6 - gamma*dE_dw6
  w7 = w7 - gamma*dE_dw7
  w8 = w8 - gamma*dE_dw8
  b1 = b1 - gamma*dE_db1
  b2 = b2 - gamma*dE_db2    
  
  # 값 갱신
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
  return(list(w,b))
}

# 강의와 같은 방식으로 진행한다.

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

input1 = 0.05
input2 = 0.10 
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

gamma = 0.5

a = forwardProp(input,w,b)
w = backProp(a, out, input, gamma)
# weight
w[[1]]

# bias
w[[2]]

# 2. 교수님 코드를 이용해서 똑같이 나왔는지 확인 
# 교수님의 코드는 iteration을 1000번 하고 있기 때문에 정확히 확인해볼 순 없지만 정확히 w1과 w5를 비교해보면 정확히 일치하는 것을 볼 수 있다. 따라서 이 계산은 정확하다고 볼 수 있다.

# 3. learning rate를 다르게 해보자. (0.1, 0.6, 1.2) 그리고 iteration을 각 10000번씩 해주자. 

numIter = 10000

### Initial settings 

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

# input and target values
input1 = 0.05
input2 = 0.10 
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

### define functions
sigmoid = function(z){ 
  return( 1/(1+exp(-z)) )
}

forwardProp = function(input, w, b){
  # input to hidden layer
  neth1 = w[1]*input[1] + w[2]*input[2] + b[1]
  neth2 = w[3]*input[1] + w[4]*input[2] + b[1]
  outh1 = sigmoid(neth1)
  outh2 = sigmoid(neth2)
  
  # hidden layer to output layer
  neto1 = w[5]*outh1 + w[6]*outh2 + b[2]
  neto2 = w[7]*outh1 + w[8]*outh2 + b[2]
  outo1 = sigmoid(neto1)
  outo2 = sigmoid(neto2)
  
  res = c(outh1, outh2, outo1, outo2)
  return(res)
}

error = function(res, out){ 
  err = 0.5*(out[1] - res[3])^2 + 0.5*(out[2] - res[4])^2 
  return(err)
}


### Implement Forward-backward propagation
gamma = 0.1
err_1 = c()

for(i in 1:numIter){
  
  ### forward
  res = forwardProp(input, w, b)
  outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
  
  ### compute error
  err_1[i] = error(res, out)
  
  ### backward propagation
  ## update w_5, w_6, w_7, w_8, b2 
  # compute dE_dw5
  dE_douto1 = -( out[1] - outo1 )
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1*douto1_dneto1*dneto1_dw5
  
  # compute dE_dw6
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1*douto1_dneto1*dneto1_dw6
  
  # compute dE_dw7
  dE_douto2 = -( out[2] - outo2 )
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2*douto2_dneto2*dneto2_dw7
  
  # compute dE_dw8
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2*douto2_dneto2*dneto2_dw8
  
  # compute dE_db2
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  ## update w_1, w_2, w_3, w_4, b1 
  # compute dE_douth1 first
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
  
  # compute dE_douth2 first
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  dE_douth2 = dE_douto1*douto1_dneto1*dneto1_douth2 + dE_douto2*douto2_dneto2*dneto2_douth2 
  
  # compute dE_dw1    
  douth1_dneth1 = outh1*(1-outh1)
  dneth1_dw1 = input[1]
  dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
  
  # compute dE_dw2
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1*douth1_dneth1*dneth1_dw2
  
  # compute dE_dw3
  douth2_dneth2 = outh2*(1-outh2)
  dneth2_dw3 = input[1] 
  dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
  
  # compute dE_dw4
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2*douth2_dneth2*dneth2_dw4  
  
  # compute dE_db1
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  ### update all parameters via a gradient descent 
  w1 = w1 - gamma*dE_dw1
  w2 = w2 - gamma*dE_dw2
  w3 = w3 - gamma*dE_dw3
  w4 = w4 - gamma*dE_dw4
  w5 = w5 - gamma*dE_dw5
  w6 = w6 - gamma*dE_dw6
  w7 = w7 - gamma*dE_dw7
  w8 = w8 - gamma*dE_dw8
  b1 = b1 - gamma*dE_db1
  b2 = b2 - gamma*dE_db2    
  
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
  print(i)
  
}

ts.plot( err_1)

pred_0.1 = forwardProp(input, w, b)
pred_0.1[3:4]

numIter = 10000

### Initial settings 

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

# input and target values
input1 = 0.05
input2 = 0.10 
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)


### gamma 0.6
gamma = 0.6
err_2 = c()

for(i in 1:numIter){
  
  ### forward
  res = forwardProp(input, w, b)
  outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
  
  ### compute error
  err_2[i] = error(res, out)
  
  ### backward propagation
  ## update w_5, w_6, w_7, w_8, b2 
  # compute dE_dw5
  dE_douto1 = -( out[1] - outo1 )
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1*douto1_dneto1*dneto1_dw5
  
  # compute dE_dw6
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1*douto1_dneto1*dneto1_dw6
  
  # compute dE_dw7
  dE_douto2 = -( out[2] - outo2 )
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2*douto2_dneto2*dneto2_dw7
  
  # compute dE_dw8
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2*douto2_dneto2*dneto2_dw8
  
  # compute dE_db2
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  ## update w_1, w_2, w_3, w_4, b1 
  # compute dE_douth1 first
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
  
  # compute dE_douth2 first
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  dE_douth2 = dE_douto1*douto1_dneto1*dneto1_douth2 + dE_douto2*douto2_dneto2*dneto2_douth2 
  
  # compute dE_dw1    
  douth1_dneth1 = outh1*(1-outh1)
  dneth1_dw1 = input[1]
  dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
  
  # compute dE_dw2
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1*douth1_dneth1*dneth1_dw2
  
  # compute dE_dw3
  douth2_dneth2 = outh2*(1-outh2)
  dneth2_dw3 = input[1] 
  dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
  
  # compute dE_dw4
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2*douth2_dneth2*dneth2_dw4  
  
  # compute dE_db1
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  ### update all parameters via a gradient descent 
  w1 = w1 - gamma*dE_dw1
  w2 = w2 - gamma*dE_dw2
  w3 = w3 - gamma*dE_dw3
  w4 = w4 - gamma*dE_dw4
  w5 = w5 - gamma*dE_dw5
  w6 = w6 - gamma*dE_dw6
  w7 = w7 - gamma*dE_dw7
  w8 = w8 - gamma*dE_dw8
  b1 = b1 - gamma*dE_db1
  b2 = b2 - gamma*dE_db2    
  
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
  print(i)
  
}

ts.plot(err_2)

pred_0.6 = forwardProp(input, w, b)
pred_0.6[3:4]


w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

# input and target values
input1 = 0.05
input2 = 0.10 
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)


### gamma 0.6
gamma = 1.2
err_3 = c()

for(i in 1:numIter){
  
  ### forward
  res = forwardProp(input, w, b)
  outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
  
  ### compute error
  err_3[i] = error(res, out)
  
  ### backward propagation
  ## update w_5, w_6, w_7, w_8, b2 
  # compute dE_dw5
  dE_douto1 = -( out[1] - outo1 )
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1*douto1_dneto1*dneto1_dw5
  
  # compute dE_dw6
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1*douto1_dneto1*dneto1_dw6
  
  # compute dE_dw7
  dE_douto2 = -( out[2] - outo2 )
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2*douto2_dneto2*dneto2_dw7
  
  # compute dE_dw8
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2*douto2_dneto2*dneto2_dw8
  
  # compute dE_db2
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  ## update w_1, w_2, w_3, w_4, b1 
  # compute dE_douth1 first
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
  
  # compute dE_douth2 first
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  dE_douth2 = dE_douto1*douto1_dneto1*dneto1_douth2 + dE_douto2*douto2_dneto2*dneto2_douth2 
  
  # compute dE_dw1    
  douth1_dneth1 = outh1*(1-outh1)
  dneth1_dw1 = input[1]
  dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
  
  # compute dE_dw2
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1*douth1_dneth1*dneth1_dw2
  
  # compute dE_dw3
  douth2_dneth2 = outh2*(1-outh2)
  dneth2_dw3 = input[1] 
  dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
  
  # compute dE_dw4
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2*douth2_dneth2*dneth2_dw4  
  
  # compute dE_db1
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  ### update all parameters via a gradient descent 
  w1 = w1 - gamma*dE_dw1
  w2 = w2 - gamma*dE_dw2
  w3 = w3 - gamma*dE_dw3
  w4 = w4 - gamma*dE_dw4
  w5 = w5 - gamma*dE_dw5
  w6 = w6 - gamma*dE_dw6
  w7 = w7 - gamma*dE_dw7
  w8 = w8 - gamma*dE_dw8
  b1 = b1 - gamma*dE_db1
  b2 = b2 - gamma*dE_db2    
  
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
  print(i)
  
}

ts.plot(err_3)

pred_1.2 = forwardProp(input, w, b)
pred_1.2[3:4]


c = cbind(err_1,err_2,err_3)

c

# Gamma가 클수록 0에 빠르게 수렴하는 것을 확인할 수 있다.
ts.plot(c, gpars= list(col= 1:3))
legend("topright", legend = c('0.1','0.6','1.2'), col = 1:3, lty = 1)

pred_0.1[3:4]
pred_0.6[3:4]
pred_1.2[3:4]

# 4. Complete perceptron code for classifying setosa and virginica based on the DL09 slide.

data(iris)
iris_sub = iris[1:100, c(1,3,5)]
names(iris_sub) = c('sepal','petal','species')
head(iris_sub)


x = iris_sub[,1:2]
y = c()
for (i in iris_sub[,3]){
  if (i == 'setosa'){
    y = c(y, -1)
  }
  else {
    y = c(y, 1)
  }
}
y

euclidean.norm = function(x) {sqrt(sum(x*x))}
distance.from.plane = function(z,w,b) { sum(z*w) + b}
classify.linear = function(x,w,b) {
  distances = apply(x,1,distance.from.plane, w,b)
  return(ifelse(distances < 0, -1, +1))
}

perceptron = function(x,y,learning.rate){
  w = c(0,0)
  b = 0
  k = 0
  R = max(apply(x,1,euclidean.norm))
  mark.complete = TRUE
  while (mark.complete) {
    mark.complete = FALSE
    yc = classify.linear(x,w,b)
    for (i in 1:nrow(x)){
      if (y[i] != yc[i]) {
        w = w + learning.rate * y[i]*x[i,]
        b = b + learning.rate * y[i]*R^2
        k = k + 1
        mark.complete = TRUE
      }
    }
  }
  
  return (list(w=w ,b=b ,k=k))
}

p = perceptron(x,y,learning.rate = 1)
w = p$w
b = p$b
k = p$k
k
w
b


library(ggplot2)
ggplot(iris_sub, aes(x = sepal, y = petal)) +
  geom_point(aes(colour = species, shape =species), size = 3 ) +
  geom_abline(intercept= 2.884511, slope = -0.1433965, color ='red', size =3)+
  xlab('sepal length')+
  ylab('petal length')
