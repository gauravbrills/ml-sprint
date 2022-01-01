# Absolute Trick 

$\alpha$ = Learning Rate
Line > y = w1x + w2

Point : (p,q)

y = (w1-p$\alpha$)x + (w2 - $\alpha$)

# Square Trick 

$\alpha$ = Learning Rate
Line > y = w1x + w2

Point : (p,q)

y = (w1+p(q-q')$\alpha$)x + (w2 + (q-q')$\alpha$)

# Gradient Descent
- Gradient of error function
- $w_i$ -> $w_i$ - $\alpha$ $\frac{d}{dw_i}$ Error

# Mean Absolute error

Error = $1/m$$\sum_{i=1}^{m}$$|y-\hat{y}|$
> Always positive .. hence the absolute

# Mean Squared Erorr 

Error = $1/2m$$\sum_{i=1}^{m}$$(y-\hat{y})^2$

# Higher dimension

Plane .. 
n - dimension .. (x1,x2..xn)
n-1 dimensional hyperplane
y_hat = w1x1 + w2x2 ... wn

# Closed form solution

Error($w_1,w_2$) = $1/2m$$\sum_{i=1}^{m}$$(y-\hat{y})^2$

Solving this to minimize error, Ignoring 1/m

Error($w_1,w_2$) = $1/2$$\sum_{i=1}^{m}$$(y-\hat{y})^2$
=> $1/2$$\sum_{i=1}^{m}$$(w_1x_i+w_2 -y)^2$

To minimize the error function we need to take the derivatives wrt w1 and w2 and set them to zero .
usign the chain rule 

$\frac{dE}{dw_1}$ = $\sum_{i=1}^{m}$$(w_1x_i+w_2 -y)x_i$

=  $w_1\sum_{i=1}^{m}$$w_1x_i^2$+$w_2\sum_{i=1}^{m}x_i$-$\sum_{i=1}^{m}x_iy_i$


$\frac{dE}{dw_2}$ = $\sum_{i=1}^{m}$$(w_1x_i+w_2 -y)$

=  $w_1\sum_{i=1}^{m}$$w_1x_i^2$+$w_2\sum_{i=1}^{m}x_i$-$\sum_{i=1}^{m}y_i$
Setting the two equations to zero gives us the following system of two equations and two variables (where the variables are w_1 and w_2)

![t](https://video.udacity-data.com/topher/2017/November/5a175b5f_codecogseqn-61/codecogseqn-61.gif)

![t](https://video.udacity-data.com/topher/2017/October/59f7f2ae_f4/f4.gif)

For n dimensions

 Our matrix XX containing the data is the following, where each row is one of our datapoints, and $x_o^i=1$ represents the bias .

 ![](https://video.udacity-data.com/topher/2017/October/59f7f63c_m/m.gif)

 Labels

![alt](https://video.udacity-data.com/topher/2017/October/59f7f716_y/y.gif)


Weights

![a](https://video.udacity-data.com/topher/2017/November/5a1b2275_codecogseqn-62/codecogseqn-62.gif)

Equation for MSE 

E(W) = $1/m((XW)^T - y^T)(XW -y)$

Ignoring 1/m

E(W) = $X^TW^TXW - (XW)^Ty - y^T(XW) + y^Ty$

Notice that in the sum above, the second and the third terms are the same, since it's the inner product of two vectors, which means it's the sum of the products of its coordinates. Therefore,

E(W) = $X^TW^TXW - 2(XW)^Ty + y^Ty$

To minimize this using chian rule 

![a](https://video.udacity-data.com/topher/2017/October/59f7fc57_e/e.gif)

In order to set this equal to zero we need

$X^TXW - X^Ty = 0$

or equivalently 

$W = (X^TX)^-1X^Ty$ 

To do this is expesice ans finding inverse of metrics $X^TX$ for is hard for a large value of n . Though if the matrix X is sparse meaning most of the entries are zero then there are ways to solve this .

# Limitations to linear regression

- Linear Regression Works Best When the Data is Linear
- Linear Regression is Sensitive to Outliers

# Polynomial Regression
used  when the points or targets are not linear . Eg of this is to use sklearn PolynomialFeatures preprocessor.

# Regularization

## L1 Regularization 
**Error = absolute value of coeff's**
Eg: 
  $2x_1^3 - 2x_1^2x_2 -4x_2^3 +3x_2^2$
  Error = |2|+|-2|+|-4|+|3| = 11 

- computationally Inefficient(unless data is sparse)
- Sparse outputs
- Feature Selection
  
## L2 Regularization

**Error = square of the coeff** 
eg . as above add square of coeff.

Simple model genralize better in some cases . Need to check how to penalize complexity . We use the $\lambda$ parameter for this. Tune $\lambda$ approriatly . Large $\lambda$ punishes complexity more .
- Computationally efficient
- Non Sparse outputs
- No Feature Selection

Example of this is Lasso to use L1 regularization

# Feature Scaling
- Standardizing : Standardizing is completed by taking each value of your column, subtracting the mean of the column, and then dividing by the standard deviation of the column.
- Normalizing :  With normalizing, data are scaled between 0 and 1 .
    Eg : 
   ```python
   df["height_normal"] = (df["height"] - df["height"].min()) /     \
                      (df["height"].max() - df['height'].min())
   ```

## When to use feature scaling
In many machine learning algorithms, the result will change depending on the units of your data. This is especially true in two specific cases:

- When your algorithm uses a distance-based metric to predict.
- When you incorporate regularization.

Distance Based Metrics > SVMs, K-nearest neighbors

**Regularization** : When you start introducing regularization, you will again want to scale the features of your model. The penalty on particular coefficients in regularized linear regression techniques depends largely on the scale associated with the features. When one feature is on a small range, say from 0 to 10, and another is on a large range, say from 0 to 1 000 000, applying regularization is going to unfairly punish the feature with the small range. Features with small ranges need to have larger coefficients compared to features with large ranges in order to have the same effect on the outcome of the data. (Think about how ab = baab=ba for two numbers aa and bb.) Therefore, if regularization could remove one of those two features with the same net increase in error, it would rather remove the small-ranged feature with the large coefficient, since that would reduce the regularization term the most.

Again, this means you will want to scale features any time you are applying regularization.
https://www.quora.com/Why-do-we-normalize-the-data.
A point raised in the article above is that feature scaling can speed up convergence of your machine learning algorithms, which is an important consideration when you scale machine learning applications.

eg in sklearn **sklearn.preprocessing.StandardScaler**

***
# Perceptron

## classificaiton

$w_1x_1 + w_2x_2 ... w_n_x_n + b = 0$ 
-> 
Wx + b = 0 

$\hat{y}$ =  1 if $Wx + b >= 0$ , 0 if $Wx + b <0$ 

Dimensions in n dimensions
$W: (1,n) , x: (n,1), b:(1,1)$

## Perceptron : 
Gets input $x_1 ...  x_n$ with weights $w_1 ...  w_n$ and computes a **Linear function** $Wx +b = \sum_{i=1}^{n}$$w_ix_i + b$ and check Wx +b >= 0
.. this then returns Yes 1 or No 0
This then sends value to a Step Function which returns 1 or 0 based on the condition before .

### Preceptrons as Logical Operators

