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
Used  when the points or targets are not linear . Eg of this is to use sklearn PolynomialFeatures preprocessor.

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

- **AND** perceptron .. 
- Similarly an **OR** perceptron 
 converting from and to or preceptron , increase the weights or decrease magnitude of bias . 
 ![pct](https://video.udacity-data.com/topher/2017/May/5912c232_and-to-or/and-to-or.png) 
- **NOT** preceptron : 
- **XOR** Perceptron
![xor](https://video.udacity-data.com/topher/2017/May/5912c2f1_xor/xor.png)
This will require a multi layer perceptron 
![mp](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/59112cdf_xor-quiz2/xor-quiz2.png)

## Perceptron Trick
take steps to move the line closer to the co-ordinates by taking steps using x,y.. multiplied by the learning rate .

<iframe width="770" height="433" src="https://www.youtube.com/embed/lif_qPmXvWA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

eg . $3x_1+4x_2-10$ = 0 will take 10 rounds with lr 0.1 with perceptron trick to come top co-ordinate $(1,1)$ 

## Perceptron algorithm
There's a small error in the above video in that $W_i$
should be updated to $W_i = W_i + \alpha x_i$  (plus or minus depending on the situation).

# Decision Tree

## Entropy
**Entropy** : more ways of organize or randomness .. more rigid less entropy.
-- High Knowledge <> less Entropy Opposites

Entropy naive formula = $log(ab) = log(a) + log(b)$
~ here we will take log of probabilities.

**Question**
4 Red and 10 blue => Entropy = $4/14log4/14 + 10/14log10/14$ = 0.863

![mn](https://video.udacity-data.com/topher/2018/May/5b046ed6_screen-shot-2018-05-22-at-12.25.34-pm/screen-shot-2018-05-22-at-12.25.34-pm.png)
$entropy = -p_1log_2(p_1) - p_2log_2(p_2)$

Hence for n elements it is 

$entropy = -p_1log_2(p_1) - p_2log_2(p_2) - ...- p_nlog_2(p_n) = -\sum_{i=1}^{n}p_ilog_2(p_i)$



## Information Gain

Information Gain DT = Entropy[parent] - Avg Sum of Entropy children

maximize information gain is how decision trees model build up .

## Hyperparameters in Decision Trees
### Maximum depth
he maximum depth of a decision tree is simply the largest possible length between the root to a leaf. A tree of maximum length kk can have at most $2^k$ leaves.
![alt](https://video.udacity-data.com/topher/2018/January/5a51b0c0_screen-shot-2018-01-06-at-9.30.27-pm/screen-shot-2018-01-06-at-9.30.27-pm.png)
### Minimum number of samples to split
A node must have at least min_samples_split samples in order to be large enough to split. If a node has fewer samples than min_samples_split samples, it will not be split, and the splitting process stops.
![alt](https://video.udacity-data.com/topher/2019/January/5c3fd4e9_min-samples-split/min-samples-split.png))
### Minimum Number of samples per leaf
When splitting a node, one could run into the problem of having 99 samples in one of them, and 1 on the other. This will not take us too far in our process, and would be a waste of resources and time. If we want to avoid this, we can set a minimum for the number of samples we allow on each leaf.
![alt](https://video.udacity-data.com/topher/2018/January/5a51b331_screen-shot-2018-01-06-at-9.41.01-pm/screen-shot-2018-01-06-at-9.41.01-pm.png)

```python
from sklearn.tree import DecisionTreeClassifier
model1 =  DecisionTreeClassifier(max_depth = 30, min_samples_leaf = 6,min_samples_split=17)
```

# Naive Bayes

Probability of R and A happening and R and B happening, by law of conditional probability.
> Prior Probability

$P(R ∩ A) = P(A)P(R|A)$

$P(R ∩ B) = P(B)P(R|B)$

Hence our universe now consist of the above two events , Thus as they dont equate to one thus we divide them by there sum to normalize.
> Posterior Probability

$P(A|R)=\dfrac{P(A)P(R|A)}{P(A)P(R|A) + P(B)P(R|B)}$

also 
$P(A|B) = \dfrac{P(B|A)P(A)}{P(B)}$

![alt](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Bayes_theorem_simple_example_tree.svg/220px-Bayes_theorem_simple_example_tree.svg.png)

Question
 ![alt](https://video.udacity-data.com/topher/2018/January/5a6a7b8f_spamham/spamham.png)

 What is the probability that an e-mail contains the word 'easy', given that it is spam?

$P=(3/8*1/3)/(1/3*3/8+5/8*1/5)=1/2$

What is the probability that an e-mail contains the word 'money', given that it is spam?

$(3/8*2/3)/((2/3*3/8)+(5/8*1/5))=2/3$

### Conditional Probabilities

$P(R ∩ A) = P(A)P(R|A)$

Naive assumption

$P(A|B) \propto P(B|A)P(A)$

$P(A|B,C) \propto P(B,C|A)P(A)$

$P(A|B,C) \propto P(B|A)P(C|A)P(A)$

Generalized naive Bayes

$P(A|B,C..N) \propto P(B|A)P(C|A)...P(N|A)P(A)$


# SVM
Maximize margin .. **Minimize** 

$CLASSIFICATION ERROR + MARGINERROR$

Margin : Distance between lines in svm 

$Margin = \frac{2}{|W|}$

$Error = |w|^2$

Hence =>

Large margin -> Small error

Small Margin -> large error

## Margin Error calculation

Need to find distance between 2 lines 
![alt](https://video.udacity-data.com/topher/2018/January/5a52bf15_margin-geometry-images.001/margin-geometry-images.001.jpeg)

Since we are measuring distance between the lines , we can translate these so that one touches the origin which makes the eq

- $Wx=0$ which is orthogonal to the vector $W=(w_1,w_2)$
- $Wx=1$

![alt](https://video.udacity-data.com/topher/2018/January/5a52bf2b_margin-geometry-images.003/margin-geometry-images.003.jpeg)

This vector interesects $Wx=1$ at the blue point hence
- $w_1p+w_2q=1$
- (p,q) is a multiple of $(w_1,w_2)$ since the point is over vector W
![alt](https://video.udacity-data.com/topher/2018/January/5a52bf35_margin-geometry-images.004/margin-geometry-images.004.jpeg)

so LEt $(p,q)=k(w_1,w_2)$ for some k , which when substituing in first equation gives
$k(w_1^2+w_2^2)= 1$  Therefore $k = \frac{1}{w_1^2+w_2^2} = \frac{1}{|w|^2}$  Hence blue point represents the vector $\frac{W}{|W|^2}$

Now, the distance between the two lines, is simply the norm of the blue vector.

![alt](https://video.udacity-data.com/topher/2018/January/5a52bf47_margin-geometry-images.005/margin-geometry-images.005.jpeg)


 we remember that the desired distance was the sum of these two distances between the consecutive parallel lines .Since one of them is $\frac{1}{|W|}$ , then the total distance is $\frac{2}{|W|}$

 ![1](https://video.udacity-data.com/topher/2018/January/5a52bf73_margin-geometry-images.008/margin-geometry-images.008.jpeg)

 $Eoor Fn = CLASSIFICATION ERROR + MARGINERROR$ minimized by gradient descent

 ## C parameter
 Multiplies to the classification error 
 Error = C * Classification Error + Margin Error 

Small C -> Large margin -> May make classification errors 

Large C => Classifies point well byt may have a small margin.

## Kernel trick 

Using polynomial . can do via circle via polynomials or via creating a 3d plane(s)

![alt](https://video.udacity-data.com/topher/2018/February/5a78e676_polynomial-kernel-2-quiz/polynomial-kernel-2-quiz.png)

both methods can be used as a circle or a 3d parabolic plane will do similar seperations between the points .Hence a polynomial kernel works to seperate data . 

The **degree** can be tuned as a hyper parameter .

### RBF kernel 
Radia basis function kernel
![alt](https://video.udacity-data.com/topher/2018/June/5b133ff6_screen-shot-2018-06-02-at-6.07.54-pm/screen-shot-2018-06-02-at-6.07.54-pm.png)
For Large $\gamma$ will give pointy mountains and small $\gamma$ will give flat mountains thus large values will tend to overfit while small ones tend to underfit .

Gamma is actually derived from a normal distribution where $y=\frac{1}{\sigma\sqrt{2\pi}}e^-\frac{(x-\mu)^2}{2\sigma^2}$ hence 

$\gamma = \frac{1}{2\sigma^2}$

Hence $\sigma$ is small $\gamma$ is high and the curve is narrow while $\sigma$ is large the curve is wide and $\gamma$ is small .

For higher dimension the formula is complex though reasoning is same .

# Ensemble Methods
Commonly the "weak" learners you use are decision trees. In fact the default for most ensemble methods is a decision tree in sklearn. However, you can change this value to any of the models you have seen so far.

There were two randomization techniques you saw to combat overfitting:

Bootstrap the data - that is, sampling the data with replacement and fitting your algorithm and fitting your algorithm to the sampled data.

Subset the features - in each split of a decision tree or with each algorithm used an ensemble only a subset of the total possible features are used.

## Bagging (Bootstrap aggregating) Bootstrap the data 
combine based on some criterion by voting .. 

## Boosting
select special weak learners to join and create a strong learner .
**Why Would We Want to Ensemble Learners Together?**

There are two competing variables in finding a well fitting machine learning model: Bias and Variance. It is common in interviews for you to be asked about this topic and how it pertains to different modeling techniques. As a first pass, the wikipedia is quite useful. However, I will give you my perspective and examples:

**Bias**: When a model has high bias, this means that means it doesn't do a good job of bending to the data. An example of an algorithm that usually has *high bias is linear regression*. Even with completely different datasets, we end up with the same line fit to the data. When models have high bias, this is bad.

**Variance**: When a model has high variance, this means that it changes drastically to meet the needs of every point in our dataset. Linear models like the one above has low variance, but high bias. An example of an algorithm that tends to have high variance and low bias is a decision tree (especially decision trees with no early stopping parameters). *A decision tree, as a high variance* algorithm, will attempt to split every point into its own branch if possible. This is a trait of high variance, low bias algorithms - they are extremely flexible to fit exactly whatever data they see.

By combining algorithms, we can often build models that perform better by meeting in the middle in terms of bias and variance. well. These ideas are based on minimizing bias and variance based on mathematical theories, like the central limit theorem.

also need to avoid with ensembles 

**High Bias, Low Variance** models tend to underfit data, as they are not flexible. Linear models fall into this category of models.

**High Variance, Low Bias** models tend to overfit data, as they are too flexible. Decision trees fall into this category of models.

**Introducing Randomness Into Ensembles**

nother method that is used to improve ensemble methods is to introduce randomness into high variance algorithms before they are ensembled together. The introduction of randomness combats the tendency of these algorithms to overfit (or fit directly to the data available). There are two main ways that randomness is introduced:

**Bootstrap the data** - that is, sampling the data with replacement and fitting your algorithm to the sampled data.

**Subset the features** - in each split of a decision tree or with each algorithm used in an ensemble, only a subset of the total possible features are used.

In fact, these are the two random components used in the next algorithm you are going to see called **random forests**. Called so as the column/features to build trees are selected in radom .

## ADABOOST 
- first fit model by accuracy .. on first weak learner
- fix mistakes and focus on misclassified points by enlarging the miscllasified points .
- punish points which are misclassified .and tries to correctly classified enlarged points .

![alt](https://video.udacity-data.com/topher/2018/January/5a4d5823_screen-shot-2018-01-03-at-2.23.38-pm/screen-shot-2018-01-03-at-2.23.38-pm.png)

https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf 

in extreme cases we can stop or ignore based on weight being $\infin$ or $-\infty$

Add the positive and negatives of the weight . make them vote and add the positives and negative of the weights .

```
>>> from sklearn.ensemble import AdaBoostClassifier
>>> model = AdaBoostClassifier()
>>> model.fit(x_train, y_train)
>>> model.predict(x_test)

can tune estimator and number of estimators 

>>> from sklearn.tree import DecisionTreeClassifier
>>> model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
```