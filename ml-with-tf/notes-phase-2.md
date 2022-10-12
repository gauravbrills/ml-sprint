# Phase 2 notes
**Bagging** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier 
**AdaBoost** - (Ada)ptive (Boost)ing, is an ensemble ethod technique that re-assigns weights to each instance, with higher weights to incorrectly classified instances.AdaBoost classifier fits on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
**Random forest** - Using 2+ decision trees on randomly picked columns.RandomForest classifier fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
 https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier


Boosting
The original paper https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf - A link to the original paper on boosting by Yoav Freund and Robert E. Schapire.
An explanation about why boosting is so important - A great article on boosting by a Kaggle master, Ben Gorman.
**A useful Quora post** - A number of useful explanations about boosting. https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting

**AdaBoost**
Here is the original paper from Freund and Schapire that is a short overview paper introducing the boosting algorithm AdaBoost, and explains the underlying theory of boosting, including an explanation of why boosting often does not suffer from overfitting as well as boosting’s the relationship to support-vector machines. https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf
A follow-up paper from the same authors regarding several experiments with Adaboost. https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf
A great tutorial by Schapire explaining the many perspectives and analyses of AdaBoost that have been applied to explain or understand it as a learning method, with comparisons of both the strengths and weaknesses of the various approaches. http://rob.schapire.net/papers/explaining-adaboost.pdf

**F1-score** : Metric that conveys the balance between the precision and the recall.
**R2** : Regression metric that represents the 'amount of variability captured by a model, or the average amount you miss across all the points and the R2 value as the amount of the variability in the points that you capture with a model. R2 score is based on comparing a model to the simplest possible model. If the R2 score is close to 1, then the model is good

**Classifications notes Stanford** https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks

*** 

# Neural Networks

**Error Functions** : Distance from the the current state to solution in NN 
- Error function must be continuous 
- Discrete step fn in grad descent can be change to a continuous fn like sigmoid .

## Sigmoid Fn

A sigmoid function is an s-shaped function that gives us:
- Output values close to 1 when the input is a large positive number
- Output values close to 0 when the input is a large negative number
- Output values close to 0.5 when the input is close to zero

$σ(x) = \frac{1}{1 + e{-x}}$
​
 
## Softmax

$P(\text{class i}) = \frac{e^{Z_i}}{e^{Z_1}+…+e^{Z_n}}$
​
Code 

```python
import numpy as np 
def softmax(L):
    return  [np.exp(e)/sum([np.exp(e1) for e1 in L])  for e in L]
```
 
## Maximum Likelihood 
​The key idea is that we want to calculate $P(\text{all})$, which is the product of all the independent probabilities of each point. This helps indicate how well the model performs in classifying all the points. To get the best model, we will want to maximize this probability.

## Cross Entropy
Sum of negative of log of all probabilities . cross-entropy. A good model gives a high probability and the negative of a logarithm of a large number is a small number—thus, in the end:

- A high cross-entropy indicates a bad model
- A low cross-entropy indicates a good model

$\text{Cross-entropy} = -\sum_{i=1}^my_iln(p_i) + (1-y_i)ln(1-p_i)$  

Where $m$ is the number of classes.

code 
```python
import numpy as np
import math
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -1*sum([(Y[i]*math.log(P[i])+ (1-Y[i])*math.log(1-P[i])) for i in range(len(Y)) ])
#or

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```    
### Multi Class Cross entropy

Cross-entropy $=−\sum_{i=1}^n\sum_{j=1}^my_ijln(p_ij)$

Where, as before, $m$ is the number of classes.

## Logistic regression
For binary classificaiton loss function is

$\text{Error function} = -\frac{1}{m}\sum_{i=1}^m(1-y_i)ln(1-\hat{y_i})+y_iln(\hat{y_i})$

Total formula for error is

$E(W, b) = -\frac{1}{m}\sum_{i=1}^m(1-y_i)ln(1-\sigma(Wx^{(i)}+b))+y_iln(\sigma(Wx^{(i)}+b)$

For multi class problems it is 

$\text{Error function} = −\frac{1}{m}\sum_{i=1}^m\sum_{j=1}^n y_{ij}ln(\hat{y}_{ij})$

## Gradient descent
Gradient

$∇E=−(y−\hat{y})(x_1,…,x_n ,1)$.

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

### Gradient Descent Step

Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function at every point, then this updates the weights in the following way:

$w_i' ← w_i - \alpha[-(y-\hat{y})x_i]$

which is equivalent to

$w_i' ← w_i - \alpha(y-\hat{y})x_i$

Similarly, it updates the bias in the following way:

$b' ← b + \alpha(y-\hat{y})$

Note: Since we've taken the average of the errors, the term we are adding should be \frac{1}{m} \cdot \alpha 
m
1
​
 ⋅α instead of \alpha,α, but as \alphaα is a constant, then in order to simplify calculations, we'll just take $\frac{1}{m}\alpha$
 ⋅ α to be our learning rate, and abuse the notation by just calling it \alpha.α.

 ## Perceptron vs Gradient Descent

### Gradient Descent
With gradient descent, we change the weights from $w_i$  to $w_i +a(y-\hat{y})x_i$.

### Perceptron Algorithm
With the perceptron algorithm we only change the weights on the misclassified points. If a point $x$ is misclassified:

We change $w_i$ :
- To $w_i +ax_iw$ if positive
- To $w_i -ax_iw$ if negative
-  If correctly classified: $y-\hat{y}=0$
- If misclassified:
    - $y-\hat{y}=1$ if positive
    - $y-\hat{y}=-1$ if negative

## Gradient descent
Also another name for rate of change of slope . https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient , basicallya derivative of $f'(x)$ (ie slope at point x)

### Caveats
Since the weights will just go wherever the gradient takes them, they can end up where the error is low, but not the lowest. These spots are called local minima. If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum, illustrated below.

![ways to repvent](https://video.udacity-data.com/topher/2017/January/587c5ebd_local-minima/local-minima.png)

This can be avoided by [momentum](https://distill.pub/2017/momentum/) also implemented in libs like pytorch https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

### The Math