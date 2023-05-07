Download Link: https://assignmentchef.com/product/solved-mlcs-homework-3-conditional-probability-models
<br>



<h1>1           Introduction</h1>

In this homework we’ll be investigating conditional probability models, with a focus on various interpretations of logistic regression, with and without regularization. Along the way we’ll discuss the calibration of probability predictions, both in the limit of infinite training data and in a more bare-hands way. On the Bayesian side, we’ll recreate from scratch the Bayesian linear gaussian regression example we discussed in lecture. We’ll also have several optional problems that work through many basic concepts in Bayesian statistics via one of the simplest problems there is: estimating the probability of heads in a coin flip. Later we’ll extend this to the probability of estimating click-through rates in mobile advertising. Along the way we’ll encounter empirical Bayes and hierarchical models.

<h1>2           From Scores to Conditional Probabilities<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></h1>

Let’s consider the classification setting, in which (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>) ∈ X × {−1<em>,</em>1} are sampled i.i.d. from some unknown distribution. For a prediction function <em>f </em>: X → <strong>R</strong>, we define the margin on an example (<em>x,y</em>) to be <em>m </em>= <em>yf</em>(<em>x</em>). Since our class predictions are given by sign(<em>f</em>(<em>x</em>)), we see that a prediction is correct iff <em>m</em>(<em>x</em>) <em>&gt; </em>0. It’s tempting to interpret the magnitude of the score |<em>f</em>(<em>x</em>)| as a measure of confidence. However, it’s hard to interpret the magnitudes beyond saying one prediction score is more or less confident than another, and without any scale to this “confidence score”, it’s hard to know what to do with it. In this problem, we investigate how we can translate the score into a probability, which is much easier to interpret. In other words, we are looking for a way to convert score function <em>f</em>(<em>x</em>) ∈ <strong>R </strong>into a conditional probability distribution <em>x </em>7→ <em>p</em>(<em>y </em>= 1 | <em>x</em>).

In this problem we will consider margin-based losses, which are loss functions of the form (<em>y,f</em>(<em>x</em>)) 7→ <em>`</em>(<em>yf</em>(<em>x</em>)), where <em>m </em>= <em>yf</em>(<em>x</em>) is called the margin. We are interested in how we can go from an empirical risk minimizer for a margin-based loss, <em>f</em><sup>ˆ </sup>= argmin, to a conditional probability estimator <em>π</em>ˆ(<em>x</em>) ≈ <em>p</em>(<em>y </em>= 1 | <em>x</em>). Our approach will be to try to find a way to use the Bayes<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> prediction function<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> <em>f</em><sup>∗ </sup>= argmin<em><sub>f </sub></em>E<em>x,y </em>[<em>`</em>(<em>yf</em>(<em>x</em>)] to get the true conditional probability <em>π</em>(<em>x</em>) = <em>p</em>(<em>y </em>= 1 | <em>x</em>), and then apply the same mapping to the empirical risk minimizer. While there is plenty that can go wrong with this “plug-in” approach (primarily, the empirical risk minimizer from a [limited] hypothesis space F may be a poor estimate for the Bayes prediction function), it is at least well-motivated, and it can work well in practice. And please note that we can do better than just hoping for success: if you have enough validation data, you can directly assess how well “calibrated” the predicted probabilities are. This blog post has some discussion of calibration plots: <a href="https://jmetzen.github.io/2015-04-14/calibration.html">https://jmetzen.github.io/2015-04-14/calibration.html</a><a href="https://jmetzen.github.io/2015-04-14/calibration.html">.</a>

It turns out it is straightforward to find the Bayes prediction function <em>f</em><sup>∗ </sup>for margin losses, at least in terms of the data-generating distribution: For any given <em>x </em>∈ X, we’ll find the best possible prediction <em>y</em>ˆ. This will be the <em>y</em>ˆ that minimizes

E<em>y </em>[<em>`</em>(<em>yy</em>ˆ) | <em>x</em>]<em>.</em>

If we can calculate this <em>y</em>ˆ for all <em>x </em>∈ X, then we will have determined <em>f</em><sup>∗</sup>(<em>x</em>). We will simply take

<em>f</em><sup>∗</sup>(<em>x</em>) = argminE<em>y </em>[<em>`</em>(<em>yy</em>ˆ) | <em>x</em>]<em>.</em>

<em>y</em>ˆ

Below we’ll calculate <em>f</em><sup>∗ </sup>for several loss functions. It will be convenient to let <em>π</em>(<em>x</em>) = <em>p</em>(<em>y </em>= 1 | <em>x</em>) in the work below.

<ol>

 <li>Write E<em>y </em>[<em>`</em>(<em>yf</em>(<em>x</em>)) | <em>x</em>] in terms of <em>π</em>(<em>x</em>), <em>`</em>(−<em>f</em>(<em>x</em>)), and <em>`</em>(<em>f</em>(<em>x</em>)). [Hint: Use the fact that <em>y </em>∈ {−1<em>,</em>1}.]</li>

 <li>Show that the Bayes prediction function <em>f</em><sup>∗</sup>(<em>x</em>) for the exponential loss function <em>`</em>(<em>y,f</em>(<em>x</em>)) =</li>

</ol>

<em>e</em>−<em>yf</em>(<em>x</em>) is given by

<em>,</em>

where we’ve assumed <em>π</em>(<em>x</em>) ∈ (0<em>,</em>1). Also, show that given the Bayes prediction function <em>f</em><sup>∗</sup>, we can recover the conditional probabilities by

<em>.</em>

[Hint: Differentiate the expression in the previous problem with respect to <em>f</em>(<em>x</em>). To make things a little less confusing, and also to write less, you may find it useful to change variables a bit: Fix an <em>x </em>∈ X. Then write <em>p </em>= <em>π</em>(<em>x</em>) and <em>y</em>ˆ = <em>f</em>(<em>x</em>). After substituting these into the expression you had for the previous problem, you’ll want to find <em>y</em>ˆ that minimizes the expression. Use differential calculus. Once you’ve done it for a single <em>x</em>, it’s easy to write the solution as a function of <em>x</em>.]

<ol start="3">

 <li>Show that the Bayes prediction function <em>f</em><sup>∗</sup>(<em>x</em>) for the logistic loss function <em>`</em>(<em>y,f</em>(<em>x</em>)) = is given by</li>

</ol>

and the conditional probabilities are given by

<em>.</em>

Again, we may assume that <em>π</em>(<em>x</em>) ∈ (0<em>,</em>1).

<ol start="4">

 <li>[Optional] Show that the Bayes prediction function <em>f</em><sup>∗</sup>(<em>x</em>) for the hinge loss function <em>`</em>(<em>y,f</em>(<em>x</em>)) = max(0<em>,</em>1 − <em>yf</em>(<em>x</em>)) is given by</li>

</ol>

<em>.</em>

Note that it is impossible to recover <em>π</em>(<em>x</em>) from <em>f</em><sup>∗</sup>(<em>x</em>) in this scenario. However, in practice we work with an empirical risk minimizer, from which we may still be able to recover a reasonable estimate for <em>π</em>(<em>x</em>). An early approach to this problem is known as “Platt scaling”: <a href="https://en.wikipedia.org/wiki/Platt_scaling">https://en.wikipedia.org/wiki/Platt_scaling</a><a href="https://en.wikipedia.org/wiki/Platt_scaling">.</a>

<h1>3           Logistic Regression</h1>

<h2>3.1           Equivalence of ERM and probabilistic approaches</h2>

In lecture we discussed two different ways to end up with logistic regression.

ERM approach: Consider the classification setting with input space X = <strong>R</strong><em><sup>d</sup></em>, outcome space Y<sub>± </sub>= {−1<em>,</em>1}, and action space A<strong><sub>R </sub></strong>= <strong>R</strong>, with the hypothesis space of linear score functions

F<sub>score </sub>. Consider the margin-based loss function <em>`</em><sub>logistic</sub>(<em>m</em>) = log(1 + <em>e</em><sup>−<em>m</em></sup>) and the training data D = ((<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>)). Then the empirical risk objective function for hypothesis space F<sub>score </sub>and the logistic loss over D is given by

<em>n</em>

logistic(<em>y<sub>i</sub></em><em><sup>w</sup></em><em><sup>T</sup></em><em>x<sub>i</sub></em>)

<em>.</em>

Bernoulli regression with logistic transfer function: Consider the conditional probability modeling setting with input space X = <strong>R</strong><em><sup>d</sup></em>, outcome space Y<sub>0<em>/</em>1 </sub>= {0<em>,</em>1}, and action space A<sub>[0<em>,</em>1] </sub>= [0<em>,</em>1], where an action corresponds to the predicted probability that an outcome is 1.

Define the standard logistic function as <em>φ</em>(<em>η</em>) = 1<em>/</em>(1 + <em>e</em><sup>−<em>η</em></sup>) and the hypothesis space F<sub>prob </sub>= . Suppose for every <em>y<sub>i </sub></em>in the dataset D above, we define, and let D<sup>0 </sup>be the resulting collection of (<em>x<sub>i</sub>,y<sub>i</sub></em><sup>0</sup>) pairs. Then the negative log-likelihood (NLL) objective function for F<sub>prob </sub>and D<sup>0 </sup>is give by

NLL

If <em>w</em>ˆ<sub>prob </sub>minimizes NLL(<em>w</em>), then <em>x </em>7→ <em>φ</em>(<em>x<sup>T</sup>w</em>ˆ<sub>prob</sub>) is a maximum likelihood prediction function over the hypothesis space F<sub>prob </sub>for the dataset D<sup>0</sup>.

Show that <em>nR</em><sup>ˆ</sup><em><sub>n</sub></em>(<em>w</em>) = NLL(<em>w</em>) for all <em>w </em>∈ <strong>R</strong><em><sup>d</sup></em>. And thus the two approaches are equivalent, in that they produce the same prediction functions.

<h2>3.2           Numerical Overflow and the log-sum-exp trick</h2>

Suppose we want to calculate log(exp(<em>η</em>)) for <em>η </em>= 1000<em>.</em>42. If we compute this literally in Python, we will get an overflow (try it!), since numpy gets infinity for <em>e</em><sup>1000<em>.</em>42</sup>, and log of infinity is still infinity. On the other hand, we can help out with some math: obviously log(exp(<em>η</em>)) = <em>η</em>, and there’s no issue.

It turns out, log(exp(<em>η</em>)) and the problem with its calculation is a special case of the <a href="https://en.wikipedia.org/wiki/LogSumExp">LogSumExp </a>function that shows up frequently in machine learning. We define

LogSumExp(<em>x</em><sub>1</sub><em>,…,x<sub>n</sub></em>) = log(<em>e<sup>x</sup></em><sup>1 </sup>+ ··· + <em>e<sup>x</sup></em><em><sup>n</sup></em>)<em>.</em>

Note that this will overflow if any of the <em>x<sub>i</sub></em>’s are large (more than 709). To compute this on a computer, we can use the “log-sum-exp trick”. We let <em>x</em><sup>∗ </sup>= max(<em>x</em><sub>1</sub><em>,…,x<sub>n</sub></em>) and compute LogSumExp as

∗                     h <em>x </em>−<em>x</em>

LogSumExp(<em>x</em>1<em>,…,x</em><em>n</em>) = <em>x </em>+ log <em>e </em>1                              ∗ + ··· + <em>e</em><em>x</em><em>n</em>−<em>x</em>∗i<em>.</em>

<ol>

 <li>Show that the new expression for LogSumExp is valid.</li>

 <li>Show that exp(<em>x<sub>i </sub></em>− <em>x</em><sup>∗</sup>) ∈ (0<em>,</em>1] for any <em>i</em>, and thus the exp calculations will not overflow.</li>

 <li>Above we’ve only spoken about the exp overflowing. However, the log part can also have problems by becoming negative infinity for arguments very close to 0. Explain why the log term in our expression will never be “-inf”.</li>

 <li>In the objective functions for logistic regression, there are expressions of the form log(1 + <em>e</em><sup>−<em>s</em></sup>) for some <em>s</em>. Note that a naive implementation gives 0 for <em>s &gt; </em>36 and inf for <em>s &lt; </em>−709. Show how to use the numpy function <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.logaddexp.html"><em>logaddexp</em></a> to correctly compute log(1 + <em>e</em><sup>−<em>s</em></sup>).</li>

</ol>

<h2>3.3           Regularized Logistic Regression</h2>

For a dataset D = ((<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>)) drawn from <strong>R</strong><em><sup>d</sup></em>×{−1<em>,</em>1}, the regularized logistic regression objective function can be defined as

<em>J</em>logistic

<em>.</em>

<ol>

 <li>Prove that the objective function <em>J</em><sub>logistic</sub>(<em>w</em>) is convex. You may use any facts mentioned in the <a href="https://davidrosenberg.github.io/mlcourse/Notes/convex-optimization.pdf">convex optimization notes</a><a href="https://davidrosenberg.github.io/mlcourse/Notes/convex-optimization.pdf">.</a></li>

 <li>Complete the f_objective function in the skeleton code, which computes the objective function for <em>J</em><sub>logistic</sub>(<em>w</em>). Make sure to use the log-sum-exp trick to get accurate calculations and to prevent overflow.</li>

 <li>Complete the fit_logistic_regression_function in the skeleton code using the minimize function from scipy.optimize. ridge_regression.py from Homework 2 gives an example of how to use the minimize function. Use this function to train a model on the provided data. Make sure to take the appropriate preprocessing steps, such as standardizing the data and adding a column for the bias term.</li>

 <li>Find the <em>`</em><sub>2 </sub>regularization parameter that minimizes the log-likelihood on the validation set. Plot the log-likelihood for different values of the regularization parameter.</li>

 <li>Based on the Bernoulli regression development of logistic regression, it seems reasonable to interpret the prediction as the probability that <em>y </em>= 1, for a randomly drawn pair (<em>x,y</em>). Since we only have a finite sample (and we are regularizing, which will bias things a bit) there is a question of how well “<a href="https://en.wikipedia.org/wiki/Calibration_(statistics)">calibrated</a><a href="https://en.wikipedia.org/wiki/Calibration_(statistics)">”</a> our predicted probabilities are. Roughly speaking, we say <em>f</em>(<em>x</em>) is well calibrated if we look at all examples (<em>x,y</em>) for which <em>f</em>(<em>x</em>) ≈ 0<em>.</em>7 and we find that close to 70% of those examples have <em>y </em>= 1, as predicted… and then we repeat that for all predicted probabilities in (0<em>,</em>1). To see how well-calibrated our predicted probabilities are, break the predictions on the validation set into groups based on the predicted probability (you can play with the size of the groups to get a result you think is informative). For each group, examine the percentage of positive labels. You can make a table or graph. Summarize the results. You may get some ideas and references from <a href="http://scikit-learn.org/stable/modules/calibration.html">scikit-learn’s discussion</a><a href="http://scikit-learn.org/stable/modules/calibration.html">.</a></li>

 <li>[Optional] If you can, create a dataset for which the log-sum-exp trick is actually necessary for your implementation of regularized logistic regression. If you don’t think such a dataset exists, explain why. If you like, you may consider the case of SGD optimization. [This problem is intentionally open-ended. You’re meant to think, explore, and experiment. Points assigned for interesting insights.]</li>

</ol>

<h1>4           Bayesian Logistic Regression with Gaussian Priors</h1>

Let’s return to the setup described in Section 3.1 and, in particular, to the Bernoulli regression setting with logistic transfer function. We had the following hypothesis space of conditional probability functions:

<em>.</em>

Now let’s consider the Bayesian setting, where we induce a prior on F<sub>prob </sub>by taking a prior <em>p</em>(<em>w</em>) on the parameter <em>w </em>∈ <strong>R</strong><em><sup>d</sup></em>.

<ol>

 <li>For the dataset D<sup>0 </sup>described in Section 1, give an expression for the posterior density <em>p</em>(<em>w </em>| D<sup>0</sup>) in terms of the negative log-likelihood function</li>

</ol>

<em>n</em>

NLL

<em>i</em>=1

and a prior density <em>p</em>(<em>w</em>) (up to a proportionality constant is fine).

<ol start="2">

 <li>Suppose we take a prior on <em>w </em>of the form <em>w </em>∼ N(0<em>,</em>Σ). Find a covariance matrix Σ such that MAP estimate for <em>w </em>after observing data D<sup>0 </sup>is the same as the minimizer of the regularized logistic regression function defined in Section 3 (and prove it). [Hint: Consider minimizing the negative log posterior of <em>w</em>. Also, remember you can drop any terms from the objective function that don’t depend on <em>w</em>. Also, you may freely use results of previous problems.]</li>

 <li>In the Bayesian approach, the prior should reflect your beliefs about the parameters before seeing the data and, in particular, should be independent on the eventual size of your dataset. Following this, you choose a prior distribution <em>w </em>∼ N(0<em>,I</em>). For a dataset D of size <em>n</em>, how should you choose <em>λ </em>in our regularized logistic regression objective function so that the minimizer is equal to the mode of the posterior distribution of <em>w </em>(i.e. is equal to the MAP estimator).</li>

</ol>

<h1>5           Bayesian Linear Regression – Implementation</h1>

In this problem, we will implement Bayesian Gaussian linear regression, essentially reproducing the example <a href="https://davidrosenberg.github.io/mlcourse/Archive/2016/Lectures/13a.bayesian-regression.pdf#page=12">from lecture</a><a href="https://davidrosenberg.github.io/mlcourse/Archive/2016/Lectures/13a.bayesian-regression.pdf#page=12">,</a> which in turn is based on the example in Figure 3.7 of Bishop’s <em>Pattern Recognition and Machine Learning </em>(page 155). We’ve provided plotting functionality in “support_code.py”. Your task is to complete “problem.py”. The implementation uses np.matrix objects, and you are welcome to use<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> the np.matrix.getI method.

<ol>

 <li>Implement likelihood_func.</li>

 <li>Implement get_posterior_params.</li>

 <li>Implement get_predictive_params.</li>

 <li>Run “python problem.py” from inside the Bayesian Regression directory to do the regression and generate the plots. This runs through the regression with three different settings for the prior covariance. You may want to change the default behavior in support_code.make_plots from plt.show, to saving the plots for inclusion in your homework submission.</li>

 <li>Comment on your results. In particular, discuss how each of the following change with sample size and with the strength of the prior: (i) the likelihood function, (ii) the posterior distribution, and (iii) the posterior predictive distribution.</li>

 <li>Our work above was very much “full Bayes”, in that rather than coming up with a single prediction function, we have a whole distribution over posterior prediction functions. However, sometimes we want a single prediction function, and a common approach is to use the MAP estimate – that is, choose the prediction function that has the highest posterior likelihood. As we discussed in class, for this setting, we can get the MAP estimate using ridge regression. Use ridge regression to get the MAP prediction function corresponding to the first prior covariance , per the support code). What value did you use for the regularization coefficient?</li>

</ol>

Why?

<h1>6           [Optional] Coin Flipping: Maximum Likelihood</h1>

<ol>

 <li>[Optional] Suppose we flip a coin and get the following sequence of heads and tails:</li>

</ol>

D = (<em>H,H,T</em>)

Give an expression for the probability of observing D given that the probability of heads is <em>θ</em>. That is, give an expression for <em>p</em>(D | <em>θ</em>). This is called the likelihood of <em>θ </em>for the data D.

<ol start="2">

 <li>[Optional] How many different sequences of 3 coin tosses have 2 heads and 1 tail? If we toss the coin 3 times, what is the probability of 2 heads and 1 tail? (Answer should be in terms of <em>θ</em>.)</li>

 <li>[Optional] More generally, give an expression for the likelihood <em>p</em>(D | <em>θ</em>) for a particular sequence of flips D that has <em>n<sub>h </sub></em>heads and <em>n<sub>t </sub></em> Make sure you have expressions that make sense even for <em>θ </em>= 0 and <em>n<sub>h </sub></em>= 0, and other boundary cases. You may use the convention that 0<sup>0 </sup>= 1, or you can break your expression into cases if needed.</li>

 <li>[Optional] Show that the maximum likelihood estimate of <em>θ </em>given we observed a sequence with <em>n<sub>h </sub></em>heads and <em>n<sub>t </sub></em>tails is</li>

</ol>

<em>θ</em>ˆMLE <em>.</em>

You may assume that <em>n<sub>h </sub></em>+ <em>n<sub>t </sub></em>≥ 1. (Hint: Maximizing the log-likelihood is equivalent and is often easier. )

<h1>7                    [Optional] Coin Flipping: Bayesian Approach with Beta Prior</h1>

We’ll now take a Bayesian approach to the coin flipping problem, in which we treat <em>θ </em>as a random variable sampled from some prior distribution <em>p</em>(<em>θ</em>). We’ll represent the <em>i</em>th coin flip by a random variable <em>X<sub>i </sub></em>∈ {0<em>,</em>1}, where <em>X<sub>i </sub></em>= 1 if the <em>i</em>th flip is heads. We assume that the <em>X<sub>i</sub></em>’s are conditionally independent given <em>θ</em>. This means that the joint distribution of the coin flips and <em>θ </em>factorizes as follows:

<table width="400">

 <tbody>

  <tr>

   <td width="101"><em>p</em>(<em>x</em><sub>1</sub><em>,…,x<sub>n</sub>,θ</em>)</td>

   <td width="24">=</td>

   <td width="275"><em>p</em>(<em>θ</em>)<em>p</em>(<em>x</em><sub>1</sub><em>,…,x<sub>n </sub></em>| <em>θ</em>) (always true)</td>

  </tr>

  <tr>

   <td width="101"> </td>

   <td width="24">=</td>

   <td width="275"><em>n </em><em>p</em>(<em>θ</em>)<sup>Y</sup><em>p</em>(<em>x<sub>i </sub></em>| <em>θ</em>) (by conditional independence).</td>

  </tr>

 </tbody>

</table>

<em>i</em>=1

<ol>

 <li>[Optional] Suppose that our prior distribution on <em>θ </em>is Beta(<em>h,t</em>), for some <em>h,t &gt; </em>0. That is, <em>p</em>(<em>θ</em>) ∝ <em>θ<sup>h</sup></em><sup>−1 </sup>(1 − <em>θ</em>)<em><sup>t</sup></em><sup>−1</sup>. Suppose that our sequence of flips D has <em>n<sub>h </sub></em>heads and <em>n<sub>t </sub></em> Show that the posterior distribution for <em>θ </em>is Beta(<em>h </em>+ <em>n<sub>h</sub>,t </em>+ <em>n<sub>t</sub></em>). That is, show that</li>

</ol>

<em>p</em>(<em>θ </em>| D) ∝ <em>θ</em><em>h</em>−1+<em>n</em><em>h </em>(1 − <em>θ</em>)<em>t</em>−1+<em>n</em><em>t </em><em>.</em>

We say that the Beta distribution is conjugate to the Bernoulli distribution since the prior and the posterior are both in the same family of distributions (i.e. both Beta distributions).

<ol start="2">

 <li>[Optional] Give expressions for the MLE, the MAP, and the posterior mean estimates of <em>θ</em>. [Hint: You may use the fact that a Beta(<em>h,t</em>) distribution has mean <em>h/</em>(<em>h </em>+ <em>t</em>) and has mode (<em>h </em>− 1)<em>/</em>(<em>h </em>+ <em>t </em>− 2) for <em>h,t &gt; </em>1. For the Bayesian solutions, you should note that as <em>h </em>+ <em>t </em>gets very large, and assuming we keep the ratio <em>h/</em>(<em>h</em>+<em>t</em>) fixed, the posterior mean and MAP approach the prior mean <em>h/</em>(<em>h </em>+ <em>t</em>), while for fixed <em>h </em>and <em>t</em>, the posterior mean approaches the MLE as the sample size <em>n </em>= <em>n<sub>h </sub></em>+ <em>n<sub>t </sub></em>→ ∞.</li>

 <li>[Optional] What happens to <em>θ</em><sup>ˆ</sup><sub>MLE </sub>, <em>θ</em><sup>ˆ</sup><sub>MAP</sub>, and <em>θ</em><sup>ˆ</sup><sub>POSTERIOR MEAN </sub>as the number of coin flips <em>n </em>= <em>n<sub>h </sub></em>+ <em>n<sub>t </sub></em>approaches infinity?</li>

 <li>[Optional] The MAP and posterior mean estimators of <em>θ </em>were derived from a Bayesian perspective. Let’s now evaluate them from a frequentist perspective. Suppose <em>θ </em>is fixed and unknown. Which of the MLE, MAP, and posterior mean estimators give unbiased estimates of <em>θ</em>, if any? [Hint: The answer may depend on the parameters <em>h </em>and <em>t </em>of the prior. Also, let’s consider the total number of flips <em>n </em>= <em>n<sub>h </sub></em>+ <em>n<sub>t </sub></em>to be given (not random), while <em>n<sub>h </sub></em>and <em>n<sub>t </sub></em>are random, with <em>n<sub>h </sub></em>= <em>n </em>− <em>n<sub>t</sub></em>.]</li>

 <li>[Optional] Suppose somebody gives you a coin and asks you to give an estimate of the probability of heads, but you can only toss the coin 3 You have no particular reason to believe this is an unfair coin. Would you prefer the MLE or the posterior mean as a point estimate of <em>θ</em>? If the posterior mean, what would you use for your prior?</li>

</ol>

<h1>8           [Optional] Hierarchical Bayes for Click-Through Rate Estimation</h1>

In mobile advertising, ads are often displayed inside apps on a phone or tablet device. When an ad is displayed, this is called an “impression.” If the user clicks on the ad, that is called a “click.” The probability that an impression leads to a click is called the “click-through rate” (CTR).

Suppose we have <em>d </em>= 1000 apps. For various reasons<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>, each app tends to have a different overall CTR. For the purposes of designing an ad campaign, we want estimates of all the app-level CTRs, which we’ll denote by <em>θ</em><sub>1</sub><em>,…,θ</em><sub>1000</sub>. Of course, the particular user seeing the impression and the particular ad that is shown have an effect on the CTR, but we’ll ignore these issues for now. [Because so many clicks on mobile ads are accidental, it turns out that the overall app-level CTR often dominates the effect of the particular user and the specific ad.]

If we have enough impressions for a particular app, then the empirical fraction of clicks will give a good estimate for the actual CTR. However, if we have relatively few impressions, we’ll have some problems using the empirical fraction. Typical CTRs are less than 1%, so it takes a fairly large number of observations to get a good estimate of CTR. For example, even with 100 impressions, the only possible CTR estimates given by the MLE would be 0%<em>,</em>1%<em>,</em>2%<em>,…,</em>100%. The 0% estimate is almost certainly much too low, and anything 2% or higher is almost certainly much too high. Our goal is to come up with reasonable point estimates for <em>θ</em><sub>1</sub><em>,…,θ</em><sub>1000</sub>, even when we have very few observations for some apps.

If we wanted to apply the Bayesian approach worked out in the previous problem, we could come up with a prior that seemed reasonable. For example, we could use the following Beta(3<em>,</em>400) as a prior distribution on each <em>θ<sub>i</sub></em>:

In this basic Bayesian approach, the parameters 3 and 400 would be chosen by the data scientist based on prior experience, or “best guess”, but without looking at the new data. Another approach would be to use the data to help you choose the parameters <em>a </em>and <em>b </em>in Beta(<em>a,b</em>). This would not be a Bayesian approach, though it is frequently used in practice. One method in this direction is called empirical Bayes. Empirical Bayes can be considered a frequentist approach, in which we estimate <em>a </em>and <em>b </em>from the data D using some estimation technique, such as maximum likelihood. The proper Bayesian approach to this type of thing is called hierarchical Bayes, in which we put another prior distribution on <em>a </em>and <em>b</em>. We’ll investigate each of these approaches below.

Mathematical Description

We’ll now give a mathematical description of our model, assuming the prior parameters <em>a </em>and <em>b </em>are directly chosen by the data scientist. Let <em>n</em><sub>1</sub><em>,…,n<sub>d </sub></em>be the number of impressions we observe for each of the <em>d </em>apps. In this problem, we will not consider these to be random numbers. For the <em>i</em>th app, letbe indicator variables determining whether or not each impression was clicked. That is, <em>c<sup>j</sup><sub>i </sub></em>= 1(<em>j</em>th impression on <em>i</em>th app was clicked). We can summarize the data on the <em>i</em>th app by is the total number of impressions that were clicked for app <em>i</em>. Let <em>θ </em>= (<em>θ</em><sub>1</sub><em>,…,θ<sub>d</sub></em>), where <em>θ<sub>i </sub></em>is the CTR for app <em>i</em>.

In our Bayesian approach, we act as though the data were generated as follows:

<ol>

 <li>Sample <em>θ</em><sub>1</sub><em>,…,θ<sub>d </sub></em>i.d. from Beta(<em>a,b</em>).</li>

 <li>For each app <em>i</em>, samplei.d. from Bernoulli(<em>θ<sub>i</sub></em>).</li>

</ol>

<h2>8.1           [Optional] Empirical Bayes for a single app</h2>

We start by working out some details for Bayesian inference for a single app. That is, suppose we only have the data D<em><sub>i </sub></em>from app <em>i</em>, and nothing else. Mathematically, this is exactly the same setting as the coin tossing setting above, but here we push it further.

<ol>

 <li>Give an expression for <em>p</em>(D<em><sub>i </sub></em>| <em>θ<sub>i</sub></em>), the likelihood of D<em><sub>i </sub></em>given the probability of click <em>θ<sub>i</sub></em>, in terms of <em>θ<sub>i</sub></em>, <em>x<sub>i </sub></em>and <em>n<sub>i</sub></em>.</li>

 <li>We will take our prior distribution on <em>θ<sub>i </sub></em>to be Beta(<em>a,b</em>). The corresponding probability density function is given by</li>

</ol>

<em>p</em>(<em>θ<sub>i</sub></em>) = Beta<em> ,</em>

where <em>B</em>(<em>a,b</em>) is called the Beta function. Explain (without calculation) why we must have

<em>.</em>

<ol start="3">

 <li>Give an expression for the posterior distribution <em>p</em>(<em>θ<sub>i </sub></em>| D<em><sub>i</sub></em>). In this case, include the constant of proportionality. In other words, do not use the “is proportional to” sign ∝ in your final expression. You may reference the Beta function defined above. [Hint: This problem is essentially a repetition of an earlier problem.]</li>

 <li>Give a closed form expression for <em>p</em>(D<em><sub>i</sub></em>), the marginal likelihood of D<em><sub>i</sub></em>, in terms of the <em>a,b,x<sub>i</sub>, </em>and <em>n<sub>i</sub></em>. You may use the normalization function <em>B</em>(·<em>,</em>) for convenience, but you should not have any integrals in your solution. (Hint: <em>p</em>(D<em><sub>i</sub></em>) = <sup>R </sup><em>p</em>(D<em><sub>i </sub></em>| <em>θ<sub>i</sub></em>)<em>p</em>(<em>θ<sub>i</sub></em>)<em>dθ<sub>i</sub></em>, and the answer will be a ratio of two beta function evaluations.)</li>

 <li>The maximum likelihood estimate for <em>θ<sub>i </sub></em>is <em>x<sub>i</sub>/n<sub>i</sub></em>. Let <em>p</em><sub>MLE</sub>(D<em><sub>i</sub></em>) be the marginal likelihood of D<em><sub>i </sub></em>when we use a prior on <em>θ<sub>i </sub></em>that puts all of its probability mass at <em>x<sub>i</sub>/n<sub>i</sub></em>. Note that</li>

</ol>

<em>p</em>MLE

Figure 1: A plot of <em>p</em>(D<em><sub>i </sub></em>| <em>a,b</em>) as a function of <em>a </em>and <em>b</em>.

Explain why, or prove, that <em>p</em><sub>MLE</sub>(D<em><sub>i</sub></em>) is larger than <em>p</em>(D<em><sub>i</sub></em>) for any other prior we might put on <em>θ<sub>i</sub></em>. If it’s too hard to reason about all possible priors, it’s fine to just consider all Beta priors. [Hint: This does not require much or any calculation. It may help to think about the integral <em>p</em>(D<em><sub>i</sub></em>) = <sup>R </sup><em>p</em>(D<em><sub>i </sub></em>| <em>θ<sub>i</sub></em>)<em>p</em>(<em>θ<sub>i</sub></em>)<em>dθ<sub>i </sub></em>as a weighted average of <em>p</em>(D<em><sub>i </sub></em>| <em>θ<sub>i</sub></em>) for different values of <em>θ<sub>i</sub></em>, where the weights are <em>p</em>(<em>θ<sub>i</sub></em>).]

<ol start="6">

 <li>One approach to getting an empirical Bayes estimate of the parameters <em>a </em>and <em>b </em>is to use maximum likelihood. Such an empirical Bayes estimate is often called an ML-2 estimate, since it’s maximum likelihood, but at a higher level in the Bayesian hierarchy. To emphasize the dependence of the likelihood of D<em><sub>i </sub></em>on the parameters <em>a </em>and <em>b</em>, we’ll now write it as <em>p</em>(D<em><sub>i </sub></em>| <em>a,b</em>)<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a>. The empirical Bayes estimates for <em>a </em>and <em>b </em>are given by</li>

</ol>

(<em>a,</em>ˆ <sup>ˆ</sup><em>b</em>) =         argmax          <em>p</em>(D<em><sub>i </sub></em>| <em>a,b</em>)<em>.</em>

(<em>a,b</em>)∈(0<em>,</em>∞)×(0<em>,</em>∞)

To make things concrete, suppose we observed <em>x<sub>i </sub></em>= 3 clicks out of <em>n<sub>i </sub></em>= 500 impressions. A plot of <em>p</em>(D<em><sub>i </sub></em>| <em>a,b</em>) as a function of <em>a </em>and <em>b </em>is given in Figure 1. It appears from this plot that the likelihood will keep increasing as <em>a </em>and <em>b </em>increase, at least if <em>a </em>and <em>b </em>maintain a particular ratio. Indeed, this likelihood function never attains its maximum, so we cannot use ML-2 here. Explain what’s happening to the prior as we continue to increase the likelihood. [Hint: It is a property of the Beta distribution (not difficult to see), that for any <em>θ </em>∈ (0<em>,</em>1), there is a Beta distribution with expected value <em>θ </em>and variance less than <em>ε</em>, for any <em>ε &gt; </em>0. What’s going in here is similar to what happens when you attempt to fit a gaussian distribution N(<em>µ,σ</em><sup>2</sup>) to a single data point using maximum likelihood.]

<h2>8.2           [Optional] Empirical Bayes Using All App Data</h2>

In the previous section, we considered working with data from a single app. With a fixed prior, such as Beta(3,400), our Bayesian estimates for <em>θ<sub>i </sub></em>seem more reasonable to me<a href="#_ftn7" name="_ftnref7"><sup>[7]</sup></a> than the MLE when our sample size <em>n<sub>i </sub></em>is small. The fact that these estimates seem reasonable is an immediate consequence of the fact that I chose the prior to give high probability to estimates that seem reasonable to me, before ever seeing the data. Our earlier attempt to use empirical Bayes (ML-2) to choose the prior in a data-driven way was not successful. With only a single app, we were essentially overfitting the prior to the data we have. In this section, we’ll consider using the data from all the apps, in which case empirical Bayes makes more sense.

<ol>

 <li>Let D = (D<sub>1</sub><em>,…,</em>D<em><sub>d</sub></em>) be the data from all the apps. Give an expression for <em>p</em>(D | <em>a,b</em>), the marginal likelihood of D. Expression should be in terms of <em>a,b,x<sub>i</sub>,n<sub>i </sub></em>for <em>i </em>= 1<em>,…,d</em>. Assume data from different apps are independent. (Hint: This problem should be easy, based on a problem from the previous section.)</li>

 <li>Explain why <em>p</em>(<em>θ<sub>i </sub></em>| D) = <em>p</em>(<em>θ<sub>i </sub></em>| D<em><sub>i</sub></em>), according to our model. In other words, once we choose values for parameters <em>a </em>and <em>b</em>, information about one app does not give any information about other apps.</li>

 <li>Suppose we have data from 6 apps. 3 of the apps have a fair number of impressions, and 3 have relatively few. Suppose we observe the following:</li>

</ol>

<table width="252">

 <tbody>

  <tr>

   <td width="52"> </td>

   <td width="84">Num Clicks</td>

   <td width="117">Num Impressions</td>

  </tr>

  <tr>

   <td width="52">App 1</td>

   <td width="84">50</td>

   <td width="117">10000</td>

  </tr>

  <tr>

   <td width="52">App 2</td>

   <td width="84">160</td>

   <td width="117">20000</td>

  </tr>

  <tr>

   <td width="52">App 3</td>

   <td width="84">180</td>

   <td width="117">60000</td>

  </tr>

  <tr>

   <td width="52">App 4</td>

   <td width="84">0</td>

   <td width="117">100</td>

  </tr>

  <tr>

   <td width="52">App 5</td>

   <td width="84">0</td>

   <td width="117">5</td>

  </tr>

  <tr>

   <td width="52">App 6</td>

   <td width="84">1</td>

   <td width="117">2</td>

  </tr>

 </tbody>

</table>

Compute the empirical Bayes estimates for <em>a </em>and <em>b</em>. (Recall, this amounts to computing

(<em>a,</em>ˆ <sup>ˆ</sup><em>b</em>) = argmax<sub>(<em>a,b</em>)∈<strong>R</strong></sub><em>&gt;</em><sub>0</sub><sub>×<strong>R</strong></sub><em>&gt;</em><sub>0 </sub><em>p</em>(D | <em>a,b</em>)<em>.</em>) This will require solving an optimization problem, for which you are free to use any optimization software you like (perhaps <a href="https://docs.scipy.org/doc/scipy/reference/optimize.html">scipy.optimize</a> would be useful). The empirical Bayes prior is then Beta(<em>a,</em>ˆ <sup>ˆ</sup><em>b</em>), where <em>a</em>ˆ and <sup>ˆ</sup><em>b </em>are our ML-2 estimates. Give the corresponding prior mean and standard deviation for this prior.

<ol start="4">

 <li>Complete the following table:</li>

</ol>

<table width="522">

 <tbody>

  <tr>

   <td width="52"> </td>

   <td width="79">NumClicks</td>

   <td width="112">NumImpressions</td>

   <td width="45">MLE</td>

   <td width="47">MAP</td>

   <td width="100">PosteriorMean</td>

   <td width="86">PosteriorSD</td>

  </tr>

  <tr>

   <td width="52">App 1</td>

   <td width="79">50</td>

   <td width="112">10000</td>

   <td width="45">0.5%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

  <tr>

   <td width="52">App 2</td>

   <td width="79">160</td>

   <td width="112">20000</td>

   <td width="45">0.8%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

  <tr>

   <td width="52">App 3</td>

   <td width="79">180</td>

   <td width="112">60000</td>

   <td width="45">0.3%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

  <tr>

   <td width="52">App 4</td>

   <td width="79">0</td>

   <td width="112">100</td>

   <td width="45">0%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

  <tr>

   <td width="52">App 5</td>

   <td width="79">0</td>

   <td width="112">5</td>

   <td width="45">0%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

  <tr>

   <td width="52">App 6</td>

   <td width="79">1</td>

   <td width="112">2</td>

   <td width="45">50%</td>

   <td width="47"> </td>

   <td width="100"> </td>

   <td width="86"> </td>

  </tr>

 </tbody>

</table>

Make sure to take a look at the PosteriorSD values and note which are big and which are small.

<h2>8.3           [Optional] Hierarchical Bayes</h2>

In Section 8.2 we managed to get empirical Bayes ML-II estimates for <em>a </em>and <em>b </em>by assuming we had data from multiple apps. However, we didn’t really address the issue that ML-II, as a maximum likelihood method, is prone to overfitting if we don’t have enough data (in this case, enough apps). Moreover, a true Bayesian would reject this approach, since we’re using our data to determine our prior. If we don’t have enough confidence to choose parameters for <em>a </em>and <em>b </em>without looking at the data, then the only proper Bayesian approach is to put another prior on the parameters <em>a </em>and <em>b</em>. If you are very uncertain about values for <em>a </em>and <em>b</em>, you could put priors on them that have high variance.

<ol>

 <li>[Optional] Suppose <em>P </em>is the Beta(<em>a,b</em>) Conceptually, rather than putting priors on <em>a </em>and <em>b</em>, it’s easier to reason about priors on the mean <em>m </em>and the variance <em>v </em>of <em>P</em>. If we parameterize <em>P </em>by its mean <em>m </em>and the variance <em>v</em>, give an expression for the density function Beta(<em>θ</em>;<em>m,v</em>). You are free to use the internet to get this expression – just be confident it’s correct. [Hint: To derive this, you may find it convenient to write some expression in terms of <em>η </em>= <em>a </em>+ <em>b</em>.]</li>

 <li>[Optional] Suggest a prior distribution to put on <em>m </em>and <em>v</em>. [Hint: You might want to use one of the distribution families given <a href="https://davidrosenberg.github.io/mlcourse/Archive/2016/Lectures/10b.conditional-probability-models.pdf#page=6">in this lecture</a><a href="https://davidrosenberg.github.io/mlcourse/Archive/2016/Lectures/10b.conditional-probability-models.pdf#page=6">.</a></li>

 <li>[Optional] Once we have our prior on <em>m </em>and <em>v</em>, we can go “full Bayesian” and compute posterior distributions on <em>θ</em><sub>1</sub><em>,…,θ<sub>d</sub></em>. However, these no longer have closed forms. We would have to use approximation techniques, typically either a Monte Carlo sampling approach or a variational method, which are beyond the scope of this course<a href="#_ftn8" name="_ftnref8"><sup>[8]</sup></a>. After observing the data D, <em>m </em>and <em>v </em>will have some posterior distribution <em>p</em>(<em>m,v </em>| D). We can approximate that distribution by a point mass at the mode of that distribution (<em>m</em><sub>MAP</sub><em>,v</em><sub>MAP</sub>) = argmax<em><sub>m,v </sub>p</em>(<em>m,v </em>| D). Give expressions for the posterior distribution <em>p</em>(<em>θ</em><sub>1</sub><em>,…,θ<sub>d </sub></em>| D), with and without this approximation. You do not need to give any explicit expressions here. It’s fine to have expressions like <em>p</em>(<em>θ</em><sub>1</sub><em>,…,θ<sub>d </sub></em>| <em>m,v</em>) in your solution. Without the approximation, you will probably need some integrals. It’s these integrals that we need sampling or variational approaches to approximate. While one can see this approach as a way to approximate the proper Bayesian approach, one could also be skeptical and say this is just another way to determine your prior from the data. The estimators (<em>m</em><sub>MAP</sub><em>,v</em><sub>MAP</sub>) are often called MAP-II estimators, since they are MAP estimators at a higher level of the Bayesian hierarchy.</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> This problem is based on Section 7.5.3 of Schapire and Freund’s book <em>Boosting: Foundations and Algorithms</em>.

<a href="#_ftnref2" name="_ftn2">[2]</a> Don’t be confused – it’s Bayes as in “Bayes optimal”, as we discussed at the beginning of the course, not Bayesian as we’ve discussed more recently.

<a href="#_ftnref3" name="_ftn3">[3]</a> In this context, the Bayes prediction function is often referred to as the “population minimizer.” In our case, “population” refers to the fact that we are minimizing with respect to the true distribution, rather than a sample. The term “population” arises from the context where we are using a sample to approximate some statistic of an entire population (e.g. a population of people or trees).

<a href="#_ftnref4" name="_ftn4">[4]</a> However, in practice we are usually interested in computing the product of a matrix inverse and a vector, i.e. <em>X</em><sup>−1</sup><em>b</em>. In this case, it’s usually faster and more accurate to use a library’s algorithms for solving a system of linear equations. Note that <em>y </em>= <em>X</em><sup>−1</sup><em>b </em>is just the solution to the linear system <em>Xy </em>= <em>b</em>. See for example <a href="https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/">John Cook’s blog </a><a href="https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/">post</a> for discussion.

<a href="#_ftnref5" name="_ftn5">[5]</a> The primary reason is that different apps place the ads differently, making it more or less difficult to avoid clicking the ad.

<a href="#_ftnref6" name="_ftn6">[6]</a> Note that this is a slight (though common) abuse of notation, because <em>a </em>and <em>b </em>are not random variables in this setting. It might be more appropriate to write this as <em>p</em>(D<em><sub>i</sub></em>;<em>a,b</em>) or <em>p<sub>a,b</sub></em>(D<em><sub>i</sub></em>). But this isn’t very common.

<a href="#_ftnref7" name="_ftn7">[7]</a> I say “to me”, since I am the one who chose the prior. You may have an entirely different prior, and think that my estimates are terrible.

<a href="#_ftnref8" name="_ftn8">[8]</a> If you’re very ambitious, you could try out a package like <a href="https://pystan.readthedocs.io/en/latest/">PyStan</a> to see what happens.