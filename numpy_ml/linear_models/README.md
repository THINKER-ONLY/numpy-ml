# Linear Models
The `linear_models` module includes:

1. [OLS linear regression](https://en.wikipedia.org/wiki/Ordinary_least_squares) with maximum likelihood parameter estimates via the normal equation. 
    - Includes optional weight arguments for [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares)
    - Supports batch and online coefficient updates.
3. [Ridge regression / Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization)
   with maximum likelihood parameter estimates via the normal equation.
2. [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) with maximum likelihood parameter estimates via gradient descent.
3. [Bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) with maximum a posteriori parameter estimates via [conjugacy](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)
    - Known coefficient prior mean and known error variance
    - Known coefficient prior mean and unknown error variance
4. [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) with Gaussian feature likelihoods.
5. [Generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) with identity, log, and logit link functions.

## Plots
<p align="center">
<img src="img/plot_logistic.png" align='center' height="550" />

<img src="img/plot_bayes.png" align='center' height="300" />

<img src="img/plot_regression.png" align='center' height="550" />
</p>

# Linear Models
# 线性模型

The `linear_models` module includes:
The `linear_models` 模块包含以下内容：

1. [OLS linear regression](https://en.wikipedia.org/wiki/Ordinary_least_squares) with maximum likelihood parameter estimates via the normal equation.
   1. [普通最小二乘法 (OLS) 线性回归](https://en.wikipedia.org/wiki/Ordinary_least_squares)：通过正规方程 (normal equation) 获得最大似然参数估计。
    - Includes optional weight arguments for [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares)
      - 包含可选的权重参数，用于实现 [加权最小二乘法](https://en.wikipedia.org/wiki/Weighted_least_squares)。
    - Supports batch and online coefficient updates.
      - 支持批量和在线（逐个样本）的系数更新。
3. [Ridge regression / Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization)
   with maximum likelihood parameter estimates via the normal equation.
   2. [岭回归 (Ridge regression) /吉洪诺夫正则化](https://en.wikipedia.org/wiki/Tikhonov_regularization)：通过正规方程获得最大似然参数估计。
2. [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) with maximum likelihood parameter estimates via gradient descent.
   3. [逻辑回归 (Logistic regression)](https://en.wikipedia.org/wiki/Logistic_regression)：通过梯度下降获得最大似然参数估计。
3. [Bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) with maximum a posteriori parameter estimates via [conjugacy](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)
   4. [贝叶斯线性回归](https://en.wikipedia.org/wiki/Bayesian_linear_regression)：通过[共轭先验](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)获得最大后验 (MAP) 参数估计。
    - Known coefficient prior mean and known error variance
      - 场景一：系数的先验均值已知，误差的方差也已知。
    - Known coefficient prior mean and unknown error variance
      - 场景二：系数的先验均值已知，但误差的方差未知。
4. [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) with Gaussian feature likelihoods.
   5. [高斯朴素贝叶斯分类器](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)：假设特征的似然函数服从高斯分布。
5. [Generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) with identity, log, and logit link functions.
   6. [广义线性模型 (Generalized linear model)](https://en.wikipedia.org/wiki/Generalized_linear_model)：支持恒等 (identity)、对数 (log) 和 logit 链接函数。

## Plots
## 图例

<p align="center">
<img src="img/plot_logistic.png" align='center' height="550" />

<img src="img/plot_bayes.png" align='center' height="300" />

<img src="img/plot_regression.png" align='center' height="550" />
</p>