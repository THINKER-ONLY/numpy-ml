"""Ridge regression module"""
"""岭回归模块"""

import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1, fit_intercept=True):
        r"""
        A ridge regression model with maximum likelihood fit via the normal
        equations.
        一个通过正规方程进行最大似然拟合的岭回归模型。

        Notes
        -----
        Ridge regression is a biased estimator for linear models which adds an
        additional penalty proportional to the L2-norm of the model
        coefficients to the standard mean-squared-error loss:

        .. math::

            \mathcal{L}_{Ridge} = (\mathbf{y} - \mathbf{X} \beta)^\top
                (\mathbf{y} - \mathbf{X} \beta) + \alpha ||\beta||_2^2

        where :math:`\alpha` is a weight controlling the severity of the
        penalty.

        Given data matrix **X** and target vector **y**, the maximum-likelihood
        estimate for ridge coefficients, :math:`\beta`, is:

        .. math::

            \hat{\beta} =
                \left(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I} \right)^{-1}
                    \mathbf{X}^\top \mathbf{y}

        It turns out that this estimate for :math:`\beta` also corresponds to
        the MAP estimate if we assume a multivariate Gaussian prior on the
        model coefficients, assuming that the data matrix **X** has been
        standardized and the target values **y** centered at 0:

        .. math::

            \beta \sim \mathcal{N}\left(\mathbf{0}, \frac{1}{2M} \mathbf{I}\right)

        Parameters
        ----------
        alpha : float
            L2 regularization coefficient. Larger values correspond to larger
            penalty on the L2 norm of the model coefficients. Default is 1.
            L2 正则化系数。值越大，对模型系数的 L2 范数的惩罚就越大。默认为 1。
        fit_intercept : bool
            Whether to fit an additional intercept term. Default is True.
            是否拟合额外的截距项。默认为 True。

        Attributes
        ----------
        beta : :py:class:`ndarray <numpy.ndarray>` of shape `(M, K)` or None
            Fitted model coefficients.
            拟合后的模型系数。
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood.
        通过最大似然估计拟合回归系数。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
            一个由 `N` 个样本组成的数据集，每个样本的维度为 `M`。
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
            `X` 中 `N` 个样本的目标，每个目标的维度为 `K`。

        Returns
        -------
        self : :class:`RidgeRegression <numpy_ml.linear_models.RidgeRegression>` instance
        """  # noqa: E501
        # 如果需要拟合截距，则将 X 转换为设计矩阵
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # A 是 L2 惩罚项 alpha * I
        A = self.alpha * np.eye(X.shape[1])
        # 计算岭回归的伪逆矩阵 (X.T * X + alpha * I)^-1 * X.T
        pseudo_inverse = np.linalg.inv(X.T @ X + A) @ X.T
        # 计算系数 beta
        self.beta = pseudo_inverse @ y
        return self

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points.
        使用训练好的模型对新的数据点集合进行预测。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.
            一个包含 `Z` 个新样本的数据集，每个样本的维度为 `M`。

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The model predictions for the items in `X`.
            模型对 `X` 中样本的预测值。
        """
        # 如果模型包含截距项，在 X 前面加上一列 1
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # 计算预测值 y = X * beta
        return np.dot(X, self.beta)
