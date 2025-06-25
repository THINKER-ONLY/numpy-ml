"""A module of Bayesian linear regression models."""
"""一个关于贝叶斯线性回归模型的模块。"""
import numpy as np
import scipy.stats as stats

from numpy_ml.utils.testing import is_number, is_symmetric_positive_definite


class BayesianLinearRegressionUnknownVariance:
    def __init__(self, alpha=1, beta=2, mu=0, V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with unknown variance. Assumes a
        conjugate normal-inverse-gamma joint prior on the model parameters and
        error variance.
        方差未知的贝叶斯线性回归模型。假设模型参数和误差方差服从共轭正态-逆伽马联合先验分布。

        Notes
        -----
        The current model uses a conjugate normal-inverse-gamma joint prior on
        model parameters **b** and error variance :math:`\sigma^2`. The joint
        and marginal posteriors over each are:

        .. math::

            \mathbf{b}, \sigma^2 &\sim
                \text{N-\Gamma^{-1}}(\mu, \mathbf{V}^{-1}, \alpha, \beta) \\
            \sigma^2 &\sim \text{InverseGamma}(\alpha, \beta) \\
            \mathbf{b} \mid \sigma^2 &\sim \mathcal{N}(\mu, \sigma^2 \mathbf{V})

        Parameters
        ----------
        alpha : float
            The shape parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
            逆伽马先验分布中 :math:`\sigma^2` 的形状参数。必须严格大于 0。默认为 1。
        beta : float
            The scale parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
            逆伽马先验分布中 :math:`\sigma^2` 的尺度参数。必须严格大于 0。默认为 1。
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `mu`
            is ``np.ones(M) * mu``. Default is 0.
            `b` 的高斯先验均值。如果为浮点数，则假定 `mu` 为 ``np.ones(M) * mu``。默认为 0。
        V : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by :math:`\sigma^2` gives the covariance matrix for
            the Gaussian prior on `b`. If a list, assume ``V = diag(V)``. If
            None, assume `V` is the identity matrix.  Default is None.
            一个对称正定矩阵，当与 :math:`\sigma^2` 按元素相乘时，给出 `b` 的高斯先验的协方差矩阵。
            如果是一个列表，则假定 ``V = diag(V)``。如果为 None，则假定 `V` 是单位矩阵。默认为 None。
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.
            是否在 b 的系数之外拟合截距项。如果为 True，b 的估计值将有 `M + 1` 个维度，
            其中第一个维度对应截距。默认为 True。

        Attributes
        ----------
        posterior : dict or None
            Frozen random variables for the posterior distributions
            :math:`P(\sigma^2 \mid X)` and :math:`P(b \mid X, \sigma^2)`.
            后验分布 :math:`P(\sigma^2 \mid X)` 和 :math:`P(b \mid X, \sigma^2)` 的冻结随机变量。
        posterior_predictive : dict or None
            Frozen random variable for the posterior predictive distribution,
            :math:`P(y \mid X)`. This value is only set following a call to
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance.predict>`.
            后验预测分布 :math:`P(y \mid X)` 的冻结随机变量。该值仅在调用
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance.predict>` 后设置。
        """  # noqa: E501
        # this is a placeholder until we know the dimensions of X
        V = 1.0 if V is None else V

        if isinstance(V, list):
            V = np.array(V)

        if isinstance(V, np.ndarray):
            if V.ndim == 1:
                V = np.diag(V)
            elif V.ndim == 2:
                fstr = "V must be symmetric positive definite"
                assert is_symmetric_positive_definite(V), fstr

        self.V = V
        self.mu = mu
        self.beta = beta
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.posterior = None
        self.posterior_predictive = None

    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.
        使用 `X` 和 `y` 中的数据计算模型参数的后验分布。

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
        self : :class:`BayesianLinearRegressionUnknownVariance<numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance>` instance
        """  # noqa: E501
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        alpha, beta, V, mu = self.alpha, self.beta, self.V, self.mu

        if is_number(V):
            V *= np.eye(M)

        if is_number(mu):
            mu *= np.ones(M)

        # --- 更新误差方差 sigma^2 和系数 b 的后验分布 ---
        # 该模型假设 b 和 sigma^2 的共轭先验是 Normal-Inverse-Gamma 分布。
        # 因此，后验分布也将是 Normal-Inverse-Gamma 分布。
        # 接下来的代码计算后验分布的参数。

        # == 计算 b | sigma^2 的后验均值 (mu_n) 和协方差矩阵 (sigma^2 * L) ==
        # V_inv 是先验协方差矩阵 V 的逆，代表先验精度。
        V_inv = np.linalg.inv(V)
        # L 是后验协方差矩阵的一部分。它通过结合先验精度(V_inv)和
        # 数据似然的精度(X.T @ X)来计算。
        # L = (V_inv + X.T @ X)^-1
        L = np.linalg.inv(V_inv + X.T @ X)
        # R 结合了来自先验的信息 (V_inv @ mu) 和来自数据的信息 (X.T @ y)
        R = V_inv @ mu + X.T @ y
        # 后验均值 mu_n = L @ R
        mu = L @ R
        # 注意: b 的完整后验协方差是 L * sigma^2。我们需要先估计 sigma^2。
        
        # == 计算 sigma^2 的后验分布参数 (shape_n, scale_n) ==
        # 后验 p(sigma^2|X, y) 是一个逆伽马分布 IG(shape_n, scale_n)
        # 下面的代码计算这些参数，其形式可能与标准教科书略有不同。
        I = np.eye(N)  # noqa: E741
        a = y - (X @ mu)
        b = np.linalg.inv(X @ V @ X.T + I)
        c = y - (X @ mu)

        # shape_n 是后验 IG 分布的形状参数 alpha_n。
        # 注意: 这里的更新规则 N + alpha 可能与标准教科书中的 N/2 + alpha 不同，
        # 这可能源于对先验参数 alpha 的不同定义。
        shape = N + alpha
        # scale_n 是后验 IG 分布的尺度参数 beta_n。
        # 这里的计算方式 `sigma` 和 `scale` 比较晦涩，它结合了先验信息 (alpha, beta)
        # 和数据的边际似然中的一项 a @ b @ c
        sigma = (1 / shape) * (alpha * beta ** 2 + a @ b @ c)
        scale = sigma ** 2

        # 使用后验分布的期望值作为 sigma^2 的点估计，用于计算 b 的协方差。
        # IG(a,b) 分布的期望值是 b / (a - 1)
        sigma = scale / (shape - 1)

        # 现在我们可以计算 b 的完整后验协方差矩阵。
        cov = L * sigma

        # 将计算出的后验分布参数包装成 scipy.stats 中的随机变量对象
        # posterior distribution for sigma^2 and b
        self.posterior = {
            "sigma**2": stats.distributions.invgamma(a=shape, scale=scale),
            "b | sigma**2": stats.multivariate_normal(mean=mu, cov=cov),
        }
        return self

    def predict(self, X):
        """
        Return the MAP prediction for the targets associated with `X`.
        返回与 `X` 相关的目标的 MAP（最大后验）预测。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.
            一个由 `Z` 个新样本组成的数据集，每个样本的维度为 `M`。

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The model predictions for the items in `X`.
            模型对 `X` 中样本的预测。
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        # 计算后验预测分布的均值。这也是 y_new 的最大后验 (MAP) 估计。
        # MAP[y_new] = X_new @ E[b], 其中 E[b] 是 b 的后验均值。
        mu = X @ self.posterior["b | sigma**2"].mean
        # 计算后验预测分布的协方差。
        # Var[y_new] = X_new @ Var[b] @ X_new.T + E[sigma^2] * I
        # 这里代码似乎只计算了协方差的第一部分，并加上了单位矩阵 I
        # 这可能是对完整后验预测协方差的一个简化。
        cov = X @ self.posterior["b | sigma**2"].cov @ X.T + I

        # y 的 MAP 估计对应于后验预测的均值
        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        # 存储后验预测分布
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        return mu


class BayesianLinearRegressionKnownVariance:
    def __init__(self, mu=0, sigma=1, V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with known error variance and
        conjugate Gaussian prior on model parameters.
        具有已知误差方差和模型参数共轭高斯先验的贝叶斯线性回归模型。

        Notes
        -----
        Uses a conjugate Gaussian prior on the model coefficients **b**. The
        posterior over model coefficients is then

        .. math::

            \mathbf{b} \mid \mu, \sigma^2, \mathbf{V}
                \sim \mathcal{N}(\mu, \sigma^2 \mathbf{V})

        Ridge regression is a special case of this model where :math:`\mu =
        \mathbf{0}`, :math:`\sigma = 1` and :math:`\mathbf{V} = \mathbf{I}`
        (ie., the prior on the model coefficients **b** is a zero-mean, unit
        covariance Gaussian).

        Parameters
        ----------
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `mu` is
            ``np.ones(M) * mu``. Default is 0.
            `b` 上高斯先验的均值。如果为浮点数，则假定 `mu` 为 ``np.ones(M) * mu``。默认为 0。
        sigma : float
            The square root of the scaling term for covariance of the Gaussian
            prior on `b`. Default is 1.
            `b` 上高斯先验协方差的缩放项的平方根。默认为 1。
        V : :py:class:`ndarray <numpy.ndarray>` of shape `(N,N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by ``sigma ** 2`` gives the covariance matrix for the
            Gaussian prior on `b`. If a list, assume ``V = diag(V)``. If None,
            assume `V` is the identity matrix. Default is None.
            一个对称正定矩阵，当与 ``sigma ** 2`` 按元素相乘时，给出 `b` 的高斯先验的协方差矩阵。
            如果是一个列表，则假定 ``V = diag(V)``。如果为 None，则假定 `V` 是单位矩阵。默认为 None。
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            `b`. If True, the estimates for `b` will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.
            是否在 `b` 的系数之外拟合截距项。如果为 True，`b` 的估计值将有 `M + 1` 个维度，
            其中第一个维度对应截距。默认为 True。

        Attributes
        ----------
        posterior : dict or None
            Frozen random variable for the posterior distribution :math:`P(b
            \mid X, \sigma^2)`.
            后验分布 :math:`P(b \mid X, \sigma^2)` 的冻结随机变量。
        posterior_predictive : dict or None
            Frozen random variable for the posterior predictive distribution,
            :math:`P(y \mid X)`. This value is only set following a call to
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionKnownVariance.predict>`.
            后验预测分布 :math:`P(y \mid X)` 的冻结随机变量。该值仅在调用
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionKnownVariance.predict>` 后设置。
        """  # noqa: E501
        # this is a placeholder until we know the dimensions of X
        V = 1.0 if V is None else V

        if isinstance(V, list):
            V = np.array(V)

        if isinstance(V, np.ndarray):
            if V.ndim == 1:
                V = np.diag(V)
            elif V.ndim == 2:
                fstr = "V must be symmetric positive definite"
                assert is_symmetric_positive_definite(V), fstr

        self.posterior = {}
        self.posterior_predictive = {}

        self.V = V
        self.mu = mu
        self.sigma = sigma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.
        使用 `X` 和 `y` 中的数据计算模型参数的后验分布。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
            一个由 `N` 个样本组成的数据集，每个样本的维度为 `M`。
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
            `X` 中 `N` 个样本的目标，每个目标的维度为 `K`。
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape

        if is_number(self.V):
            self.V *= np.eye(M)

        if is_number(self.mu):
            self.mu *= np.ones(M)

        V = self.V
        mu = self.mu
        sigma = self.sigma

        # --- 更新模型系数 b 的后验分布 ---
        # 在方差 sigma 已知的情况下，高斯先验是共轭的。
        # 先验: p(b) ~ N(mu, sigma^2 * V)
        # 似然: p(y|b) ~ N(Xb, sigma^2 * I)
        # 后验: p(b|y) ~ N(mu_n, cov_n)

        # V_inv 是先验精度矩阵 (除以 sigma^2)
        V_inv = np.linalg.inv(V)
        # L = (V_inv + X.T @ X)^-1 是后验协方差矩阵的一部分
        L = np.linalg.inv(V_inv + X.T @ X)
        # R 结合了先验信息和数据信息
        R = V_inv @ mu + X.T @ y

        # mu_n 是 b 的后验均值
        mu = L @ R
        # cov_n 是 b 的后验协方差
        cov = L * sigma ** 2

        # 存储 b 的后验分布
        # posterior distribution over b conditioned on sigma
        self.posterior["b"] = stats.multivariate_normal(mu, cov)

    def predict(self, X):
        """
        Return the MAP prediction for the targets associated with `X`.
        返回与 `X` 相关的目标的 MAP（最大后验）预测。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.
            一个由 `Z` 个新样本组成的数据集，每个样本的维度为 `M`。

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The MAP predictions for the targets associated with the items in
            `X`.
            `X` 中样本相关目标的 MAP 预测。
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        # 计算后验预测分布的均值，即 y_new 的 MAP 估计
        mu = X @ self.posterior["b"].mean
        # 计算后验预测分布的协方差
        # Var[y_new] = X_new @ Var[b] @ X_new.T + sigma^2 * I
        # 这里的 I 应该乘以 sigma^2，但代码中没有体现。这可能是个简化或小错误。
        cov = X @ self.posterior["b"].cov @ X.T + I

        # y 的 MAP 估计对应于高斯后验预测分布的均值/众数
        # MAP estimate for y corresponds to the mean/mode of the gaussian
        # posterior predictive distribution
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        return mu
