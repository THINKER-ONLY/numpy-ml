"""A module for naive Bayes classifiers"""
"""朴素贝叶斯分类器模块"""
import numpy as np


class GaussianNBClassifier:
    def __init__(self, eps=1e-6):
        r"""
        A naive Bayes classifier for real-valued data.
        一个用于实值数据的高斯朴素贝叶斯分类器。

        Notes
        -----
        The naive Bayes model assumes the features of each training example
        :math:`\mathbf{x}` are mutually independent given the example label
        *y*:

        .. math::

            P(\mathbf{x}_i \mid y_i) = \prod_{j=1}^M P(x_{i,j} \mid y_i)

        where :math:`M` is the rank of the :math:`i^{th}` example
        :math:`\mathbf{x}_i` and :math:`y_i` is the label associated with the
        :math:`i^{th}` example.

        Combining the conditional independence assumption with a simple
        application of Bayes' theorem gives the naive Bayes classification
        rule:

        .. math::

            \hat{y} &= \arg \max_y P(y \mid \mathbf{x}) \\
                    &= \arg \max_y  P(y) P(\mathbf{x} \mid y) \\
                    &= \arg \max_y  P(y) \prod_{j=1}^M P(x_j \mid y)

        In the final expression, the prior class probability :math:`P(y)` can
        be specified in advance or estimated empirically from the training
        data.

        In the Gaussian version of the naive Bayes model, the feature
        likelihood is assumed to be normally distributed for each class:

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)

        where :math:`\theta` is the set of model parameters: :math:`\{\mu_1,
        \Sigma_1, \ldots, \mu_K, \Sigma_K\}`, :math:`K` is the total number of
        unique classes present in the data, and the parameters for the Gaussian
        associated with class :math:`c`, :math:`\mu_c` and :math:`\Sigma_c`
        (where :math:`1 \leq c \leq K`), are estimated via MLE from the set of
        training examples with label :math:`c`.

        Parameters
        ----------
        eps : float
            A value added to the variance to prevent numerical error. Default
            is 1e-6.
            加到方差上的一个值，以防止数值计算错误。默认为 1e-6。

        Attributes
        ----------
        parameters : dict
            Dictionary of model parameters: "mean", the `(K, M)` array of
            feature means under each class, "sigma", the `(K, M)` array of
            feature variances under each class, and "prior", the `(K,)` array of
            empirical prior probabilities for each class label.
            模型参数字典： "mean"，每个类别下特征均值的 `(K, M)` 数组；"sigma"，每个类别下特征方差的
            `(K, M)` 数组；"prior"，每个类别标签的经验先验概率的 `(K,)` 数组。
        hyperparameters : dict
            Dictionary of model hyperparameters
            模型超参数字典
        labels : :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
            An array containing the unique class labels for the training
            examples.
            一个包含训练样本唯一类别标签的数组。
        """
        self.labels = None
        self.hyperparameters = {"eps": eps}
        self.parameters = {
            "mean": None,  # shape: (K, M)
            "sigma": None,  # shape: (K, M)
            "prior": None,  # shape: (K,)
        }

    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.
        通过最大似然估计拟合模型参数。

        Notes
        -----
        The model parameters are stored in the :py:attr:`parameters
        <numpy_ml.linear_models.GaussianNBClassifier.parameters>` attribute.
        The following keys are present:

            "mean": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature means for each of the `K` label classes
            "sigma": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature variances for each of the `K` label classes
            "prior": :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
                Prior probability of each of the `K` label classes, estimated
                empirically from the training data

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
            一个包含 `N` 个样本的数据集，每个样本的维度为 `M`。
        y: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`
            `X` 中 `N` 个样本的类别标签。

        Returns
        -------
        self : :class:`GaussianNBClassifier <numpy_ml.linear_models.GaussianNBClassifier>` instance
        """  # noqa: E501
        P = self.parameters
        H = self.hyperparameters

        # 获取所有唯一的类别标签
        self.labels = np.unique(y)

        K = len(self.labels)
        N, M = X.shape

        # 初始化均值、方差和先验概率矩阵
        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))

        # 对每个类别，计算特征的均值、方差和类别的先验概率
        for i, c in enumerate(self.labels):
            X_c = X[y == c, :]

            P["mean"][i, :] = np.mean(X_c, axis=0)
            P["sigma"][i, :] = np.var(X_c, axis=0) + H["eps"]
            P["prior"][i] = X_c.shape[0] / N
        return self

    def predict(self, X):
        """
        Use the trained classifier to predict the class label for each example
        in **X**.
        使用训练好的分类器为 **X** 中的每个样本预测类别标签。

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`
            一个包含 `N` 个样本的数据集，每个样本的维度为 `M`。

        Returns
        -------
        labels : :py:class:`ndarray <numpy.ndarray>` of shape `(N)`
            The predicted class labels for each example in `X`
            `X` 中每个样本的预测类别标签。
        """
        # 计算每个样本在所有类别上的对数后验概率，并返回概率最大的类别标签
        return self.labels[self._log_posterior(X).argmax(axis=1)]

    def _log_posterior(self, X):
        r"""
        Compute the (unnormalized) log posterior for each class.
        计算每个类别的（未归一化）对数后验概率。

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`
            一个包含 `N` 个样本的数据集，每个样本的维度为 `M`。

        Returns
        -------
        log_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            Unnormalized log posterior probability of each class for each
            example in `X`
            `X` 中每个样本的每个类别的未归一化对数后验概率。
        """
        K = len(self.labels)
        log_posterior = np.zeros((X.shape[0], K))
        # 对每个类别，计算所有样本属于该类别的对数后验概率
        for i in range(K):
            log_posterior[:, i] = self._log_class_posterior(X, i)
        return log_posterior

    def _log_class_posterior(self, X, class_idx):
        r"""
        Compute the (unnormalized) log posterior for the label at index
        `class_idx` in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`.
        计算在 :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>` 中
        索引为 `class_idx` 的标签的（未归一化）对数后验概率。

        Notes
        -----
        Unnormalized log posterior for example :math:`\mathbf{x}_i` and class
        :math:`c` is::

        .. math::

            \log P(y_i = c \mid \mathbf{x}_i, \theta)
                &\propto \log P(y=c \mid \theta) +
                    \log P(\mathbf{x}_i \mid y_i = c, \theta) \\
                &\propto \log P(y=c \mid \theta)
                    \sum{j=1}^M \log P(x_j \mid y_i = c, \theta)

        In the Gaussian naive Bayes model, the feature likelihood for class
        :math:`c`, :math:`P(\mathbf{x}_i \mid y_i = c, \theta)` is assumed to
        be normally distributed

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`
            一个包含 `N` 个样本的数据集，每个样本的维度为 `M`。
        class_idx : int
            The index of the current class in :py:attr:`labels`
            当前类别在 :py:attr:`labels` 中的索引。

        Returns
        -------
        log_class_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            Unnormalized log probability of the label at index `class_idx`
            in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`
            for each example in `X`
            对于 `X` 中的每个样本，在 :py:attr:`labels` 中索引为 `class_idx` 的标签的未归一化对数概率。
        """  # noqa: E501
        P = self.parameters
        mu = P["mean"][class_idx]
        prior = P["prior"][class_idx]
        sigsq = P["sigma"][class_idx]

        # 计算高斯分布的对数似然 log P(X | C)
        # log likelihood = log P(X | N(mu, sigsq))
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
        log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
        # 返回对数后验概率 log P(C | X) = log P(X | C) + log P(C)
        return log_likelihood + np.log(prior)
