import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from libsvm.svmutil import *


class DTMKL_f:
    """Domain Transfer Multiple Kernel Learning with Prelearned Classifiers (DTMKL_f)
    实现论文第3.3节描述的算法，结合基分类器决策值和虚拟标签
    """

    def __init__(self, X_train_A, Y_train_A, X_train_T, Y_train_T, X_unlabeled_T,
                 kernel_types=['linear', 'rbf'], C=1.0, theta=1e-3, zeta=0.1, eta=1e-3,
                 gamma_rbf=1.0, degree_poly=2, lamda=0.1, epsilon=1e-3, max_iter=10):
        """
        初始化函数，对应论文第3.3节参数设置

        参数：
        X_train_A: 辅助域标记数据特征矩阵 (nA, d)
        Y_train_A: 辅助域标签向量 (nA,)
        X_train_T: 目标域标记数据特征矩阵 (nT_labeled, d)
        Y_train_T: 目标域标签向量 (nT_labeled,)
        X_unlabeled_T: 目标域未标记数据特征矩阵 (nT_unlabeled, d)
        kernel_types: 基核类型列表，支持'linear','rbf','poly'
        C: SVM正则化参数，对应论文公式(15)中的C
        theta: 目标函数中分布差异项的权重，对应论文公式(5)中的θ
        zeta: 虚拟标签正则化参数，对应论文公式(15)中的ζ
        lamda: 未标记数据正则化权重，对应论文公式(15)中的λ
        """
        # 数据初始化
        self.X_train_A = X_train_A
        self.Y_train_A = Y_train_A.reshape(-1, 1)
        self.X_train_T = X_train_T
        self.Y_train_T = Y_train_T.reshape(-1, 1)
        self.X_unlabeled_T = X_unlabeled_T

        # 组合标记数据（辅助域+目标域）
        self.X_labeled = np.vstack([X_train_A, X_train_T])
        self.Y_labeled = np.vstack([self.Y_train_A, self.Y_train_T]).flatten()
        self.nA = len(X_train_A)
        self.nT_labeled = len(X_train_T)
        self.nT_unlabeled = len(X_unlabeled_T)

        # 用于MMD计算的全数据（辅助域+目标域标记+目标域未标记）
        self.X_all_mmd = np.vstack([X_train_A, X_train_T, X_unlabeled_T])
        self.n_total_mmd = len(self.X_all_mmd)

        # 算法参数（论文实验部分推荐值）
        self.C = C
        self.theta = theta
        self.zeta = zeta
        self.lamda = lamda
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.eta = eta

        # 核参数设置
        print("Initial kernel_types:", kernel_types)
        self.kernel_types = kernel_types
        self.M = len(kernel_types)  # 基核数量
        self.gamma_rbf = gamma_rbf
        self.degree_poly = int(degree_poly)  # 确保degree是整数
        # 初始化核权重（均匀分布）
        self.d = np.ones(self.M) / self.M
        print("Initial d:", self.d)

        self.nA_mmd = self.nA
        self.nT_mmd = self.nT_labeled + self.nT_unlabeled

        # 为MMD计算创建域指示向量s
        self.s = self._create_s_vector()

        # 预计算基核矩阵
        print("Precomputing kernel matrices...")
        self.Km_list_labeled = self._precompute_base_kernels(self.X_labeled, self.X_labeled)
        self.Km_list_all = self._precompute_base_kernels(self.X_all_mmd, self.X_all_mmd)

        # 预计算MMD相关项（论文公式(7)）
        self.p = self.compute_mmd_vector()
        print("MMD vector p shape:", self.p.shape)

        # 训练基分类器（论文3.3节描述）
        print("Training base classifiers...")
        self.base_classifiers = self._train_base_classifiers()

        # 初始化SVM参数
        self.alpha = None
        self.b = 0
        self.sv_indices = None
        self.y_v = None  # 虚拟标签将在fit过程中计算

    def _kernel_function(self, X1, X2, kernel_type):
        """计算给定核类型的核矩阵"""
        if kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif kernel_type == 'rbf':
            gamma = 1.0 / (X1.shape[1] * self.gamma_rbf)
            pairwise_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + \
                             np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * pairwise_dists)
        elif kernel_type.startswith("poly"):
            # 从poly2.0获取degree = 2
            degree = int(float(kernel_type[4:]))
            return (np.dot(X1, X2.T) + 1) ** degree
        else:
            raise ValueError(f"不支持的核类型: {kernel_type}")

    def _precompute_base_kernels(self, X1, X2):
        """预计算所有基核矩阵"""
        return [self._kernel_function(X1, X2, kt) for kt in self.kernel_types]

    def _create_s_vector(self):
        """
        创建用于MMD计算的域指示向量
        返回一个向量，其中辅助域的条目为1/nA，目标域的条目为-1/nT
        """
        s_A = np.full(self.nA_mmd, 1.0 / self.nA_mmd)
        s_T = np.full(self.nT_mmd, -1.0 / self.nT_mmd)
        s = np.concatenate([s_A, s_T]).reshape(-1, 1)  # 确保为列向量
        return s

    def compute_mmd_vector(self):
        """计算MMD向量 p = [tr(K1S), ..., tr(KmS)]^T"""
        p = []
        for Km in self.Km_list_all:
            s_matrix = np.dot(self.s, self.s.T)
            product = np.dot(Km, s_matrix)
            trace_val = np.trace(product)
            p.append(trace_val)
        p = np.array(p)
        return p

    def _train_base_classifiers(self):
        """为每种核类型训练基本SVM分类器"""
        base_classifiers = []
        # 组合来自两个域的标记数据用于训练基分类器
        X_combined = np.vstack([self.X_train_A, self.X_train_T])
        y_combined = np.concatenate([self.Y_train_A.flatten(), self.Y_train_T.flatten()])

        for kernel_type in self.kernel_types:
            # 为每个核创建并训练基分类器
            if kernel_type == 'linear':
                clf = SVC(kernel='linear', C=self.C)
                clf.fit(X_combined, y_combined)
                base_classifiers.append(clf)
            elif kernel_type == 'rbf':
                clf = SVC(kernel='rbf', gamma=1.0 / (X_combined.shape[1] * self.gamma_rbf), C=self.C)
                clf.fit(X_combined, y_combined)
                base_classifiers.append(clf)
            elif kernel_type.startswith("poly"):
                degree = int(float(kernel_type[4:]))
                clf = SVC(kernel='poly', degree=degree, C=self.C)
                clf.fit(X_combined, y_combined)
                base_classifiers.append(clf)
            else:
                raise ValueError(f"不支持的核类型: {kernel_type}")
        return base_classifiers

    def _get_base_decisions(self):
        """获取基分类器在未标记目标数据上的决策值"""
        base_decisions = np.zeros((len(self.X_unlabeled_T), len(self.kernel_types)))

        for m, clf in enumerate(self.base_classifiers):
            base_decisions[:, m] = clf.decision_function(self.X_unlabeled_T)
        return base_decisions

    def _get_virtual_labels(self):
        """获取虚拟标签：基分类器决策值的加权组合"""
        base_decisions = self._get_base_decisions()
        return np.dot(base_decisions, self.d.reshape(-1, 1))

    def _compute_structural_risk(self, d):
        """计算结构风险项J(d)（论文公式(16)）"""
        try:
            # 获取虚拟标签
            virtual_labels = self._get_virtual_labels().flatten()

            # 准备训练数据
            X_combined = np.vstack([self.X_labeled, self.X_unlabeled_T])
            n_labeled = len(self.Y_labeled)
            n_unlabeled = len(self.X_unlabeled_T)
            n_total = n_labeled + n_unlabeled

            # 创建标签数组，包括真实标签和虚拟标签
            y_train = np.concatenate([self.Y_labeled, virtual_labels])

            # 计算组合核矩阵
            K_combined = np.zeros((n_total, n_total))
            for m in range(self.M):
                km = self._kernel_function(X_combined, X_combined, self.kernel_types[m])
                K_combined += d[m] * km

            # 添加论文中的正则化项
            reg_diag = np.concatenate([
                np.ones(n_labeled),
                (1.0 / self.lamda) * np.ones(n_unlabeled)
            ])
            K_combined += (1.0 / self.zeta) * np.diag(reg_diag)

            # 使用LIBSVM求解器
            # 将核矩阵转换为适合libsvm的格式
            K_list = []
            for i in range(n_total):
                row = {}
                for j in range(n_total):
                    row[j + 1] = float(K_combined[i, j])
                K_list.append(row)

            # 确保标签是列表
            y_list = y_train.tolist()

            # 设置参数并训练
            prob = svm_problem(y_list, K_list, isKernel=True)
            param = svm_parameter(f'-s 0 -t 4 -c {self.C} -q')

            try:
                model = svm_train(prob, param)

                # 提取支持向量和系数
                sv_indices = [i - 1 for i in model.get_sv_indices()]
                nr_sv = model.get_nr_sv()

                if nr_sv > 0:
                    self.alpha = np.zeros(n_total)
                    # 处理支持向量系数
                    for i in range(nr_sv):
                        idx = sv_indices[i]
                        coef = model.get_sv_coef(i, 0)
                        self.alpha[idx] = coef * y_train[idx]

                    # 获取偏置项
                    self.b = -model.rho[0] if hasattr(model, 'rho') else 0
                    self.sv_indices = sv_indices

                    # 计算对偶目标函数值
                    obj_val = -0.5 * np.dot(self.alpha, np.dot(K_combined, self.alpha))
                    return obj_val
                else:
                    self.alpha = np.zeros(n_total)
                    self.b = 0
                    self.sv_indices = []
                    return 0.0

            except Exception as e:
                # 使用sklearn的SVC作为后备
                n_samples = n_labeled + n_unlabeled
                K_train = np.zeros((n_samples, n_samples + 1))
                K_train[:, 0] = np.arange(1, n_samples + 1)  # 样本索引
                K_train[:, 1:] = K_combined

                clf = SVC(kernel='precomputed', C=self.C)
                clf.fit(K_train, y_train)

                # 提取alpha和b
                dual_coef = clf.dual_coef_[0]
                support = clf.support_

                self.alpha = np.zeros(n_samples)
                for i, idx in enumerate(support):
                    self.alpha[idx] = dual_coef[i]

                self.b = clf.intercept_[0]
                self.sv_indices = support

                # 计算目标函数值
                obj_val = -0.5 * np.dot(self.alpha, np.dot(K_combined, self.alpha))
                return obj_val

        except Exception as e:
            self.alpha = np.zeros(n_labeled + n_unlabeled)
            self.b = 0
            self.sv_indices = []
            return 0.0

    def _compute_gradient(self, d):
        """计算目标函数关于d的梯度"""
        # MMD部分的梯度
        grad_mmd = np.dot(np.outer(self.p, self.p), d)

        # 结构风险项的梯度
        if self.alpha is None or len(self.alpha) == 0:
            return grad_mmd

        X_combined = np.vstack([self.X_labeled, self.X_unlabeled_T])

        # 为每个基核计算梯度
        grad_J = np.zeros(self.M)

        for m in range(self.M):
            Km = self._kernel_function(X_combined, X_combined, self.kernel_types[m])
            # 计算风险项的梯度
            grad_J[m] = -0.5 * np.dot(self.alpha, np.dot(Km, self.alpha))

        return grad_mmd + self.theta * grad_J

    def _compute_hessian(self, d):
        """计算目标函数的Hessian矩阵"""
        # 计算主项
        pp_term = np.outer(self.p, self.p)

        # 添加正则项确保矩阵正定
        reg_term = 1e-3 * np.eye(len(self.p))

        return pp_term + reg_term

    def _project_simplex(self, v):
        """投影到单纯形空间（满足sum(d)=1且d>=0）"""
        # 处理简单情况
        if np.sum(v) == 1.0 and np.all(v >= 0):
            return v

        # 降序排列
        u = np.sort(v)[::-1]
        cumsum_u = np.cumsum(u)

        # 查找rho
        rho = np.nonzero(u > (cumsum_u - 1) / np.arange(1, len(v) + 1))[0]
        if len(rho) == 0:
            return np.ones(len(v)) / len(v)  # 均匀分布

        rho = rho[-1]  # 取最大的rho
        theta = (cumsum_u[rho] - 1.0) / (rho + 1.0)

        # 执行投影
        w = np.maximum(v - theta, 0)
        return w

    def fit(self, verbose=True):
        """主训练循环，对应论文算法1"""
        prev_obj = -np.inf

        for iter in range(self.max_iter):
            # 步骤1：获取虚拟标签
            self.y_v = self._get_virtual_labels().flatten()

            # 步骤2：固定d，优化分类器
            J_val = self._compute_structural_risk(self.d)

            # 步骤3：固定分类器，更新d
            # 计算梯度
            grad = self._compute_gradient(self.d)

            # 计算Hessian矩阵并加入正则项确保正定
            hessian = self._compute_hessian(self.d)

            # 计算更新方向
            try:
                g = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # 如果Hessian矩阵接近奇异，使用梯度下降
                g = grad

            # 更新d
            new_d = self.d - self.eta * g

            # 投影到单纯形空间
            self.d = self._project_simplex(new_d)

            # 计算当前目标函数值
            obj = 0.5 * np.dot(self.d, np.dot(np.outer(self.p, self.p), self.d)) + self.theta * J_val

            # 打印进度
            if verbose:
                print(f"Iter {iter}: Obj={obj:.4f}, d={np.round(self.d, 3)}")

            # 检查收敛性
            if np.abs(obj - prev_obj) < self.epsilon * (np.abs(prev_obj) + 1e-8):
                if verbose:
                    print(f"在第 {iter + 1} 次迭代后收敛。")
                break

            prev_obj = obj

    def predict(self, X_test):
        """预测新样本的类别，严格按照论文公式(20)"""
        # 组合训练数据
        X_combined = np.vstack([self.X_labeled, self.X_unlabeled_T])
        n_test = len(X_test)

        # 计算测试样本与所有训练样本的核矩阵
        K_test = np.zeros((n_test, len(X_combined)))
        for m in range(self.M):
            km_test = self._kernel_function(X_test, X_combined, self.kernel_types[m])
            K_test += self.d[m] * km_test

        # 备选
        if self.alpha is None or len(self.alpha) == 0 or np.all(self.alpha == 0):

            predictions = np.zeros(n_test)
            for m in range(self.M):
                clf_pred = self.base_classifiers[m].predict(X_test)
                predictions += self.d[m] * clf_pred

            return np.sign(predictions)

        # 计算决策值
        if self.sv_indices is not None and len(self.sv_indices) > 0:
            decision_values = np.zeros(n_test)
            for i in self.sv_indices:
                decision_values += self.alpha[i] * K_test[:, i]
            decision_values += self.b
        else:
            decision_values = np.dot(K_test, self.alpha) + self.b

        # 按照论文公式(20)返回符号函数结果
        return np.sign(decision_values)