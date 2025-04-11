import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from last_DTMKL import DTMKL_f



def load_newsgroup_data(setting='comp_vs_rec'):
    """
    Load and prepare 20 Newsgroups dataset for domain adaptation experiment.
    
    Parameters:
    - setting: Which setting to use ('comp_vs_rec', 'comp_vs_sci', or 'comp_vs_talk')
    
    Returns:
    - Data for auxiliary and target domains
    """
    print(f"Loading {setting} data...")
    
    # Define category mappings according to the paper's Table 1
    settings = {
        'comp_vs_rec': {
            'auxiliary': ['comp.windows.x', 'rec.sport.hockey'],
            'target': ['comp.sys.ibm.pc.hardware', 'rec.motorcycles'],
            'positive_class': 'comp',
            'negative_class': 'rec'
        },
        'comp_vs_sci': {
            'auxiliary': ['comp.windows.x', 'sci.crypt'],
            'target': ['comp.sys.ibm.pc.hardware', 'sci.med'],
            'positive_class': 'comp',
            'negative_class': 'sci'
        },
        'comp_vs_talk': {
            'auxiliary': ['comp.windows.x', 'talk.politics.mideast'],
            'target': ['comp.sys.ibm.pc.hardware', 'talk.politics.guns'],
            'positive_class': 'comp',
            'negative_class': 'talk'
        }
    }
    
    # Get the specific setting
    current_setting = settings[setting]
    
    # Load the auxiliary domain data
    auxiliary_categories = current_setting['auxiliary']
    auxiliary_data = fetch_20newsgroups(subset='all', 
                                       categories=auxiliary_categories,
                                       shuffle=True, 
                                       random_state=42)
    
    # Load the target domain data
    target_categories = current_setting['target']
    target_data = fetch_20newsgroups(subset='all', 
                                    categories=target_categories,
                                    shuffle=True, 
                                    random_state=42)
    
    # Create labels
    # Positive class (comp) = 1, Negative class (rec/sci/talk) = -1
    auxiliary_labels = []
    for target_idx in auxiliary_data.target:
        category = auxiliary_data.target_names[target_idx]
        if category.startswith(current_setting['positive_class']):
            auxiliary_labels.append(1)
        else:
            auxiliary_labels.append(-1)
    auxiliary_labels = np.array(auxiliary_labels)

    target_labels = []
    for target_idx in target_data.target:
        category = target_data.target_names[target_idx]
        if category.startswith(current_setting['positive_class']):
            target_labels.append(1)
        else:
            target_labels.append(-1)
    target_labels = np.array(target_labels)
    

    
    # Create feature vectors - use TF-IDF on the text data
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, sublinear_tf=True, use_idf=True)
    
    # Fit the vectorizer on all data
    all_texts = auxiliary_data.data + target_data.data
    vectorizer.fit(all_texts)
    
    # Transform the auxiliary and target data
    auxiliary_features = vectorizer.transform(auxiliary_data.data).toarray()
    target_features = vectorizer.transform(target_data.data).toarray()
    
    print(f"Auxiliary domain: {len(auxiliary_labels)} samples")
    print(f"Target domain: {len(target_labels)} samples")
    
    return auxiliary_features, auxiliary_labels, target_features, target_labels
X_aux, y_aux, X_tar, y_tar = load_newsgroup_data('comp_vs_rec')
print("X_aux",X_aux)
print("y_aux",y_aux)
print("X_tar",X_tar)
print("y_tar",y_tar)

        # 随机选择标记样本
pos_indices = np.where(y_tar == 1)[0]
neg_indices = np.where(y_tar == -1)[0]
        
np.random.seed(123)  # 设置随机种子以确保可重复性
labeled_samples_per_class=5
# 确保选择的样本数不超过可用样本数
n_pos = min(labeled_samples_per_class, len(pos_indices))
n_neg = min(labeled_samples_per_class, len(neg_indices))
        
if n_pos < labeled_samples_per_class or n_neg < labeled_samples_per_class:
    print(f"Warning: Requested {labeled_samples_per_class} samples per class, but only found {n_pos} positive and {n_neg} negative samples")
        
pos_labeled_idx = np.random.choice(pos_indices, n_pos, replace=False)
neg_labeled_idx = np.random.choice(neg_indices, n_neg, replace=False)
        
labeled_idx = np.concatenate([pos_labeled_idx, neg_labeled_idx])
unlabeled_idx = np.array([i for i in range(len(y_tar)) if i not in labeled_idx])
        
# 分割目标域数据为标记和未标记
X_tar_labeled = X_tar[labeled_idx]
y_tar_labeled = y_tar[labeled_idx]
X_tar_unlabeled = X_tar[unlabeled_idx]
y_tar_unlabeled = y_tar[unlabeled_idx]  # 用于评估的真实标签

print("X_aux",X_aux.shape)
print("y_aux",y_aux.shape)
print("X_tar_labeled",X_tar_labeled.shape)
print("y_tar_labeled",y_tar_labeled.shape)
print("X_tar_unlabeled",X_tar_unlabeled.shape)


#kernel_types = ['linear', 'poly1.5', 'poly1.6', 'poly1.7', 'poly1.8', 'poly1.9', 'poly2.0']
kernel_types = ['linear', 'poly2.0']


dtmkl = DTMKL_f(
      X_train_A=X_aux,
      Y_train_A=y_aux, 
      X_train_T=X_tar_labeled,
      Y_train_T=y_tar_labeled,
      X_unlabeled_T=X_tar_unlabeled,
      kernel_types=kernel_types,
      C=5,
      theta = 2e-3,
      eta=2e-3,
      max_iter=10  # 减少迭代次数
)
                        
                        
# 训练DTMKL模型
dtmkl.fit(verbose=False)
                        
# 预测
y_pred = dtmkl.predict(X_tar_unlabeled)

z = 0
for i in y_pred:
    if i == 1:
        z += 1
print(z)


print("y_pred",y_pred)
print((len(y_pred)-np.count_nonzero(y_pred-y_tar_unlabeled))/len(y_pred))

'''clf = SVC(kernel='linear', C=5)
clf.fit(X_tar_labeled, y_tar_labeled)
y_pred1 = clf.predict(X_tar_unlabeled)
print("ypre-svm",y_pred1)
print((len(y_pred1)-np.count_nonzero(y_pred1-y_tar_unlabeled))/len(y_pred1))     '''


 
        
