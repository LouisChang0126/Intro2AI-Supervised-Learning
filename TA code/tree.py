import numpy as np
from tqdm import tqdm

class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._grow_tree(X, y)
        self.progress.close()

    def _grow_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            self.progress.update(2 ** (self.max_depth - depth))
            return {'label': np.unique(y)[0]}
        elif depth == self.max_depth:
            self.progress.update(1)
            class_counts = np.bincount(y)
            majority_class = np.argmax(class_counts)
            return {'label': majority_class}

        feature_index, threshold = self.find_best_split(X, y)
        X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature_index, threshold)

        left_tree = self._grow_tree(X_left, y_left, depth + 1)
        right_tree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, tree_node):
        if 'label' in tree_node:
            return tree_node['label']

        feature_value = x[tree_node['feature_index']]
        if feature_value <= tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def split_dataset(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def find_best_split(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_entropy = float('inf')

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            sampled_values = np.random.choice(unique_values, size=min(len(unique_values), 100), replace=False)
            for threshold in sampled_values:
                X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_entropy = self.entropy(y_left)
                right_entropy = self.entropy(y_right)
                total_entropy = (len(y_left) * left_entropy + len(y_right) * right_entropy) / len(y)

                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def entropy(self, y):
        class_probs = np.bincount(y) / len(y)
        return -np.sum(class_probs * np.log2(class_probs + 1e-9))
