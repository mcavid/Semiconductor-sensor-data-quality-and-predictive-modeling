import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import os


class ETLPipeline:
    def __init__(
        self,
        raw_path="data/raw/secom.data",
        label_path="data/raw/secom_labels.data",
        processed_path="data/processed/",
        pca_components=None,
        variance_threshold=0.0,
        corr_threshold=0.95
    ):
        self.raw_path = raw_path
        self.label_path = label_path
        self.processed_path = processed_path
        self.pca_components = pca_components
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

    def load_data(self):
        print("Loading data...")
        X = pd.read_csv(self.raw_path, sep=" ", header=None)
        y = pd.read_csv(self.label_path, header=None)
        y = y.replace(-1, 0)  # Convert -1/1 labels → 0/1
        return X, y

    def handle_missing(self, X):
        print("Handling missing values...")
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns)

    def scale_data(self, X):
        print("Scaling data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def remove_low_variance(self, X):
        print("Removing low-variance features...")
        selector = VarianceThreshold(self.variance_threshold)
        X_sel = selector.fit_transform(X)
        cols = X.columns[selector.get_support()]
        return pd.DataFrame(X_sel, columns=cols)

    def remove_correlated(self, X):
        print("Removing highly correlated features...")
        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns if any(upper[column] > self.corr_threshold)
        ]

        X_filtered = X.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} highly correlated features.")
        return X_filtered

    def apply_pca(self, X):
        if self.pca_components is None:
            return X

        print("Applying PCA...")
        pca = PCA(n_components=self.pca_components)
        X_pca = pca.fit_transform(X)
        cols = [f"PC{i+1}" for i in range(self.pca_components)]
        return pd.DataFrame(X_pca, columns=cols)

    def run(self):
        X, y = self.load_data()
        X = self.handle_missing(X)
        X = self.scale_data(X)
        X = self.remove_low_variance(X)
        X = self.remove_correlated(X)
        X = self.apply_pca(X)

        # Save processed data
        processed_file = os.path.join(self.processed_path, "processed_data.csv")
        labels_file = os.path.join(self.processed_path, "labels.csv")

        X.to_csv(processed_file, index=False)
        y.to_csv(labels_file, index=False)

        print("ETL pipeline complete.")
        print(f"Saved: {processed_file}")
        print(f"Saved: {labels_file}")

        return X, y
