from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA  # Added PCA import
import pickle
import warnings


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""
    
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> 'BaseProcessor':
        """Fit the processor on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform the data using fitted parameters."""
        pass
    
    def fit_transform(self, data: pd.Series, **kwargs) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(data, **kwargs).transform(data)


class MissingValueProcessor(BaseProcessor):
    """Handle missing values with various strategies."""
    
    def __init__(self, strategy: str = 'mean'):
        self.strategy = strategy
        self.fill_value = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'MissingValueProcessor':
        if self.strategy == 'mean':
            self.fill_value = data.mean()
        elif self.strategy == 'median':
            self.fill_value = data.median()
        elif self.strategy == 'mode':
            mode_val = data.mode()
            self.fill_value = mode_val.iloc[0] if not mode_val.empty else 0
        elif isinstance(self.strategy, (int, float)):
            self.fill_value = self.strategy
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if pd.isna(self.fill_value):
            warnings.warn(f"Fill value is NaN for strategy '{self.strategy}'")
            self.fill_value = 0
        
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        return data.fillna(self.fill_value)


class OutlierProcessor(BaseProcessor):
    """Handle outliers using IQR or Z-score methods."""
    
    def __init__(self, method: str = 'iqr', action: str = 'clip', threshold: float = 3.0):
        self.method = method
        self.action = action
        self.threshold = threshold
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'OutlierProcessor':
        if self.method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - 1.5 * IQR
            self.upper_bound = Q3 + 1.5 * IQR
        elif self.method == 'zscore':
            mean = data.mean()
            std = data.std()
            self.lower_bound = mean - self.threshold * std
            self.upper_bound = mean + self.threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {self.method}")
        
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        if self.action == 'clip':
            return data.clip(lower=self.lower_bound, upper=self.upper_bound)
        elif self.action == 'mask':
            # Return mask for filtering (caller handles removal)
            return (data >= self.lower_bound) & (data <= self.upper_bound)
        else:
            raise ValueError(f"Unknown action: {self.action}")


class TransformProcessor(BaseProcessor):
    """Apply custom transformations like log, sqrt, etc."""
    
    def __init__(self, transform_func: Callable):
        self.transform_func = transform_func
    
    def fit(self, data: pd.Series, **kwargs) -> 'TransformProcessor':
        # Transformations are typically stateless
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        try:
            return self.transform_func(data)
        except Exception as e:
            warnings.warn(f"Transform function failed: {e}")
            return data


class ScalingProcessor(BaseProcessor):
    """Handle different scaling methods."""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'ScalingProcessor':
        data_reshaped = data.values.reshape(-1, 1)
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler.fit(data_reshaped)
        return self
    
    def transform(self, data: pd.Series) -> pd.Series:
        data_reshaped = data.values.reshape(-1, 1)
        return pd.Series(
            self.scaler.transform(data_reshaped).flatten(),
            index=data.index,
            name=data.name
        )


class EncodingProcessor(BaseProcessor):
    """Handle categorical encoding."""
    
    def __init__(self, method: str = 'label'):
        self.method = method
        self.encoder = None
        self.encoded_columns = None
    
    def fit(self, data: pd.Series, target: Optional[pd.Series] = None, **kwargs) -> 'EncodingProcessor':
        if self.method == 'label':
            self.encoder = LabelEncoder()
            # Handle potential mixed types by converting to string, filtering NaNs for fit
            self.encoder.fit(data.dropna().astype(str))
        
        elif self.method == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            data_reshaped = data.fillna('Unknown').astype(str).values.reshape(-1, 1)
            self.encoder.fit(data_reshaped)
            self.encoded_columns = self.encoder.get_feature_names_out([data.name]).tolist()
        
        elif self.method == 'mean':
            if target is None:
                raise ValueError("Target series required for mean encoding")
            
            aligned_data, aligned_target = data.align(target, join='inner')
            mean_target = aligned_target.mean()
            
            mapping = {}
            for cat in aligned_data.unique():
                if pd.isnull(cat):
                    continue
                mask = aligned_data == cat
                if mask.any():
                    mapping[cat] = aligned_target[mask].mean()
            
            # Use defaultdict for unseen categories
            self.encoder = defaultdict(lambda: mean_target, mapping)
        
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
        
        return self
    
    def transform(self, data: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        if self.method == 'label':
            # Handle unseen categories
            data_clean = data.fillna('Unknown').astype(str)
            if hasattr(self.encoder, 'classes_'):
                unseen_mask = ~data_clean.isin(self.encoder.classes_)
                if unseen_mask.any():
                    # Map unseen to the first class (often 0) or handle differently
                    most_frequent = self.encoder.classes_[0]
                    data_clean.loc[unseen_mask] = most_frequent
            
            return pd.Series(
                self.encoder.transform(data_clean),
                index=data.index,
                name=data.name
            )
        
        elif self.method == 'onehot':
            data_reshaped = data.fillna('Unknown').astype(str).values.reshape(-1, 1)
            encoded_array = self.encoder.transform(data_reshaped)
            
            # Return DataFrame with proper column names
            return pd.DataFrame(
                encoded_array,
                index=data.index,
                columns=self.encoded_columns
            )
        
        elif self.method == 'mean':
            return pd.Series(
                data.map(self.encoder),
                index=data.index,
                name=f"{data.name}_mean"
            )


class FeaturePipeline:
    """Pipeline for processing a single feature with multiple processors."""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.processors = []
        self.is_fitted = False
    
    def add_processor(self, processor: BaseProcessor) -> 'FeaturePipeline':
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        return self
    
    def fit(self, data: pd.Series, target: Optional[pd.Series] = None) -> 'FeaturePipeline':
        """Fit all processors in sequence."""
        current_data = data.copy()
        
        for processor in self.processors:
            if isinstance(processor, EncodingProcessor):
                processor.fit(current_data, target=target)
            else:
                processor.fit(current_data)
            
            # Transform for next processor (except for outlier masking)
            if not (isinstance(processor, OutlierProcessor) and processor.action == 'mask'):
                current_data = processor.transform(current_data)
                if isinstance(current_data, pd.DataFrame):
                    # For one-hot encoding, use first column for next processors if further processing is needed
                    # (Usually encoding is the last step, but this handles edge cases)
                    current_data = current_data.iloc[:, 0]
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """Transform data through all processors."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        current_data = data.copy()
        
        for processor in self.processors:
            result = processor.transform(current_data)
            
            if isinstance(processor, OutlierProcessor) and processor.action == 'mask':
                # Handle outlier masking differently (caller handles removal)
                continue
            
            current_data = result
            if isinstance(current_data, pd.DataFrame):
                # For one-hot encoding, keep DataFrame structure and stop
                break
        
        return current_data


class DataPreprocessor:
    """Main class orchestrating all preprocessing operations."""
    
    def __init__(self, target_column: str = 'price', verbose: bool = False):
        self.target_column = target_column
        self.feature_pipelines: Dict[str, FeaturePipeline] = {}
        self.is_fitted = False
        self.verbose = verbose
        
        # PCA Configuration
        self.pca_config: Optional[Dict] = None
        self.pca_transformer: Optional[PCA] = None
    
    def add_feature_pipeline(self, feature_name: str, 
                           missing_strategy: Optional[str] = None,
                           outlier_method: Optional[str] = None,
                           outlier_action: str = 'clip',
                           transform_func: Optional[Callable] = None,
                           scaling_method: Optional[str] = None,
                           encoding_method: Optional[str] = None) -> 'DataPreprocessor':
        """
        Add a complete preprocessing pipeline for a feature.
        """
        pipeline = FeaturePipeline(feature_name)
        
        # Add processors in logical order
        if missing_strategy:
            pipeline.add_processor(MissingValueProcessor(missing_strategy))
        
        if outlier_method:
            pipeline.add_processor(OutlierProcessor(outlier_method, outlier_action))
        
        if transform_func:
            pipeline.add_processor(TransformProcessor(transform_func))
        
        if scaling_method:
            pipeline.add_processor(ScalingProcessor(scaling_method))
        
        if encoding_method:
            pipeline.add_processor(EncodingProcessor(encoding_method))
        
        self.feature_pipelines[feature_name] = pipeline
        return self

    def configure_pca(self, **kwargs) -> 'DataPreprocessor':
        """
        Configure global dimensionality reduction using PCA.
        
        Args:
            **kwargs: Arguments passed to sklearn.decomposition.PCA
                      (e.g., n_components=0.95, random_state=42)
        """
        self.pca_config = kwargs
        self.pca_transformer = PCA(**kwargs)
        if self.verbose:
            print(f" PCA configured with: {kwargs}")
        return self
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit all feature pipelines and optionally PCA."""
        target = data[self.target_column] if self.target_column in data.columns else None
        
        # 1. Fit individual Feature Pipelines
        for feature_name, pipeline in self.feature_pipelines.items():
            if feature_name in data.columns:
                print(f" Fitting pipeline for '{feature_name}'") if self.verbose else None
                pipeline.fit(data[feature_name], target=target)
            else:
                warnings.warn(f"Feature '{feature_name}' not found in data")
        
        self.is_fitted = True
        
        # 2. Fit PCA if configured (requires transforming data first)
        if self.pca_transformer is not None:
            if self.verbose:
                print(" Fitting global PCA...")
            
            # Get clean data WITHOUT applying PCA yet
            X_clean, _ = self.transform(data, apply_pca=False)
            
            # Fit PCA on the clean, scaled, encoded data
            self.pca_transformer.fit(X_clean)
            
            if self.verbose:
                n_comps = self.pca_transformer.n_components_
                print(f"  - PCA fitted. Components: {n_comps}")
                
        return self
    
    def transform(self, data: pd.DataFrame, 
                 features: Optional[List[str]] = None, 
                 apply_pca: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform data using fitted pipelines.
        
        Args:
            data: Input DataFrame
            features: List of features to process (default: all configured)
            apply_pca: Whether to apply PCA if it was configured (default: True)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        features_to_process = features or list(self.feature_pipelines.keys())
        result_data = data.copy()
        
        # 1. Apply Feature Pipelines
        processed_features = []
        
        for feature_name in features_to_process:
            if feature_name in self.feature_pipelines and feature_name in data.columns:
                print(f" Transforming '{feature_name}'") if self.verbose else None
                
                pipeline = self.feature_pipelines[feature_name]
                transformed = pipeline.transform(data[feature_name])
                
                # Remove original column
                if feature_name in result_data.columns:
                    result_data = result_data.drop(columns=[feature_name])
                
                if isinstance(transformed, pd.DataFrame):
                    # Handle one-hot encoded features
                    result_data = pd.concat([result_data, transformed], axis=1)
                    processed_features.extend(transformed.columns.tolist())
                else:
                    result_data[feature_name] = transformed
                    processed_features.append(feature_name)
        
        # Extract X and y
        y = result_data[self.target_column] if self.target_column in result_data.columns else None
        X = result_data.drop(columns=[self.target_column]) if self.target_column in result_data.columns else result_data
        
        # Filter X to only contain processed columns (clean output)
        # This ensures we don't carry over raw columns that weren't in the pipeline
        if len(processed_features) > 0:
            # Only keep columns that resulted from our pipelines
            # (Intersection with current columns to be safe)
            valid_cols = [c for c in processed_features if c in X.columns]
            X = X[valid_cols]

        # 2. Apply PCA
        if self.pca_transformer is not None and apply_pca:
            if self.verbose:
                print(" Applying PCA transformation")
            
            X_pca_array = self.pca_transformer.transform(X)
            
            # Wrap back into DataFrame
            n_components = X_pca_array.shape[1]
            pca_cols = [f'PC{i+1}' for i in range(n_components)]
            
            X = pd.DataFrame(
                X_pca_array,
                index=X.index,
                columns=pca_cols
            )

        return X, y
    
    def fit_transform(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, features)
    
    def get_feature_info(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about configured features."""
        info = {}
        if feature_name:
            if feature_name in self.feature_pipelines:
                pipeline = self.feature_pipelines[feature_name]
                info = {
                    'feature_name': feature_name,
                    'processors': [type(p).__name__ for p in pipeline.processors],
                    'is_fitted': pipeline.is_fitted
                }
            else:
                info = {'error': f"No pipeline found for feature '{feature_name}'"}
        else:
            info = {
                'configured_features': list(self.feature_pipelines.keys()),
                'fitted_features': [name for name, pipeline in self.feature_pipelines.items() if pipeline.is_fitted]
            }
        
        # Add PCA info
        if self.pca_transformer:
            info['pca_config'] = self.pca_config
            if hasattr(self.pca_transformer, 'n_components_'):
                info['pca_components'] = self.pca_transformer.n_components_
                
        return info
    
    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'target_column': self.target_column,
                'feature_pipelines': self.feature_pipelines,
                'is_fitted': self.is_fitted,
                'pca_config': self.pca_config,         # Save config
                'pca_transformer': self.pca_transformer # Save fitted model
            }, f)
        print(f" Pipeline saved to '{filepath}'") if self.verbose else None
    
    def load_pipeline(self, filepath: str):
        """Load a preprocessing pipeline."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.target_column = data['target_column']
        self.feature_pipelines = data['feature_pipelines']
        self.is_fitted = data['is_fitted']
        
        # Load PCA if present (backward compatible)
        self.pca_config = data.get('pca_config', None)
        self.pca_transformer = data.get('pca_transformer', None)
        
        print(f" Pipeline loaded from '{filepath}'") if self.verbose else None