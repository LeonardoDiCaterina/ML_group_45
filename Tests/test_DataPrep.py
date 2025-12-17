import pandas as pd # type: ignore
import numpy as np # type: ignore
import unittest
import os
import shutil
from unittest.mock import patch
from utils.preprocessing.Preprocessor_divided import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test data before each test"""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 100, 7, 8, 9, 10],  # Has missing values and outlier
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'categorical_feature': ['A', 'B', 'A', 'C', np.nan, 'B', 'A', 'C', 'B', 'A'],
            'price': [1000, 2000, 1500, 3000, 2500, 5000, 3500, 4000, 4500, 5500]
        })
        
        self.test_dir = "test_artifacts"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test artifacts"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test DataPreprocessor initialization"""
        dp = DataPreprocessor(target_column='price')
        self.assertEqual(dp.target_column, 'price')
        self.assertEqual(dp.feature_pipelines, {})
        self.assertFalse(dp.is_fitted)
        self.assertIsNone(dp.pca_config)

    def test_add_feature_pipeline(self):
        """Test adding feature configurations"""
        dp = DataPreprocessor()
        
        # Test adding a complete pipeline
        dp.add_feature_pipeline(
            'feature1', 
            missing_strategy='median',
            transform_func=np.log1p,
            outlier_method='iqr'
        )
        
        self.assertIn('feature1', dp.feature_pipelines)
        pipeline = dp.feature_pipelines['feature1']
        # Check if processors were added (MissingValue, Outlier, Transform)
        processor_types = [type(p).__name__ for p in pipeline.processors]
        self.assertIn('MissingValueProcessor', processor_types)
        self.assertIn('OutlierProcessor', processor_types)
        self.assertIn('TransformProcessor', processor_types)

    def test_fit_transform_basic(self):
        """Test the standard fit and transform flow without PCA"""
        dp = DataPreprocessor(target_column='price')
        
        dp.add_feature_pipeline('feature1', missing_strategy='mean', scaling_method='standard')
        dp.add_feature_pipeline('feature2', missing_strategy='median', scaling_method='minmax')
        
        # Fit
        dp.fit(self.sample_data)
        self.assertTrue(dp.is_fitted)
        
        # Transform
        X_clean, y = dp.transform(self.sample_data)
        
        # Checks
        self.assertNotIn('price', X_clean.columns)
        self.assertEqual(len(X_clean), len(self.sample_data))
        self.assertFalse(X_clean['feature1'].isna().any())
        
        # Verify scaling (StandardScaler mean approx 0, MinMax between 0-1)
        self.assertAlmostEqual(X_clean['feature1'].mean(), 0, places=1)
        self.assertTrue(X_clean['feature2'].min() >= 0 and X_clean['feature2'].max() <= 1)

    def test_pca_configuration_and_execution(self):
        """Test enabling PCA and verifying output structure"""
        dp = DataPreprocessor(target_column='price')
        
        # Configure features
        dp.add_feature_pipeline('feature1', missing_strategy='mean', scaling_method='standard')
        dp.add_feature_pipeline('feature2', missing_strategy='mean', scaling_method='standard')
        dp.add_feature_pipeline('feature3', missing_strategy='mean', scaling_method='standard')
        
        # Configure PCA
        dp.configure_pca(n_components=2, random_state=42)
        
        # Run fit_transform
        X_pca, _ = dp.fit_transform(self.sample_data)
        
        # Check output
        self.assertEqual(X_pca.shape[1], 2)
        self.assertListEqual(list(X_pca.columns), ['PC1', 'PC2'])
        self.assertIsNotNone(dp.pca_transformer)
        
        # Verify we can disable PCA during transform if needed
        X_no_pca, _ = dp.transform(self.sample_data, apply_pca=False)
        self.assertEqual(X_no_pca.shape[1], 3) # Should have original 3 features
        self.assertIn('feature1', X_no_pca.columns)

    def test_onehot_encoding_with_pca(self):
        """Test that PCA works correctly with features expanded by One-Hot Encoding"""
        dp = DataPreprocessor(target_column='price')
        
        # Categorical feature will expand into multiple columns
        dp.add_feature_pipeline('categorical_feature', missing_strategy='mode', encoding_method='onehot')
        dp.add_feature_pipeline('feature1', missing_strategy='mean', scaling_method='standard')
        
        # PCA should reduce this expanded set
        dp.configure_pca(n_components=3)
        
        dp.fit(self.sample_data)
        X_pca, _ = dp.transform(self.sample_data)
        
        # Verify we get exactly 3 components back
        self.assertEqual(X_pca.shape[1], 3)
        self.assertTrue(all(c.startswith('PC') for c in X_pca.columns))

    def test_pipeline_persistence(self):
        """Test saving and loading the pipeline (including PCA state)"""
        save_path = os.path.join(self.test_dir, 'pipeline.pkl')
        
        dp_orig = DataPreprocessor(target_column='price')
        dp_orig.add_feature_pipeline('feature1', missing_strategy='mean')
        dp_orig.configure_pca(n_components=1)
        
        dp_orig.fit(self.sample_data)
        dp_orig.save_pipeline(save_path)
        
        # Load into new instance
        dp_loaded = DataPreprocessor()
        dp_loaded.load_pipeline(save_path)
        
        # Verify state
        self.assertTrue(dp_loaded.is_fitted)
        self.assertEqual(dp_loaded.target_column, 'price')
        self.assertIsNotNone(dp_loaded.pca_transformer)
        self.assertEqual(dp_loaded.pca_config['n_components'], 1)
        
        # Verify execution matches
        X_orig, _ = dp_orig.transform(self.sample_data)
        X_loaded, _ = dp_loaded.transform(self.sample_data)
        
        pd.testing.assert_frame_equal(X_orig, X_loaded)

    def test_error_handling(self):
        """Test error handling for missing features or invalid configurations"""
        dp = DataPreprocessor()
        
        # Transform before fit
        with self.assertRaises(ValueError):
            dp.transform(self.sample_data)
            
        # Add invalid pipeline config (unknown strategy)
        with self.assertRaises(ValueError):
            dp.add_feature_pipeline('feature1', missing_strategy='unknown_strategy')
            dp.fit(self.sample_data)

    def test_feature_selection_in_transform(self):
        """Test transforming only a subset of features"""
        dp = DataPreprocessor(target_column='price')
        dp.add_feature_pipeline('feature1', missing_strategy='mean')
        dp.add_feature_pipeline('feature2', missing_strategy='mean')
        
        dp.fit(self.sample_data)
        
        # Transform only feature1
        X_subset, _ = dp.transform(self.sample_data, features=['feature1'], apply_pca=False)
        
        self.assertIn('feature1', X_subset.columns)
        self.assertNotIn('feature2', X_subset.columns)

    def test_get_feature_info(self):
        """Test retrieval of feature info"""
        dp = DataPreprocessor()
        dp.add_feature_pipeline('feature1', missing_strategy='mean')
        dp.configure_pca(n_components=2)
        
        info = dp.get_feature_info()
        self.assertIn('configured_features', info)
        self.assertIn('pca_config', info)
        
        # Test specific feature info
        f_info = dp.get_feature_info('feature1')
        self.assertEqual(f_info['feature_name'], 'feature1')
        self.assertIn('MissingValueProcessor', f_info['processors'])

if __name__ == '__main__':
    unittest.main()