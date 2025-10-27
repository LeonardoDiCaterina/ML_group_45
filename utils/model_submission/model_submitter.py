import pickle
from datetime import datetime
import os
import pandas as pd # type: ignore

class ModelSubmitter:
    def __init__(self, model, test_data, feature_columns, filepath='Submission/'):
        self.model = model
        self.test_data = test_data
        self.feature_columns = feature_columns
        self.filepath = filepath # where to save the model and submission file
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
    
    def get_predictions(self):
        X_test = self.test_data[self.feature_columns]
        predictions = self.model.predict(X_test)
        return pd.DataFrame({'Index': self.test_data.index, 'Predicted': predictions})
    
    def save_submission(self, filename='submission.csv'):
        predictions_df = self.get_predictions()
        # get today's date
        date_str = datetime.now().strftime('%Y%m%d')
        full_filename = f"{self.filepath}{date_str}_{filename}"
        count = 1
        while os.path.exists(full_filename):
            full_filename = f"{self.filepath}{date_str}_{count}_{filename}"
            count += 1
        predictions_df.to_csv(full_filename, index=False)
        print(f"Submission saved to {full_filename}")
        # now save the model
        model_filename = full_filename.replace('.csv', '_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_filename}")