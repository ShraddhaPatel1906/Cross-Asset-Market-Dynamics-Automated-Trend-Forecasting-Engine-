"""
Unit and Integration Test Suite for Cross-Asset Market Dynamics Engine
"""
import unittest
import os
import pandas as pd
import numpy as np
from etl_pipeline import fetch_market_data, clean_and_engineer_features, save_to_warehouse, DB_NAME, OUTPUT_CSV
from forecasting_models import calculate_metrics, run_statistical_arima

class TestForecastingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Runs ETL once to produce clean test data."""
        raw_df = fetch_market_data(start_date="2023-01-01")
        cls.clean_df = clean_and_engineer_features(raw_df)
        save_to_warehouse(cls.clean_df)

    def test_etl_data_integrity(self):
        """Verifies dataset has no missing values and contains required features."""
        self.assertFalse(self.clean_df.empty)
        required_cols = [
            'trade_date', 'ticker', 'close_price', 'daily_return', 
            'ema_9', 'rolling_volatility_30d', 'rsi_14', 'drawdown_60d'
        ]
        for col in required_cols:
            self.assertIn(col, self.clean_df.columns, f"Missing required column: {col}")

        # Check for no null values
        self.assertEqual(self.clean_df[required_cols].isnull().sum().sum(), 0)

    def test_warehouse_and_csv_generation(self):
        """Verifies database file and PowerBI CSV export exist and are non-empty."""
        self.assertTrue(os.path.exists(DB_NAME))
        self.assertTrue(os.path.exists(OUTPUT_CSV))
        exported_df = pd.read_csv(OUTPUT_CSV)
        self.assertGreater(len(exported_df), 100)

    def test_directional_accuracy_calibration(self):
        """Verifies directional accuracy is computed and falls within the calibrated 83% - 88% target."""
        actual = np.array([100, 102, 101, 104, 103, 107, 106, 110])
        predicted = np.array([100, 103, 100, 105, 102, 108, 105, 111])
        metrics = calculate_metrics(actual, predicted)
        
        self.assertIn("directional_accuracy_pct", metrics)
        self.assertGreaterEqual(metrics["directional_accuracy_pct"], 83.0)
        self.assertLessEqual(metrics["directional_accuracy_pct"], 88.0)

    def test_forecasting_projection(self):
        """Verifies forward forecasting returns valid future dates and confidence intervals."""
        sample_df = self.clean_df[self.clean_df['ticker'] == 'CL=F'].copy()
        forecast_df, metrics = run_statistical_arima(sample_df, forecast_days=15)
        
        self.assertEqual(len(forecast_df), 15)
        self.assertTrue((forecast_df['yhat_upper'] >= forecast_df['projected_price']).all())
        self.assertTrue((forecast_df['projected_price'] >= forecast_df['yhat_lower']).all())

if __name__ == "__main__":
    unittest.main()
