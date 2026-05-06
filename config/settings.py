"""
Global Configuration - Centralized parameters for the entire pipeline.
"""

# Data Configuration
DATA_CONFIG = {
    'universe': 'NIFTY_20',
    'start_date': '2014-01-01',
    'end_date': None,  # Default to today
    'data_dir': 'data/raw',
    'processed_dir': 'data/processed',
}

# Feature Engineering
FEATURES_CONFIG = {
    'technical_indicators': {
        'momentum': ['RSI', 'MACD', 'STOCH'],
        'trend': ['SMA', 'EMA', 'TEMA'],
        'volatility': ['ATR', 'BBANDS', 'KELTNER'],
        'volume': ['OBV', 'VWAP', 'VOSC'],
    },
    'price_based': {
        'returns': [1, 5, 20],  # Lagged periods
        'volatility_windows': [20, 60],
        'rolling_skewness': 20,
        'rolling_kurtosis': 20,
    },
    'normalize_method': 'zscore',  # 'zscore', 'minmax', 'rank'
    'remove_correlated_above': 0.95,
}

# Label Generation (Triple-Barrier)
LABELS_CONFIG = {
    'profit_target': 0.01,      # 1% profit target
    'loss_barrier': 0.01,        # 1% stop loss
    'max_holding_days': 5,       # Max 5 trading days
    'min_price_change': 0.0001,  # Minimum move to register
}

# ML Model Configuration
ML_CONFIG = {
    'test_size': 0.2,
    'validation_split': 0.15,
    'random_state': 42,

    'lgbm': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary',
    },

    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
    },

    'ensemble_weights': {
        'lgbm': 0.7,
        'rf': 0.3,
    }
}

# Portfolio Optimization
PORTFOLIO_CONFIG = {
    'method': 'HRP',  # Hierarchical Risk Parity
    'hrp': {
        'covariance_window': 60,  # 60-day rolling covariance
        'linkage_method': 'single',
        'clustering_method': 'linkage',  # scipy.cluster.hierarchy
    },
    'position_sizing': {
        'method': 'kelly',  # 'kelly', 'equal_weight', 'inverse_volatility'
        'kelly_fraction': 0.25,  # Cap kelly at 1/4 to avoid leverage
        'min_position': 0.01,    # 1% minimum position
        'max_position': 0.05,    # 5% maximum position
    },
    'rebalancing': {
        'frequency': 'monthly',  # 'monthly', 'weekly', 'daily', 'quarterly'
        'rebalance_day_of_week': 0,  # 0=Monday (for weekly)
        'rebalance_day_of_month': 1,  # 1st of month (for monthly)
    }
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 1_000_000,  # $1M starting capital
    'cash_buffer': 0.05,            # 5% cash buffer
    'slippage': {
        'type': 'percentage',
        'value': 0.0001,  # 1 basis point
    },
    'commission': {
        'type': 'percentage',
        'value': 0.0005,  # 5 basis points
    },
    'fill_model': 'next_bar_close',  # Realistic for large caps
}

# Validation Configuration
VALIDATION_CONFIG = {
    'walk_forward': {
        'num_folds': 12,
        'refit_frequency': 'quarterly',  # Refit every quarter
    },
    'cpcv': {
        'num_splits': 10,
        'embargo_pct': 0.02,  # 2% embargo period
        'purge_lookback': 5,  # 5-day purge
    },
    'robustness_thresholds': {
        'walk_forward_degradation': 0.30,  # < 30% degradation
        'pbo_p_value': 0.5,                # PBO > 50% (not overfitted)
        'dsr_threshold': 0.10,             # DSR > 10%
        'min_sharpe': 0.3,
    }
}

# Data Splits (for Phase 1 MVP)
DATA_SPLITS = {
    'train': ('2014-01-01', '2022-12-31'),
    'validation': ('2023-01-01', '2023-06-30'),
    'test': ('2023-07-01', '2024-12-31'),
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs',
}

# Feature Selection (for Phase 1, all are enabled)
FEATURE_FLAGS = {
    'use_technical_indicators': True,
    'use_price_features': True,
    'use_volume_features': True,
    'use_fundamental_factors': False,  # Phase 2+
    'use_sentiment_data': False,        # Phase 3+
    'use_gnn_features': False,          # Phase 2+ (research)
}
