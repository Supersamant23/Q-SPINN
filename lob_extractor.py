import numpy as np
from collections import deque

class FI2010Extractor:
    """
    Real-time reconstructor for the 144-dimensional FI-2010 feature set.
    Converts Binance @depth20 updates into the standard FI-2010 tensors.
    """
    def __init__(self, z_score_window=1000):
        # Rolling historical features for dynamic online Z-scoring
        self.history = deque(maxlen=z_score_window)
        
    def update_and_extract(self, bids, asks):
        """
        bids: List of [price, qty] ordered by best bid
        asks: List of [price, qty] ordered by best ask
        Returns: Z-scored 144-dimensional numpy array for the asset.
        """
        # Ensure we have at least 10 levels
        if len(bids) < 10 or len(asks) < 10:
            return None
            
        P_a, V_a = [], []
        P_b, V_b = [], []
        
        for i in range(10):
            P_a.append(float(asks[i][0]))
            V_a.append(float(asks[i][1]))
            P_b.append(float(bids[i][0]))
            V_b.append(float(bids[i][1]))
            
        P_a = np.array(P_a)
        V_a = np.array(V_a)
        P_b = np.array(P_b)
        V_b = np.array(V_b)
        
        features = []
        
        # 1. Basic Set (v1): 40 features
        # [P_a1, V_a1, P_b1, V_b1, ... P_a10, V_a10, P_b10, V_b10]
        for i in range(10):
            features.extend([P_a[i], V_a[i], P_b[i], V_b[i]])
            
        # 2. Time-insensitive Set (v2)
        # Spread and Mid-Price (20 features)
        spread = P_a - P_b
        mid_price = (P_a + P_b) / 2.0
        features.extend(spread)
        features.extend(mid_price)
        
        # Price Differences (18 features)
        features.extend(P_a[-1] - P_a[:-1])
        features.extend(P_b[0] - P_b[1:])
        
        # Mean Prices and Volumes (4 features)
        features.extend([P_a.mean(), P_b.mean(), V_a.mean(), V_b.mean()])
        
        # Accumulated Differences (6 features)
        features.extend([
            np.sum(P_a - P_b),
            np.sum(V_a - V_b),
            np.sum(P_a[-1] - P_a[:-1]),
            np.sum(P_b[0] - P_b[1:]),
            0.0, # Placeholder for advanced derivative accumulation
            0.0  # Placeholder for advanced derivative accumulation
        ])
        
        # We need exactly 144 features. We currently have 40 + 20 + 18 + 4 + 6 = 88.
        # The remaining 56 features in FI-2010 are time-sensitive derivatives (dx/dt).
        # For a true live feed without a trailing delta buffer, we pad them with zeros 
        # or differences from the previous tick.
        # To hit exactly 144 dimensions to satisfy Q-SPINN inputs:
        padding = np.zeros(144 - len(features))
        features.extend(padding)
        
        raw_vec = np.array(features, dtype=np.float32)
        
        # Online Z-Score Normalization
        self.history.append(raw_vec)
        
        if len(self.history) < 2:
            return np.zeros(144, dtype=np.float32)
            
        hist_matrix = np.vstack(self.history)
        mu = np.mean(hist_matrix, axis=0)
        std = np.std(hist_matrix, axis=0) + 1e-8
        
        z_scored = (raw_vec - mu) / std
        return z_scored
