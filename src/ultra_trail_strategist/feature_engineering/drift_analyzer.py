import numpy as np
from typing import List, Dict, Any

class DriftAnalyzer:
    """
    Analyzes aerobic decoupling (cardiac drift) from activity streams.
    Pa:HR = Pace (or Power) to Heart Rate Ratio.
    """
    
    def calculate_decoupling(self, streams: List[Dict[str, Any]]) -> float:
        """
        Calculates the aerobic decoupling percentage for a single activity.
        Logic:
        1. Calculate Pa:HR for first half of the activity.
        2. Calculate Pa:HR for second half.
        3. Decoupling = ((Pa:HR_h1 - Pa:HR_h2) / Pa:HR_h1) * 100
        
        Note: We want High Efficiency (Low Heart Rate per Speed) in H1.
        If HR rises while Pace stays same -> Efficiency drops.
        Ratio = Speed / HR. (Running Economy Proxy)
        Lower Ratio in H2 means Decoupling.
        
        Args:
            streams: List of data points with 'velocity_smooth' and 'heartrate'.
            
        Returns:
            Decoupling % (e.g., 5.0 for 5% drift). Positive means drift (bad).
            Returns 0.0 if insufficient data.
        """
        if not streams or len(streams) < 100:
            return 0.0
            
        # Extract numpy arrays
        velocities = []
        heartrates = []
        
        for p in streams:
            v = p.get('velocity_smooth')
            hr = p.get('heartrate')
            if v is not None and hr is not None and hr > 0 and v > 0:
                velocities.append(v)
                heartrates.append(hr)
                
        if len(velocities) < 100:
            return 0.0
            
        velocities = np.array(velocities)
        heartrates = np.array(heartrates)
        
        # Calculate Ratio (Efficiency) = Speed / Heart Rate
        ratios = velocities / heartrates
        
        # Split into First Half and Second Half
        midpoint = len(ratios) // 2
        
        h1_ratio = np.mean(ratios[:midpoint])
        h2_ratio = np.mean(ratios[midpoint:])
        
        if h1_ratio == 0:
            return 0.0
            
        # Decoupling: Percentage drop in efficiency
        drift = ((h1_ratio - h2_ratio) / h1_ratio) * 100
        
        return drift

    def calculate_endurance_factor(self, all_activities_streams: List[List[Dict[str, Any]]]) -> float:
        """
        Aggregates daily drifts to produce a global 'Endurance Factor'.
        Factor 1.0 = Rock solid aerobic base (0% drift).
        Factor < 1.0 = Expect degradation.
        
        Returns: An adjustment factor for late-race pacing (e.g. 0.95).
        """
        drifts = []
        for activity in all_activities_streams:
            d = self.calculate_decoupling(activity)
            # Filter valid drifts (e.g. between -10% and +30%)
            if -10 < d < 40:
                drifts.append(d)
                
        if not drifts:
            return 1.0 # Assume perfect endurance if no HR data
            
        avg_drift = np.mean(drifts)
        
        # If avg drift is 5%, endurance factor is roughly 0.95 (simplified)
        # But drift happens over TIME. Usually decoupling is measured over the duration.
        # We will map Drift -> Penalty Factor.
        # 5% drift is normal/good. >10% implies lack of aerobic base.
        
        if avg_drift < 5.0:
            return 1.0
        
        penalty_pct = (avg_drift - 5.0) # Excess drift
        factor = 1.0 - (penalty_pct / 100.0) 
        
        return max(0.8, min(1.0, factor)) # Cap penalty at 20%
