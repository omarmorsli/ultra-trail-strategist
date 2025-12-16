import math

class FatigueModel:
    """
    Models the athlete's W' (W Prime) / Anaerobic Work Capacity.
    Adapted for running pace instead of power (CP equivalent).
    """

    def __init__(self, critical_pace_min_km: float, w_prime_balance: float = 15000):
        """
        Args:
            critical_pace_min_km: The threshold pace (minutes per km). 
                                  Running faster than this depletes reserves.
                                  Running slower recharges.
            w_prime_balance: Total anaerobic capacity in arbitrary units (Joules approx).
                             For runners, we might use "meters above CP", but let's stick to 
                             an energy unit equivalent or a generic capacity score.
                             Default 15000 is a placeholder for "Deep Energy".
                             
        Note: Modeling W' directly from pace is an approximation. 
        Ideally we'd use Power (Watts). Since we have Grade-Adjusted Pace, 
        we treat it as a proxy for intensity.
        """
        self.cp_min_km = critical_pace_min_km
        self.capacity = w_prime_balance
        self.current_balance = w_prime_balance
        self.segment_penalties = 0.0

    def update_balance(self, segment_pace_min_km: float, segment_length_km: float) -> float:
        """
        Updates the W' balance based on the segment effort.
        
        Args:
            segment_pace_min_km: The planned pace for the segment.
            segment_length_km: Length of segment.
            
        Returns:
            The new balance.
        """
        # Convert paces to speed (km/min) for easier physics math or keep in pace?
        # Let's use simple logic: 
        # Intensity = (1/pace)
        # CP_Speed = (1/cp_pace)
        # Delta = Speed - CP_Speed
        
        if segment_pace_min_km <= 0:
            return self.current_balance

        current_speed = 1000 / segment_pace_min_km # m/min
        cp_speed = 1000 / self.cp_min_km # m/min
        
        duration_min = segment_pace_min_km * segment_length_km
        
        # Depletion: Running faster than CP
        if current_speed > cp_speed:
            # How much "work" above CP?
            # Work = Power * Time. Proxy: Excess Speed * Time
            excess_speed = current_speed - cp_speed # m/min
            expenditure = excess_speed * duration_min # meters ??? 
            # This unit is "meters run above threshold speed". 
            # It's a valid Distance-based W' model (D').
            
            self.current_balance -= expenditure
            
        # Recovery: Running slower than CP
        else:
            # Recovery is usually exponential or linear up to CP.
            # Skiba et al. model for recovery:
            # W_bal(t) = W_prime - (W_prime - W_bal(t-1)) * e^(-D_u / Tau_w)
            # where D_u is time spent below CP?? No. 
            # Simple linear recovery for V3 MVP:
            recovery_speed = cp_speed - current_speed
            recovery = recovery_speed * duration_min
            # Apply a recovery efficiency factor (you don't recover 1:1 usually)
            recovery_factor = 0.5 
            self.current_balance += (recovery * recovery_factor)
            
        # Clamp
        if self.current_balance > self.capacity:
            self.current_balance = self.capacity
        
        return self.current_balance

    def get_exhaustion_level(self) -> float:
        """Returns 0.0 (fresh) to 1.0 (empty)."""
        if self.capacity == 0: return 1.0
        level = 1.0 - (self.current_balance / self.capacity)
        return max(0.0, min(1.0, level))

    def get_penalty_factor(self) -> float:
        """
        Returns a multiplier for pace (1.0 = no penalty, 1.2 = 20% slower).
        Penalty kicks in when W' gets low.
        """
        exhaustion = self.get_exhaustion_level()
        
        # If > 80% exhausted, start slowing down exponentially
        if exhaustion < 0.8:
            return 1.0
        
        # 0.8 -> 1.0, 0.9 -> 1.1, 1.0 -> 1.5
        # Maps 0.8-1.0 range to penalty
        severity = (exhaustion - 0.8) * 5 # 0 to 1
        return 1.0 + (severity * 0.5) # Max 50% slowdown
