from typing import Optional

class HealthClient:
    """
    Client for fetching athlete health and readiness metrics.
    Currently a mock implementation until API integrations (Whoop/Oura) are built.
    """
    
    def __init__(self):
        pass
        
    def get_readiness_score(self, manual_override: Optional[int] = None) -> int:
        """
        Returns the athlete's daily readiness score (0-100).
        
        Args:
            manual_override: If provided (e.g. from UI), use this instead of API.
            
        Returns:
            int: 0-100 score. 100 = Fully recovered.
        """
        if manual_override is not None:
            return max(0, min(100, manual_override))
            
        # Default mock value if no API and no override
        return 85
