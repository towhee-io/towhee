class ControlPlane:
    def __init__(self) -> None:
        self._dag = {}
    
    @property
    def dag(self):
        return self._dag
