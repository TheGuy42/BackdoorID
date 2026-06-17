import time
from typing import Optional
import torch
from src.utils.better_logging import get_logger

logger = get_logger(__name__)


class EventTimer:
    """Timer class for measuring event durations with optional CUDA synchronization."""
    
    def __init__(self, name: str, sync_cuda: bool = True, pre_sync_cuda: bool = False, raise_exceptions: bool = True):
        """
        Initialize an EventTimer.
        
        Args:
            name: Name of the event being timed
            sync_cuda: Whether to synchronize CUDA before stopping the timer
            pre_sync_cuda: Whether to synchronize CUDA before starting the timer
            raise_exceptions: Whether to raise exceptions or just log warnings
        """
        self.name = name
        self.sync_cuda_before = pre_sync_cuda
        self.sync_cuda_after = sync_cuda
        self.raise_exceptions = raise_exceptions
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_time: float = 0.0
        self.count: int = 0
    
    def start(self):
        """Start the timer."""
        if self.sync_cuda_before and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop the timer and accumulate the elapsed time.
        
        Returns:
            The elapsed time for this timing cycle
        """
        if self.start_time is None:
            error_msg = f"Timer '{self.name}' was stopped without being started"
            if self.raise_exceptions:
                raise RuntimeError(error_msg)
            else:
                logger.warning(error_msg)
                return 0.0
        
        if self.sync_cuda_after and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        
        elapsed = self.end_time - self.start_time
        self.total_time += elapsed
        self.count += 1
        
        return elapsed
    
    def reset(self):
        """Reset the timer statistics."""
        self.start_time = None
        self.end_time = None
        self.total_time = 0.0
        self.count = 0
    
    def get_elapsed(self) -> float:
        """Get the most recent elapsed time."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0
    
    def get_total(self) -> float:
        """Get the total accumulated time."""
        return self.total_time
    
    def get_average(self) -> float:
        """Get the average time per timing event."""
        if self.count > 0:
            return self.total_time / self.count
        return 0.0


class TimingManager:
    """Manager class for holding and managing named EventTimer objects."""
    
    _instance: Optional['TimingManager'] = None
    
    def __init__(self):
        """Initialize the TimerManager."""
        self.timers: dict[str, EventTimer] = {}
    
    @classmethod
    def get_instance(cls) -> 'TimingManager':
        """
        Get the singleton instance of TimerManager.
        
        Returns:
            The active TimerManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_timer(self, name: str, sync_cuda: bool = True, pre_sync_cuda: bool = False, raise_exceptions: bool = True) -> EventTimer:
        """
        Get or create a timer with the given name.
        
        Args:
            name: Name of the timer
            sync_cuda: Whether to synchronize CUDA before stopping (only used when creating new timer)
            pre_sync_cuda: Whether to synchronize CUDA before starting (only used when creating new timer)
            raise_exceptions: Whether to raise exceptions or just log warnings (only used when creating new timer)
            
        Returns:
            The EventTimer instance
        """
        if name not in self.timers:
            self.timers[name] = EventTimer(name, sync_cuda, pre_sync_cuda, raise_exceptions)
        return self.timers[name]
    
    def start(self, name: str, sync_cuda: bool = True, pre_sync_cuda: bool = False, raise_exceptions: bool = True):
        """
        Start a timer. If the timer doesn't exist, create it first.
        
        Args:
            name: Name of the timer
            sync_cuda: Whether to synchronize CUDA before stopping (only used when creating new timer)
            pre_sync_cuda: Whether to synchronize CUDA before starting (only used when creating new timer)
            raise_exceptions: Whether to raise exceptions or just log warnings (only used when creating new timer)
        """
        timer = self.get_timer(name, sync_cuda, pre_sync_cuda, raise_exceptions)
        timer.start()
    
    def stop(self, name: str) -> float:
        """
        Stop a timer and return the elapsed time.
        
        Args:
            name: Name of the timer
            
        Returns:
            The elapsed time for this timing cycle
            
        Raises:
            KeyError: If the timer doesn't exist
        """
        if name not in self.timers:
            error_msg = f"Timer '{name}' does not exist"
            logger.error(error_msg)
            raise KeyError(error_msg)
        return self.timers[name].stop()
    
    def reset_all(self):
        """Reset all timers."""
        for timer in self.timers.values():
            timer.reset()
    
    def get_summary(self) -> dict[str, dict[str, float]]:
        """
        Get a summary of all timer statistics.
        
        Returns:
            Dictionary mapping timer names to their statistics
        """
        summary = {}
        for name, timer in self.timers.items():
            summary[name] = {
                'total': timer.get_total(),
                'count': timer.count,
                'average': timer.get_average()
            }
        return summary
    
    def print_summary(self):
        """Print a formatted summary of all timer statistics."""
        summary = self.get_summary()
        if not summary:
            print("No timers recorded.")
            return
        
        print("\n" + "="*70)
        print(f"{'Timer Name':<30} {'Count':>10} {'Total (s)':>12} {'Avg (s)':>12}")
        print("="*70)
        for name, stats in summary.items():
            print(f"{name:<30} {stats['count']:>10} {stats['total']:>12.6f} {stats['average']:>12.6f}")
        print("="*70 + "\n")
