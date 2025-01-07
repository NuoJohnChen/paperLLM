from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, TypeVar

T = TypeVar('T')

class ConcurrentTaskManager:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers
        self.executor = None
        self.futures = []
        
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()
        
    def submit_task(self, func: Callable[..., T], *args, **kwargs) -> 'ConcurrentTaskManager':
        """Submit a task to be executed concurrently"""
        future = self.executor.submit(func, *args, **kwargs)
        self.futures.append(future)
        return self

    def get_results(self) -> List[Any]:
        """Get all results from completed tasks"""
        results = []
        for future in as_completed(self.futures):
            results.append(future.result())
        return results

    def wait(self):
        """Wait for all tasks to complete"""
        for future in as_completed(self.futures):
            pass