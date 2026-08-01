"""
AgentCost Event Batcher

Hybrid batching system: sends events when batch is full OR every N seconds.
Thread-safe and handles graceful shutdown.
"""

import threading
import time
import atexit
import warnings
from typing import List, Dict, Callable, Optional


class HybridBatcher:
    """
    Efficient event batcher with both size and time triggers.
    
    Features:
    - Thread-safe event queue
    - Size-based flushing (when batch reaches N events)
    - Time-based flushing (every N seconds)
    - Graceful shutdown (flushes remaining events on exit)
    - Retry queue for failed batches
    """

    # Budget for the delivery attempts shutdown() makes, checked BETWEEN
    # attempts. It is not a hard wall-clock bound: flush_callback is a blocking
    # POST that cannot be interrupted once started, so the true worst case is
    # this budget plus one in-flight attempt, plus the 2s flush-thread join.
    # Measured ~8-10s against a refusing backend. Without any budget it was
    # max_retry_batches (100) x the HTTP client's full retry schedule, which is
    # minutes of apparent hang at interpreter exit.
    SHUTDOWN_GRACE_SECONDS = 5.0


    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        flush_callback: Optional[Callable[[List[Dict]], bool]] = None,
        max_retry_batches: int = 100,
        debug: bool = False
    ):
        """
        Args:
            batch_size: Max events before auto-flush
            flush_interval: Seconds between time-based flushes
            flush_callback: Function to call with batch data. Should return True on success.
            max_retry_batches: Max failed batches to keep for retry
            debug: Enable debug logging
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.flush_callback = flush_callback or self._default_flush
        self.max_retry_batches = max_retry_batches
        self.debug = debug
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self._batch: List[Dict] = []
        self._failed_batches: List[List[Dict]] = []
        
        # Threading control
        self._running = True
        self._flush_thread: Optional[threading.Thread] = None
        # Handles for the per-batch send threads started by _flush_locked.
        # Without these, shutdown() has no way to wait for a POST that is
        # already in flight: those events are in neither _batch nor
        # _failed_batches, so they are unrecoverable once the interpreter
        # kills the daemon thread carrying them.
        self._inflight: List[threading.Thread] = []
        
        # Statistics
        self._stats = {
            'events_added': 0,
            'events_sent': 0,
            'batches_sent': 0,
            'batches_failed': 0,
        }
        
        # Start background flush thread
        self._start_flush_thread()
        
        # atexit does not deduplicate automatically, so we track registration
        if not getattr(HybridBatcher, '_atexit_registered', False):
            atexit.register(self.shutdown)
            HybridBatcher._atexit_registered = True
    
    def add(self, event: Dict) -> None:
        """
        Add event to batch. Thread-safe.
        
        Args:
            event: Event dictionary to add
        """
        with self._lock:
            self._batch.append(event)
            self._stats['events_added'] += 1
            
            # Size-based trigger
            if len(self._batch) >= self.batch_size:
                self._flush_locked()
    
    def flush(self) -> None:
        """Manually flush current batch. Thread-safe."""
        with self._lock:
            self._flush_locked()
    
    def _flush_locked(self) -> None:
        """
        Flush batch (must hold lock).
        Sends current batch to callback and clears it.
        """
        if not self._batch:
            return
        
        # Copy batch
        events_to_send = self._batch.copy()
        self._batch = []
        
        # Send in separate thread to avoid blocking
        sender = threading.Thread(
            target=self._send_batch,
            args=(events_to_send,),
            daemon=True
        )
        # Retain the handle so shutdown() can wait for it. Drop finished
        # threads first so a long-lived process does not accumulate them.
        self._inflight = [t for t in self._inflight if t.is_alive()]
        self._inflight.append(sender)
        sender.start()
    
    def _send_batch(self, events: List[Dict]) -> None:
        """
        Send batch to backend.
        Handles failures and adds to retry queue.
        """
        try:
            success = self.flush_callback(events)
            
            with self._lock:
                if success:
                    self._stats['events_sent'] += len(events)
                    self._stats['batches_sent'] += 1
                    
                    if self.debug:
                        print(f"[AgentCost] Sent {len(events)} events")
                else:
                    self._handle_failed_batch(events)
                    
        except Exception as e:
            if self.debug:
                print(f"[AgentCost] Error: Batch send error: {e}")
            
            with self._lock:
                self._handle_failed_batch(events)
    
    def _handle_failed_batch(self, events: List[Dict]) -> None:
        """Handle a failed batch (must hold lock)"""
        self._stats['batches_failed'] += 1
        
        # Add to retry queue (limited size)
        if len(self._failed_batches) < self.max_retry_batches:
            self._failed_batches.append(events)
            
            if self.debug:
                print(f"[AgentCost] Batch queued for retry ({len(self._failed_batches)} pending)")
        else:
            if self.debug:
                print(f"[AgentCost] Warning: Retry queue full, dropping {len(events)} events")
    
    def retry_failed_batches(self, deadline: Optional[float] = None) -> int:
        """
        Retry all failed batches.

        Args:
            deadline: Optional time.monotonic() value past which no further
                batch is attempted. Untried batches are returned to the queue
                so the caller can still account for them. Used at shutdown,
                where the queue can hold max_retry_batches (100) entries and
                each unreachable-backend attempt costs the HTTP client its full
                retry schedule -- minutes of apparent hang at interpreter exit.

        Returns:
            Number of batches successfully retried
        """
        with self._lock:
            batches_to_retry = self._failed_batches.copy()
            self._failed_batches = []

        success_count = 0

        for position, batch in enumerate(batches_to_retry):
            if deadline is not None and time.monotonic() >= deadline:
                with self._lock:
                    self._failed_batches.extend(batches_to_retry[position:])
                break
            try:
                if self.flush_callback(batch):
                    success_count += 1
                    with self._lock:
                        self._stats['events_sent'] += len(batch)
                        self._stats['batches_sent'] += 1
                else:
                    with self._lock:
                        self._failed_batches.append(batch)
            except Exception:
                with self._lock:
                    self._failed_batches.append(batch)
        
        return success_count
    
    def _start_flush_thread(self) -> None:
        """Start the background flush thread"""
        self._flush_thread = threading.Thread(
            target=self._periodic_flush_loop,
            daemon=True,
            name="AgentCost-Batcher"
        )
        self._flush_thread.start()
    
    def _periodic_flush_loop(self) -> None:
        """Background thread that flushes periodically"""
        while self._running:
            time.sleep(self.flush_interval)
            
            with self._lock:
                if self._batch:
                    self._flush_locked()
                has_failures = bool(self._failed_batches)

            # Also try to retry failed batches periodically
            if has_failures:
                self.retry_failed_batches()
    
    def _join_inflight(self, deadline: float) -> None:
        """Wait for the in-flight send threads, giving up at ``deadline``."""
        with self._lock:
            pending = [t for t in self._inflight if t.is_alive()]
            self._inflight = []

        for thread in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def shutdown(self) -> None:
        """
        Graceful shutdown.
        Flushes remaining events and stops background thread.
        """
        if not self._running:
            return
        
        self._running = False

        if self.debug:
            print("[AgentCost] Shutting down batcher...")

        deadline = time.monotonic() + self.SHUTDOWN_GRACE_SECONDS

        # Wait for sends already in flight before looking at what is buffered.
        # They resolve into either the sent counters or _failed_batches, so
        # doing this first means the retry pass below sees their failures and
        # the drop warning reports an accurate count.
        self._join_inflight(deadline)

        # Flush remaining events synchronously (don't use thread)
        with self._lock:
            if self._batch:
                events = self._batch.copy()
                self._batch = []

                try:
                    if self.flush_callback(events):
                        self._stats['events_sent'] += len(events)
                        self._stats['batches_sent'] += 1
                    else:
                        # A rejected final flush is still recoverable below.
                        self._handle_failed_batch(events)
                except Exception as e:
                    if self.debug:
                        print(f"[AgentCost] Final flush failed: {e}")
                    self._handle_failed_batch(events)

        # Anything that failed earlier is still only in memory, and this process
        # is about to exit — this is the last chance to deliver it.
        if self._failed_batches:
            if self.debug:
                print(f"[AgentCost] Retrying {len(self._failed_batches)} failed batch(es) before exit")
            self.retry_failed_batches(deadline=deadline)
            if self._failed_batches:
                # Unconditional: this is data loss at exit, and staying quiet
                # about it is what makes "nothing shows in the dashboard"
                # impossible to diagnose.
                dropped = sum(len(b) for b in self._failed_batches)
                try:
                    warnings.warn(
                        f"AgentCost dropped {dropped} event(s) that could not be "
                        f"delivered before shutdown.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    # Raises under -W error, and this runs from atexit, where an
                    # escaping exception prints an `atexit._run_exitfuncs`
                    # traceback and can change the process exit status. Losing
                    # events must not also corrupt the caller's shutdown.
                    pass

        # Wait for flush thread to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2)
        
        if self.debug:
            print(f"[AgentCost] Batcher stopped. Stats: {self._stats}")
    
    def get_stats(self) -> Dict:
        """Get batcher statistics"""
        with self._lock:
            return {
                **self._stats,
                'pending_events': len(self._batch),
                'failed_batches': len(self._failed_batches),
            }
    
    def _default_flush(self, events: List[Dict]) -> bool:
        """Default flush behavior (just prints)"""
        print(f"[AgentCost] Would send {len(events)} events (no callback configured)")
        return True


class LocalBatcher(HybridBatcher):
    """
    Batcher variant that stores events locally (for testing/debugging).
    """
    
    def __init__(self, **kwargs):
        self._all_events: List[Dict] = []
        super().__init__(**kwargs)
    
    def _default_flush(self, events: List[Dict]) -> bool:
        """Store events locally instead of sending"""
        self._all_events.extend(events)
        return True
    
    def get_all_events(self) -> List[Dict]:
        """Get all stored events"""
        return self._all_events.copy()
    
    def clear_events(self) -> None:
        """Clear stored events"""
        self._all_events = []
