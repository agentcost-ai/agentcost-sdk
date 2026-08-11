"""
AgentCost HTTP Client

Handles communication with the AgentCost backend API.
Features retry logic, timeouts, rate limiting, and error handling.
"""

import requests
import time
import threading
import logging
import warnings
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# warnings.warn() is invisible in most production logging setups; everything
# user-actionable is reported through both channels so at least one lands.
logger = logging.getLogger("agentcost")


OUTCOME_RECORD = "outcome"


def partition_records(records: List[Dict]):
    """Split a batch into LLM events and run-outcome records."""
    events, outcomes = [], []
    for record in records:
        if record.get("record_type") == OUTCOME_RECORD:
            outcomes.append({k: v for k, v in record.items() if k != "record_type"})
        else:
            events.append(record)
    return events, outcomes


class RateLimiter:
    """Simple rate limiter to prevent overwhelming the backend"""
    
    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        """
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = []
        self._lock = threading.Lock()
    
    def acquire(self) -> float:
        """
        Try to acquire a request slot.
        
        Returns:
            Time to wait before making request (0 if no wait needed)
        """
        with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            self._requests = [t for t in self._requests if now - t < self.window_seconds]
            
            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return 0.0
            else:
                # Calculate wait time until oldest request expires
                oldest = min(self._requests)
                wait_time = self.window_seconds - (now - oldest)
                return max(0, wait_time)
    
    def wait_and_acquire(self) -> None:
        """Wait if necessary, then acquire a slot"""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)
            self.acquire()


class AgentCostHTTPClient:
    """HTTP client for sending telemetry to AgentCost backend"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        timeout: float = 10.0,
        max_retries: int = 3,
        debug: bool = False
    ):
        """
        Args:
            api_key: User's AgentCost API key
            base_url: Backend API URL (default: AGENTCOST_API_URL env var or https://api.agentcost.tech)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            debug: Enable debug logging
        """
        self.api_key = api_key
        if not base_url:
            import os
            base_url = os.getenv("AGENTCOST_API_URL", "https://api.agentcost.tech")
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.debug = debug
        self._closed = False
        self._fatal_reported = False
        
        # Rate limiter (10 requests per second max)
        self._rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        
        # Create session with retry logic
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """Create requests session with retry logic"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry these HTTP codes
            allowed_methods=["POST", "GET"],  # Updated from deprecated method_whitelist
            # Hand the final response back instead of raising RetryError. The
            # default (True) raises before requests can attach the response, so
            # an exhausted 429 lands in the catch-all `except Exception` and is
            # dropped without ever reaching _report_fatal -- which means a
            # project that hit its budget cap gets no warning at all. With this
            # off, raise_for_status() below turns it into an HTTPError that
            # still carries the response.
            raise_on_status=False,
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def send_events(self, project_id: str, records: List[Dict]) -> bool:
        """
        Send a batch of records to the backend.

        Args:
            project_id: User's project ID
            records: Event dicts, plus any outcome records the batcher carried

        Returns:
            True if successful, False otherwise
        """
        # Apply rate limiting
        self._rate_limiter.wait_and_acquire()

        url = f"{self.base_url}/v1/events/batch"

        from . import __version__
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'AgentCost-SDK/{__version__}',
            'X-AgentCost-SDK-Version': __version__,
        }

        events, outcomes = partition_records(records)
        payload = {
            'project_id': project_id,
            'events': events,
        }
        if outcomes:
            payload['outcomes'] = outcomes

        if self.debug:
            print(f"[AgentCost] Sending {len(events)} events to {url}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check response
            response.raise_for_status()  # Raises exception for 4xx/5xx
            
            # Parse response
            data = response.json()
            
            if data.get('status') == 'ok':
                # The backend drops malformed events individually and still
                # returns 200; without this check a fully-rejected batch looked
                # like a success and the events silently never appeared.
                rejected = data.get('events_rejected') or 0
                if rejected:
                    reasons = data.get('rejected') or []
                    first = reasons[0].get('reason') if reasons else 'unknown'
                    msg = (
                        f"AgentCost: backend rejected {rejected} of "
                        f"{len(events)} events (first reason: {first})."
                    )
                    logger.error(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                if self.debug:
                    print(f"[AgentCost] Sent {len(events)} events successfully")
                return True
            else:
                if self.debug:
                    print(f"[AgentCost] Error: Backend returned error: {data}")
                return False
        
        except requests.exceptions.Timeout:
            self._report_unreachable(f"request timed out after {self.timeout}s")
            if self.debug:
                print(f"[AgentCost] Error: Request timed out after {self.timeout}s")
            return False

        except requests.exceptions.ConnectionError as e:
            self._report_unreachable(str(e))
            if self.debug:
                print(f"[AgentCost] Error: Connection error: {e}")
            return False
        
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            text = e.response.text if e.response is not None else str(e)
            self._report_fatal(status, text)
            if self.debug:
                print(f"[AgentCost] Error: HTTP error: {status} - {text}")
            return False
        
        except Exception as e:
            if self.debug:
                print(f"[AgentCost] Error: Unexpected error: {e}")
            return False
    
    def _report_unreachable(self, detail: str) -> None:
        """Warn once when the backend cannot be reached at all.

        Without this, a DNS/firewall/proxy problem was debug-only: zero events,
        zero output, indistinguishable from the SDK not being installed.
        """
        if self._fatal_reported:
            return
        self._fatal_reported = True
        msg = (
            f"AgentCost cannot reach {self.base_url}: {detail[:200]}. "
            f"Events are being retried but will be dropped if this persists."
        )
        logger.error(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _report_fatal(self, status: Optional[int], detail: str) -> None:
        """
        Warn once about a rejection the caller has to act on, not wait out.

        A bad api_key or project_id otherwise produces zero events and zero
        output — identical to the SDK not being installed — while the batcher
        retries the doomed payload every few seconds until it drops it. 429 is
        listed because the session already exhausted its retries by the time it
        surfaces here, and a budget cap will not clear on its own.
        """
        if self._fatal_reported or status not in (401, 403, 404, 422, 429):
            return
        self._fatal_reported = True
        reason = {
            401: "the API key was rejected",
            403: "the API key is not allowed to write to this project",
            404: "the project was not found",
            422: "the event payload was rejected as invalid",
            429: "the server is rate limiting or the project budget cap was reached",
        }[status]
        snippet = (detail or "").strip().replace("\n", " ")[:200]
        hint = ""
        if status == 403:
            hint = (
                " Note: project_id must be the project UUID from "
                "Settings, not the project name."
            )
        msg = (
            f"AgentCost is not recording events: {reason} (HTTP {status}). "
            f"Check api_key/project_id in track_costs.init().{hint} "
            f"Server said: {snippet}"
        )
        logger.error(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def test_connection(self) -> bool:
        """Test if backend is reachable"""
        url = f"{self.base_url}/v1/health"
        
        try:
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_project_info(self, project_id: str) -> Optional[Dict]:
        """
        Get project information from backend
        
        Args:
            project_id - Project ID
            
        Returns:
            Project info dict or None if failed
        """
        url = f"{self.base_url}/v1/projects/{project_id}"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if self.debug:
                print(f"[AgentCost] Failed to get project info: {e}")
            return None
    
    def close(self) -> None:
        """Close the session and release resources"""
        if not self._closed:
            self.session.close()
            self._closed = True
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures session is closed"""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - cleanup on garbage collection"""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


class MockHTTPClient:
    """
    Mock HTTP client for testing and offline development.
    Stores events locally instead of sending to backend.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.sent_events: List[Dict] = []
        self.send_count = 0
    
    def send_events(self, project_id: str, events: List[Dict]) -> bool:
        """Store events locally"""
        self.sent_events.extend(events)
        self.send_count += 1
        
        if self.debug:
            print(f"[AgentCost Mock] Stored {len(events)} events (total: {len(self.sent_events)})")
        
        return True
    
    def test_connection(self) -> bool:
        """Always returns True"""
        return True
    
    def get_all_events(self) -> List[Dict]:
        """Get all stored events"""
        return self.sent_events.copy()
    
    def clear(self) -> None:
        """Clear stored events"""
        self.sent_events = []
        self.send_count = 0
    
    def close(self) -> None:
        """No-op for mock client"""
