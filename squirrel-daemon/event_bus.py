import threading
import queue
import time
from typing import Callable, Dict, List, Any, Tuple


class EventBus:
    """A tiny thread-based pub/sub event bus.

    - publish(topic, data): enqueue an event (non-blocking)
    - subscribe(topic, handler): register a callable to receive events
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1000)
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="EventBus", daemon=True)
        self._thread.start()

    def publish(self, topic: str, data: Any) -> None:
        try:
            self._q.put_nowait((str(topic), data))
        except queue.Full:
            # Drop oldest to make space, then put
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((str(topic), data))
            except queue.Full:
                pass

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._subs.setdefault(str(topic), []).append(handler)

    def _worker(self) -> None:
        while self._running:
            try:
                topic, data = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            handlers: List[Callable[[Any], None]]
            with self._lock:
                handlers = list(self._subs.get(topic, []))
            for h in handlers:
                try:
                    h(data)
                except Exception:
                    # Swallow handler errors to keep the bus alive
                    pass

    def stop(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

