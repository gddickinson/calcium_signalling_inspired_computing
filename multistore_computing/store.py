"""
CalciumStore -- a single computational memory store.

Models one calcium-inspired storage compartment with read/write
operations, activation decay, and query-by-similarity.
"""

import numpy as np
from collections import deque


class CalciumStore:
    def __init__(self, name, size, speed, capacity, reliability=0.99,
                analog_ratio=0.5, persistence=0.95):
        """
        Initialize a calcium store for computation.

        Parameters:
        - name: Store name (e.g., "ER", "Lysosome")
        - size: Storage capacity (number of elements)
        - speed: Access speed (higher is faster)
        - capacity: Maximum activation per element
        - reliability: Reliability of information storage
        - analog_ratio: Ratio of analog vs. digital processing
        - persistence: How long information persists (decay rate)
        """
        self.name = name
        self.size = size
        self.speed = speed
        self.capacity = capacity
        self.reliability = reliability
        self.analog_ratio = analog_ratio
        self.persistence = persistence

        self.activation = np.zeros(size)
        self.contents = np.zeros((size, 3))
        self.recent_changes = deque(maxlen=10)
        self.access_history = []

        self.properties = {
            'specialized_for': [],
            'connection_strength': 0.5,
            'threshold': 0.3
        }

    def write(self, address, data, activation_level=1.0):
        """Write data to the store."""
        if isinstance(address, slice):
            start = address.start or 0
            stop = address.stop or self.size
            step = address.step or 1
            start = max(0, min(start, self.size-1))
            stop = max(0, min(stop, self.size))
            addresses = range(start, stop, step)
            if len(addresses) == 0:
                return False
            if hasattr(data, '__iter__') and len(data) >= len(addresses):
                for i, addr in enumerate(addresses):
                    self._write_single(addr, data[i], activation_level)
            else:
                for addr in addresses:
                    self._write_single(addr, data, activation_level)
        else:
            if 0 <= address < self.size:
                return self._write_single(address, data, activation_level)
            else:
                return False
        return True

    def _write_single(self, address, data, activation_level):
        """Write to a single address."""
        if np.random.random() > self.reliability:
            return False
        self.access_history.append(('write', address))
        if data is None:
            self.contents[address] = [0, 0, 0]
        elif isinstance(data, (int, float)):
            value = min(1.0, max(0.0, float(data)))
            self.contents[address] = [value, value * 0.8, value * 0.5]
        elif len(data) == 3:
            self.contents[address] = data
        else:
            value = min(1.0, max(0.0, float(data[0])))
            self.contents[address] = [value, value * 0.8, value * 0.5]
        self.activation[address] = min(self.capacity, activation_level)
        self.recent_changes.append((address, data))
        return True

    def read(self, address):
        """Read data from the store."""
        if isinstance(address, slice):
            start = address.start or 0
            stop = address.stop or self.size
            step = address.step or 1
            start = max(0, min(start, self.size-1))
            stop = max(0, min(stop, self.size))
            addresses = range(start, stop, step)
            return [self._read_single(addr) for addr in addresses]
        else:
            if 0 <= address < self.size:
                return self._read_single(address)
            return None

    def _read_single(self, address):
        """Read from a single address."""
        if np.random.random() > self.reliability:
            return None
        self.access_history.append(('read', address))
        return self.contents[address]

    def update(self):
        """Update store state for one time step."""
        self.activation *= self.persistence
        noise = np.random.normal(0, 0.01, self.size)
        self.activation += noise
        self.activation = np.clip(self.activation, 0, self.capacity)
        self.contents = np.clip(self.contents, 0, 1.0)

    def get_active_regions(self, threshold=None):
        """Get regions with activation above threshold."""
        if threshold is None:
            threshold = self.properties['threshold']
        active_indices = np.where(self.activation > threshold)[0]
        return active_indices, self.activation[active_indices]

    def process_query(self, query_vector):
        """Process a query against this store."""
        if len(query_vector) != 3:
            query_rgb = [float(query_vector[0]), float(query_vector[0]) * 0.8, float(query_vector[0]) * 0.5]
        else:
            query_rgb = query_vector
        similarities = np.zeros(self.size)
        for i in range(self.size):
            dist = np.sqrt(sum((self.contents[i, j] - query_rgb[j])**2 for j in range(3)))
            similarities[i] = max(0, 1 - dist)
        weighted_sim = similarities * self.activation
        if np.max(weighted_sim) > 0:
            best_idx = np.argmax(weighted_sim)
            result = self.contents[best_idx]
            confidence = weighted_sim[best_idx]
        else:
            result = np.zeros(3)
            confidence = 0
        return result, confidence
