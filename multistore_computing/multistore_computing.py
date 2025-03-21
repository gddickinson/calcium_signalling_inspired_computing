"""
Multi-Store Hybrid Computing Architecture Inspired by Calcium Stores

This simulation models a computational architecture inspired by multiple calcium
stores (ER, lysosomes, etc.) that serve different signaling needs. It demonstrates
how hybrid computing systems with specialized memory types can efficiently process
different types of information and communication between stores.

The model implements:
1. Hierarchical memory system with different store types
2. Cross-store communication (inspired by store-operated calcium entry)
3. Hybrid analog-digital processing
4. Specialized computation in different stores
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.signal import convolve2d
from collections import deque
import heapq

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

        # Initialize state variables
        self.activation = np.zeros(size)
        self.contents = np.zeros((size, 3))  # RGB representation of content
        self.recent_changes = deque(maxlen=10)  # Track recent changes
        self.access_history = []  # Track access patterns

        # Store-specific properties
        self.properties = {
            'specialized_for': [],
            'connection_strength': 0.5,
            'threshold': 0.3
        }

    def write(self, address, data, activation_level=1.0):
        """
        Write data to the store.

        Parameters:
        - address: Address to write to (index or slice)
        - data: Data to write
        - activation_level: Activation level for the write

        Returns:
        - success: Whether write was successful
        """
        if isinstance(address, slice):
            start = address.start or 0
            stop = address.stop or self.size
            step = address.step or 1

            # Ensure bounds
            start = max(0, min(start, self.size-1))
            stop = max(0, min(stop, self.size))

            # Create address range
            addresses = range(start, stop, step)

            if len(addresses) == 0:
                return False

            # Check if data is iterable and matches size
            if hasattr(data, '__iter__') and len(data) >= len(addresses):
                for i, addr in enumerate(addresses):
                    self._write_single(addr, data[i], activation_level)
            else:
                # Replicate single value
                for addr in addresses:
                    self._write_single(addr, data, activation_level)
        else:
            # Single address
            if 0 <= address < self.size:
                return self._write_single(address, data, activation_level)
            else:
                return False

        return True

    def _write_single(self, address, data, activation_level):
        """Write to a single address"""
        # Check reliability - random failures
        if np.random.random() > self.reliability:
            return False

        # Record access
        self.access_history.append(('write', address))

        # Write data
        if data is None:
            # Handle None data - set default values
            self.contents[address] = [0, 0, 0]  # Default to black/no value
        elif isinstance(data, (int, float)):
            # Convert to RGB with intensity based on value
            value = min(1.0, max(0.0, float(data)))
            self.contents[address] = [value, value * 0.8, value * 0.5]
        elif len(data) == 3:
            # Direct RGB values
            self.contents[address] = data
        else:
            # Convert to RGB
            value = min(1.0, max(0.0, float(data[0])))
            self.contents[address] = [value, value * 0.8, value * 0.5]

        # Set activation
        self.activation[address] = min(self.capacity, activation_level)

        # Record change
        self.recent_changes.append((address, data))

        return True

    def read(self, address):
        """
        Read data from the store.

        Parameters:
        - address: Address to read from (index or slice)

        Returns:
        - data: Data read from store
        """
        if isinstance(address, slice):
            start = address.start or 0
            stop = address.stop or self.size
            step = address.step or 1

            # Ensure bounds
            start = max(0, min(start, self.size-1))
            stop = max(0, min(stop, self.size))

            # Create address range
            addresses = range(start, stop, step)

            # Read data for each address
            data = []
            for addr in addresses:
                data.append(self._read_single(addr))

            return data
        else:
            # Single address
            if 0 <= address < self.size:
                return self._read_single(address)
            else:
                return None

    def _read_single(self, address):
        """Read from a single address"""
        # Check reliability - random failures
        if np.random.random() > self.reliability:
            return None

        # Record access
        self.access_history.append(('read', address))

        # Return RGB content
        return self.contents[address]

    def update(self):
        """Update store state for one time step"""
        # Apply persistence (decay)
        self.activation *= self.persistence

        # Apply background noise
        noise = np.random.normal(0, 0.01, self.size)
        self.activation += noise
        self.activation = np.clip(self.activation, 0, self.capacity)

        # Ensure content values stay in range after noise
        self.contents = np.clip(self.contents, 0, 1.0)

    def get_active_regions(self, threshold=None):
        """Get regions with activation above threshold"""
        if threshold is None:
            threshold = self.properties['threshold']

        active_indices = np.where(self.activation > threshold)[0]
        return active_indices, self.activation[active_indices]

    def process_query(self, query_vector):
        """
        Process a query against this store.

        Parameters:
        - query_vector: Query representation

        Returns:
        - result: Query result
        - confidence: Confidence in result
        """
        # Convert query to comparable format
        if len(query_vector) != 3:
            query_rgb = [float(query_vector[0]), float(query_vector[0]) * 0.8, float(query_vector[0]) * 0.5]
        else:
            query_rgb = query_vector

        # Calculate similarity to each element
        similarities = np.zeros(self.size)
        for i in range(self.size):
            # Euclidean distance in RGB space
            dist = np.sqrt(sum((self.contents[i, j] - query_rgb[j])**2 for j in range(3)))
            # Convert distance to similarity (0 to 1)
            similarities[i] = max(0, 1 - dist)

        # Weight by activation level
        weighted_sim = similarities * self.activation

        # Find best matches
        if np.max(weighted_sim) > 0:
            best_idx = np.argmax(weighted_sim)
            result = self.contents[best_idx]
            confidence = weighted_sim[best_idx]
        else:
            result = np.zeros(3)
            confidence = 0

        return result, confidence


class MultiStoreComputer:
    def __init__(self):
        """Initialize a multi-store computing system"""
        # Create different types of stores
        self.stores = {
            # Fast, low-capacity, highly reliable store (similar to ER)
            'primary': CalciumStore(
                name='Primary',
                size=100,
                speed=0.9,
                capacity=1.0,
                reliability=0.99,
                analog_ratio=0.7,
                persistence=0.9
            ),

            # Medium speed, medium capacity (similar to mitochondria)
            'secondary': CalciumStore(
                name='Secondary',
                size=200,
                speed=0.5,
                capacity=1.5,
                reliability=0.95,
                analog_ratio=0.5,
                persistence=0.97
            ),

            # Slow, high-capacity, less reliable store (similar to lysosomes)
            'tertiary': CalciumStore(
                name='Tertiary',
                size=500,
                speed=0.2,
                capacity=3.0,
                reliability=0.9,
                analog_ratio=0.3,
                persistence=0.99
            )
        }

        # Specialize stores
        self.stores['primary'].properties['specialized_for'] = ['rapid_retrieval', 'frequent_access']
        self.stores['secondary'].properties['specialized_for'] = ['pattern_matching', 'intermediate_storage']
        self.stores['tertiary'].properties['specialized_for'] = ['long_term_storage', 'complex_patterns']

        # Set up store connections
        self.connections = {
            ('primary', 'secondary'): 0.8,  # Strong connection
            ('secondary', 'primary'): 0.5,  # Medium connection
            ('secondary', 'tertiary'): 0.7,  # Strong connection
            ('tertiary', 'secondary'): 0.4,  # Weak connection
            ('primary', 'tertiary'): 0.3,    # Weak connection
            ('tertiary', 'primary'): 0.2     # Very weak connection
        }

        # Initialize state variables
        self.time_step = 0
        self.store_activations = {}
        self.cross_store_transfers = []

    def update(self):
        """Update the system for one time step"""
        # First update individual stores
        for name, store in self.stores.items():
            store.update()

        # Then handle cross-store communication
        self._process_cross_store_communication()

        # Record store activations for analysis
        self.store_activations[self.time_step] = {
            name: np.copy(store.activation) for name, store in self.stores.items()
        }

        self.time_step += 1

    def _process_cross_store_communication(self):
        """Process communication between stores"""
        transfers = []

        # For each store pair
        for (source_name, target_name), strength in self.connections.items():
            source = self.stores[source_name]
            target = self.stores[target_name]

            # Get active regions in source
            active_indices, active_values = source.get_active_regions()

            if len(active_indices) == 0:
                continue

            # Filter to only highly active regions
            high_threshold = 0.7
            high_indices = [i for i, v in zip(active_indices, active_values) if v > high_threshold]

            if len(high_indices) == 0:
                continue

            # Sample a subset based on connection strength
            n_transfer = max(1, int(len(high_indices) * strength))
            transfer_indices = np.random.choice(high_indices, size=min(n_transfer, len(high_indices)), replace=False)

            # Transfer data between stores
            for idx in transfer_indices:
                # Map source index to target (scale by relative sizes)
                target_idx = int(idx * target.size / source.size)

                # Ensure target index is valid
                if target_idx >= target.size:
                    target_idx = target_idx % target.size

                # Get data from source
                data = source.read(idx)

                # Scale activation by connection strength
                activation = source.activation[idx] * strength

                # Write to target
                target.write(target_idx, data, activation)

                # Record transfer
                transfers.append((source_name, target_name, idx, target_idx, activation))

        self.cross_store_transfers.append(transfers)

    def write_data(self, data, priority='auto'):
        """
        Write data to the system, automatically selecting appropriate store.

        Parameters:
        - data: Data to write
        - priority: Priority level ('high', 'medium', 'low') or 'auto'

        Returns:
        - success: Whether write was successful
        - store_name: Name of store where data was written
        """
        if priority == 'auto':
            # Determine priority based on data characteristics
            if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 100:
                priority = 'low'  # Large data
            elif hasattr(data, 'urgency') and data.urgency > 0.7:
                priority = 'high'  # Data marked as urgent
            else:
                priority = 'medium'  # Default

        # Select store based on priority
        if priority == 'high':
            store_name = 'primary'
        elif priority == 'medium':
            store_name = 'secondary'
        else:  # low
            store_name = 'tertiary'

        store = self.stores[store_name]

        # Find suitable address (prefer empty space)
        empty_indices = np.where(store.activation < 0.1)[0]

        if len(empty_indices) > 0:
            # Choose random empty location
            address = np.random.choice(empty_indices)
        else:
            # Choose least active location
            address = np.argmin(store.activation)

        # Convert data to appropriate format
        if isinstance(data, (list, tuple, np.ndarray)):
            # Ensure data fits
            if len(data) > store.size:
                # Chunking & overflow handling
                chunk_size = store.size
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

                # Write first chunk
                success = store.write(slice(0, chunk_size), chunks[0])

                # Handle overflow to other stores if needed
                if len(chunks) > 1:
                    remainder = [item for chunk in chunks[1:] for item in chunk]
                    overflow_success, _ = self.write_data(remainder, priority='low')
                    return success and overflow_success, f"{store_name} + overflow"
            else:
                # Data fits in store
                success = store.write(slice(0, len(data)), data)
        else:
            # Single value
            success = store.write(address, data)

        return success, store_name

    def query(self, query_data):
        """
        Query the system with hierarchical search.

        Parameters:
        - query_data: Query to search for

        Returns:
        - results: Best matching results
        - source: Source store for each result
        """
        results = []

        # Convert query to vector representation if needed
        if not isinstance(query_data, (list, tuple, np.ndarray)) or len(query_data) != 3:
            if isinstance(query_data, (int, float)):
                query_vector = [float(query_data), float(query_data) * 0.8, float(query_data) * 0.5]
            else:
                query_vector = [0.5, 0.4, 0.3]  # Default vector
        else:
            query_vector = query_data

        # Query each store with priority
        store_priority = ['primary', 'secondary', 'tertiary']

        for store_name in store_priority:
            store = self.stores[store_name]
            result, confidence = store.process_query(query_vector)

            if confidence > 0.4:  # Only include reasonably confident results
                results.append({
                    'result': result,
                    'confidence': confidence,
                    'source': store_name
                })

                # If very confident result from fast store, can stop
                if confidence > 0.8 and store_name == 'primary':
                    break

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def run_simulation(self, steps=100):
        """Run simulation for specified number of steps"""
        for _ in range(steps):
            self.update()

    def visualize_stores(self):
        """Visualize the current state of stores"""
        fig, axes = plt.subplots(1, len(self.stores), figsize=(15, 5))

        for i, (name, store) in enumerate(self.stores.items()):
            # Plot activation levels
            ax = axes[i]
            ax.bar(range(store.size), store.activation)
            ax.set_title(f"{name} Store")
            ax.set_xlabel("Address")
            ax.set_ylabel("Activation")
            ax.set_ylim(0, store.capacity * 1.1)

        plt.tight_layout()
        return fig

    def visualize_content(self):
        """Visualize the content of stores"""
        fig, axes = plt.subplots(len(self.stores), 1, figsize=(12, 8))

        for i, (name, store) in enumerate(self.stores.items()):
            # Plot content as colored cells
            ax = axes[i]

            # Create 2D representation of content
            height = max(1, store.size // 50)
            width = min(50, store.size)
            if store.size < width:
                width = store.size
                height = 1

            # Reshape contents for visualization
            content_2d = np.zeros((height, width, 3))
            for j in range(min(store.size, height * width)):
                row = j // width
                col = j % width
                content_2d[row, col] = store.contents[j]

            ax.imshow(content_2d)
            ax.set_title(f"{name} Store Content")
            ax.set_yticks([])

            # Custom x-axis labels
            if width <= 50:
                ax.set_xticks(range(0, width, max(1, width // 10)))
                ax.set_xticklabels([str(x) for x in range(0, store.size, max(1, store.size // 10))])
            else:
                ax.set_xticks([])

        plt.tight_layout()
        return fig

    def visualize_connections(self):
        """Visualize connections between stores"""
        G = nx.DiGraph()

        # Add nodes
        for name, store in self.stores.items():
            G.add_node(name, size=store.size, speed=store.speed)

        # Add edges
        for (source, target), strength in self.connections.items():
            G.add_edge(source, target, weight=strength)

        # Create figure
        plt.figure(figsize=(8, 6))

        # Create layout
        pos = nx.spring_layout(G)

        # Draw nodes with size based on store size
        sizes = [store.size/50 for store in self.stores.values()]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')

        # Draw edges with width based on connection strength
        edge_widths = [strength * 3 for strength in self.connections.values()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                              arrowsize=15, connectionstyle='arc3,rad=0.1')

        # Add labels
        nx.draw_networkx_labels(G, pos)

        # Add edge labels
        edge_labels = {(source, target): f"{strength:.1f}"
                      for (source, target), strength in self.connections.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title('Store Connections')
        plt.axis('off')

        return plt.gcf()

    def visualize_activation_history(self, window=None):
        """
        Visualize activation history across stores.

        Parameters:
        - window: Time window to display (None for all)
        """
        if not self.store_activations:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine time range
        times = sorted(self.store_activations.keys())
        if window is not None:
            start_time = max(0, times[-1] - window)
            times = [t for t in times if t >= start_time]

        # Calculate mean activation for each store over time
        activations = {}
        for name in self.stores.keys():
            activations[name] = [np.mean(self.store_activations[t][name]) for t in times]

        # Plot activation curves
        for name, values in activations.items():
            ax.plot(times, values, label=name)

        # Plot transfer events
        transfer_times = []
        transfer_counts = []

        for i, transfers in enumerate(self.cross_store_transfers):
            if i in times:
                transfer_times.append(i)
                transfer_counts.append(len(transfers))

        if transfer_times:
            ax2 = ax.twinx()
            ax2.bar(transfer_times, transfer_counts, alpha=0.3, color='gray', label='Transfers')
            ax2.set_ylabel('Transfer Count')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Store Activation History')
        ax.legend(loc='upper left')

        if transfer_times:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        return fig


# Implementation of specific computing tasks
class HybridComputing:
    def __init__(self, computer):
        self.computer = computer

    def hierarchical_memory(self, data_sequence, retrieval_pattern, steps=50):
        """
        Demonstrate hierarchical memory with data moving between stores.

        Parameters:
        - data_sequence: Sequence of data to write
        - retrieval_pattern: Pattern of retrievals to perform
        - steps: Number of simulation steps

        Returns:
        - Results of memory operation
        """
        # Reset computer for clean experiment
        self.computer = MultiStoreComputer()

        # Store data with different priorities
        write_results = []
        for i, data in enumerate(data_sequence):
            # Alternate priorities
            if i % 3 == 0:
                priority = 'high'
            elif i % 3 == 1:
                priority = 'medium'
            else:
                priority = 'low'

            success, store = self.computer.write_data(data, priority=priority)
            write_results.append({
                'data': data,
                'priority': priority,
                'success': success,
                'store': store
            })

        # Run simulation to allow cross-store transfers
        self.computer.run_simulation(steps=steps)

        # Perform retrievals
        retrieval_results = []
        for query in retrieval_pattern:
            results = self.computer.query(query)
            retrieval_results.append({
                'query': query,
                'results': results
            })

        return {
            'write_results': write_results,
            'retrieval_results': retrieval_results,
            'store_activations': self.computer.store_activations
        }

    def cross_store_processing(self, input_data, processing_steps=30):
        """
        Demonstrate computation that requires multiple stores.

        Parameters:
        - input_data: Input data
        - processing_steps: Number of processing steps

        Returns:
        - Processing results
        """
        # Reset computer
        self.computer = MultiStoreComputer()

        # Split data into different components
        if isinstance(input_data, (list, tuple)) and len(input_data) >= 3:
            # Use RGB components
            fast_component = input_data[0]  # Red channel to primary
            medium_component = input_data[1]  # Green channel to secondary
            slow_component = input_data[2]  # Blue channel to tertiary
        else:
            # Create synthetic components
            fast_component = 0.8  # High activation in primary
            medium_component = 0.5  # Medium activation in secondary
            slow_component = 0.3  # Low activation in tertiary

        # Write components to appropriate stores
        self.computer.write_data(fast_component, priority='high')
        self.computer.write_data(medium_component, priority='medium')
        self.computer.write_data(slow_component, priority='low')

        # Run simulation to allow cross-store processing
        processing_states = []

        for _ in range(processing_steps):
            # Save current state
            state = {
                name: {
                    'activation': np.copy(store.activation),
                    'content': np.copy(store.contents)
                } for name, store in self.computer.stores.items()
            }
            processing_states.append(state)

            # Update
            self.computer.update()

        # Final query to get combined result
        final_result = self.computer.query([fast_component, medium_component, slow_component])

        return {
            'processing_states': processing_states,
            'final_result': final_result
        }

    def specialized_computation(self, task_data):
        """
        Demonstrate specialized computation in different stores.

        Parameters:
        - task_data: Data for different types of tasks

        Returns:
        - Task results
        """
        # Reset computer
        self.computer = MultiStoreComputer()

        # Define specialized tasks for each store
        tasks = {
            'primary': {
                'name': 'rapid_lookup',
                'data': task_data.get('lookup_data', [0.9, 0.1, 0.2])
            },
            'secondary': {
                'name': 'pattern_matching',
                'data': task_data.get('pattern_data', [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
            },
            'tertiary': {
                'name': 'complex_analysis',
                'data': task_data.get('complex_data', np.random.random(20))
            }
        }

        # Load task data into stores
        for store_name, task in tasks.items():
            store = self.computer.stores[store_name]

            # Write task data
            if isinstance(task['data'], (list, tuple, np.ndarray)) and len(task['data']) > 1:
                store.write(slice(0, min(len(task['data']), store.size)), task['data'])
            else:
                store.write(0, task['data'])

        # Run specialized computation in each store
        results = {}

        # Primary: Fast lookup (max activation)
        primary = self.computer.stores['primary']
        if len(primary.activation) > 0:
            max_idx = np.argmax(primary.activation)
            lookup_result = primary.read(max_idx)
            results['primary'] = {
                'task': 'rapid_lookup',
                'result': lookup_result,
                'latency': 1.0 / primary.speed  # Lower is faster
            }

        # Secondary: Pattern matching
        secondary = self.computer.stores['secondary']
        if len(secondary.activation) > 0:
            # Find patterns (regions with similar activation)
            threshold = 0.3
            active = secondary.activation > threshold

            # Group consecutive active regions
            pattern_regions = []
            current_region = []

            for i, is_active in enumerate(active):
                if is_active:
                    current_region.append(i)
                elif current_region:
                    pattern_regions.append(current_region)
                    current_region = []

            if current_region:
                pattern_regions.append(current_region)

            # Extract patterns
            patterns = []
            for region in pattern_regions:
                if len(region) > 1:  # Only meaningful patterns
                    pattern_data = secondary.read(slice(region[0], region[-1] + 1))
                    patterns.append({
                        'region': region,
                        'data': pattern_data
                    })

            results['secondary'] = {
                'task': 'pattern_matching',
                'patterns': patterns,
                'latency': 1.0 / secondary.speed
            }

        # Tertiary: Complex analysis (integration)
        tertiary = self.computer.stores['tertiary']
        if len(tertiary.activation) > 0:
            # Calculate weighted average of content
            weights = tertiary.activation
            weighted_sum = np.zeros(3)

            for i in range(tertiary.size):
                content = tertiary.contents[i]
                weighted_sum += weights[i] * content

            weight_total = weights.sum()
            if weight_total > 0:
                integrated_result = weighted_sum / weight_total
            else:
                integrated_result = np.zeros(3)

            results['tertiary'] = {
                'task': 'complex_analysis',
                'integrated_result': integrated_result,
                'latency': 1.0 / tertiary.speed
            }

        # Run simulation to allow cross-store interaction
        self.computer.run_simulation(steps=20)

        # Final result combining all stores
        weights = {
            'primary': 0.5,
            'secondary': 0.3,
            'tertiary': 0.2
        }

        combined_result = np.zeros(3)
        weight_total = 0

        for store_name, store_result in results.items():
            if store_name == 'primary' and 'result' in store_result:
                combined_result += weights[store_name] * store_result['result']
                weight_total += weights[store_name]
            elif store_name == 'secondary' and 'patterns' in store_result and store_result['patterns']:
                # Use first pattern
                pattern_avg = np.mean([d for d in store_result['patterns'][0]['data']], axis=0)
                combined_result += weights[store_name] * pattern_avg
                weight_total += weights[store_name]
            elif store_name == 'tertiary' and 'integrated_result' in store_result:
                combined_result += weights[store_name] * store_result['integrated_result']
                weight_total += weights[store_name]

        if weight_total > 0:
            combined_result /= weight_total

        results['combined'] = combined_result

        return results


# Example usage
if __name__ == "__main__":
    # Initialize multi-store computer
    computer = MultiStoreComputer()

    print("Initializing multi-store computing system...")

    # Visualize initial state
    fig = computer.visualize_stores()
    plt.savefig('multistore_initial.png')
    plt.close()

    # Visualize connections
    fig = computer.visualize_connections()
    plt.savefig('multistore_connections.png')
    plt.close()

    # Write some test data
    print("Writing test data to stores...")
    test_data = [
        0.9,  # High priority
        [0.5, 0.2, 0.7],  # Medium priority
        np.random.random(50)  # Low priority
    ]

    for i, data in enumerate(test_data):
        priority = ['high', 'medium', 'low'][i]
        success, store = computer.write_data(data, priority=priority)
        print(f"  {priority} priority data written to {store} (success: {success})")

    # Run simulation
    print("Running simulation...")
    computer.run_simulation(steps=50)

    # Visualize final state
    fig = computer.visualize_stores()
    plt.savefig('multistore_final.png')
    plt.close()

    fig = computer.visualize_content()
    plt.savefig('multistore_content.png')
    plt.close()

    fig = computer.visualize_activation_history()
    plt.savefig('multistore_history.png')
    plt.close()

    # Test hybrid computing tasks
    hybrid_computing = HybridComputing(computer)

    # Test 1: Hierarchical Memory
    print("\nTesting hierarchical memory...")
    data_sequence = [
        0.8,
        [0.2, 0.4, 0.6],
        0.5,
        [0.1, 0.3, 0.5],
        0.3
    ]

    retrieval_pattern = [0.8, 0.5, 0.3]

    memory_results = hybrid_computing.hierarchical_memory(
        data_sequence=data_sequence,
        retrieval_pattern=retrieval_pattern,
        steps=30
    )

    # Print results
    print("  Write results:")
    for i, result in enumerate(memory_results['write_results']):
        print(f"    Data {i}: {result['priority']} priority -> {result['store']}")

    print("  Retrieval results:")
    for i, result in enumerate(memory_results['retrieval_results']):
        if result['results']:
            best_match = result['results'][0]
            print(f"    Query {result['query']}: best match from {best_match['source']} store (confidence: {best_match['confidence']:.2f})")
        else:
            print(f"    Query {result['query']}: no match found")

    # Test 2: Cross-Store Processing
    print("\nTesting cross-store processing...")
    input_data = [0.8, 0.5, 0.3]  # RGB data

    processing_results = hybrid_computing.cross_store_processing(
        input_data=input_data,
        processing_steps=20
    )

    if processing_results['final_result']:
        best_match = processing_results['final_result'][0]
        print(f"  Final result: {best_match['result']} from {best_match['source']} store (confidence: {best_match['confidence']:.2f})")

    # Test 3: Specialized Computation
    print("\nTesting specialized computation...")
    task_data = {
        'lookup_data': 0.95,
        'pattern_data': [[0.6, 0.6, 0.6], [0.65, 0.65, 0.65], [0.7, 0.7, 0.7]],
        'complex_data': np.linspace(0, 1, 10)
    }

    computation_results = hybrid_computing.specialized_computation(task_data)

    for store, result in computation_results.items():
        if store != 'combined':
            print(f"  {store.capitalize()} store ({result['task']}): latency {result['latency']:.2f}")

    if 'combined' in computation_results:
        print(f"  Combined result: {computation_results['combined']}")

    print("\nThis multi-store hybrid computing model demonstrates how:")
    print("1. Different store types can be specialized for different tasks")
    print("2. Information can flow between stores via cross-store communication")
    print("3. Hybrid analog-digital processing can emerge from store interactions")
    print("4. Hierarchical memory access optimizes retrieval performance")
