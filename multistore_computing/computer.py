"""
MultiStoreComputer -- orchestrates multiple CalciumStore instances.

Manages cross-store communication, data routing by priority, and
hierarchical query search. Provides visualization of store states
and activation history.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .store import CalciumStore


class MultiStoreComputer:
    def __init__(self):
        """Initialize a multi-store computing system."""
        self.stores = {
            'primary': CalciumStore(
                name='Primary', size=100, speed=0.9, capacity=1.0,
                reliability=0.99, analog_ratio=0.7, persistence=0.9
            ),
            'secondary': CalciumStore(
                name='Secondary', size=200, speed=0.5, capacity=1.5,
                reliability=0.95, analog_ratio=0.5, persistence=0.97
            ),
            'tertiary': CalciumStore(
                name='Tertiary', size=500, speed=0.2, capacity=3.0,
                reliability=0.9, analog_ratio=0.3, persistence=0.99
            )
        }
        self.stores['primary'].properties['specialized_for'] = ['rapid_retrieval', 'frequent_access']
        self.stores['secondary'].properties['specialized_for'] = ['pattern_matching', 'intermediate_storage']
        self.stores['tertiary'].properties['specialized_for'] = ['long_term_storage', 'complex_patterns']

        self.connections = {
            ('primary', 'secondary'): 0.8,
            ('secondary', 'primary'): 0.5,
            ('secondary', 'tertiary'): 0.7,
            ('tertiary', 'secondary'): 0.4,
            ('primary', 'tertiary'): 0.3,
            ('tertiary', 'primary'): 0.2
        }
        self.time_step = 0
        self.store_activations = {}
        self.cross_store_transfers = []

    def update(self):
        """Update the system for one time step."""
        for store in self.stores.values():
            store.update()
        self._process_cross_store_communication()
        self.store_activations[self.time_step] = {
            name: np.copy(store.activation) for name, store in self.stores.items()
        }
        self.time_step += 1

    def _process_cross_store_communication(self):
        """Process communication between stores."""
        transfers = []
        for (source_name, target_name), strength in self.connections.items():
            source = self.stores[source_name]
            target = self.stores[target_name]
            active_indices, active_values = source.get_active_regions()
            if len(active_indices) == 0:
                continue
            high_indices = [i for i, v in zip(active_indices, active_values) if v > 0.7]
            if len(high_indices) == 0:
                continue
            n_transfer = max(1, int(len(high_indices) * strength))
            transfer_indices = np.random.choice(
                high_indices, size=min(n_transfer, len(high_indices)), replace=False
            )
            for idx in transfer_indices:
                target_idx = int(idx * target.size / source.size) % target.size
                data = source.read(idx)
                activation = source.activation[idx] * strength
                target.write(target_idx, data, activation)
                transfers.append((source_name, target_name, idx, target_idx, activation))
        self.cross_store_transfers.append(transfers)

    def write_data(self, data, priority='auto'):
        """Write data to the system, selecting appropriate store."""
        if priority == 'auto':
            if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 100:
                priority = 'low'
            elif hasattr(data, 'urgency') and data.urgency > 0.7:
                priority = 'high'
            else:
                priority = 'medium'
        store_map = {'high': 'primary', 'medium': 'secondary', 'low': 'tertiary'}
        store_name = store_map.get(priority, 'secondary')
        store = self.stores[store_name]
        empty_indices = np.where(store.activation < 0.1)[0]
        if len(empty_indices) > 0:
            address = np.random.choice(empty_indices)
        else:
            address = np.argmin(store.activation)

        if isinstance(data, (list, tuple, np.ndarray)):
            if len(data) > store.size:
                chunk_size = store.size
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                success = store.write(slice(0, chunk_size), chunks[0])
                if len(chunks) > 1:
                    remainder = [item for chunk in chunks[1:] for item in chunk]
                    overflow_success, _ = self.write_data(remainder, priority='low')
                    return success and overflow_success, f"{store_name} + overflow"
            else:
                success = store.write(slice(0, len(data)), data)
        else:
            success = store.write(address, data)
        return success, store_name

    def query(self, query_data):
        """Query the system with hierarchical search."""
        results = []
        if not isinstance(query_data, (list, tuple, np.ndarray)) or len(query_data) != 3:
            if isinstance(query_data, (int, float)):
                query_vector = [float(query_data), float(query_data) * 0.8, float(query_data) * 0.5]
            else:
                query_vector = [0.5, 0.4, 0.3]
        else:
            query_vector = query_data
        for store_name in ['primary', 'secondary', 'tertiary']:
            store = self.stores[store_name]
            result, confidence = store.process_query(query_vector)
            if confidence > 0.4:
                results.append({
                    'result': result,
                    'confidence': confidence,
                    'source': store_name
                })
                if confidence > 0.8 and store_name == 'primary':
                    break
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def run_simulation(self, steps=100):
        """Run simulation for specified number of steps."""
        for _ in range(steps):
            self.update()

    def visualize_stores(self):
        """Visualize the current state of stores."""
        fig, axes = plt.subplots(1, len(self.stores), figsize=(15, 5))
        for i, (name, store) in enumerate(self.stores.items()):
            ax = axes[i]
            ax.bar(range(store.size), store.activation)
            ax.set_title(f"{name} Store")
            ax.set_xlabel("Address")
            ax.set_ylabel("Activation")
            ax.set_ylim(0, store.capacity * 1.1)
        plt.tight_layout()
        return fig

    def visualize_content(self):
        """Visualize the content of stores."""
        fig, axes = plt.subplots(len(self.stores), 1, figsize=(12, 8))
        for i, (name, store) in enumerate(self.stores.items()):
            ax = axes[i]
            height = max(1, store.size // 50)
            width = min(50, store.size)
            if store.size < width:
                width = store.size
                height = 1
            content_2d = np.zeros((height, width, 3))
            for j in range(min(store.size, height * width)):
                row = j // width
                col = j % width
                content_2d[row, col] = store.contents[j]
            ax.imshow(content_2d)
            ax.set_title(f"{name} Store Content")
            ax.set_yticks([])
            if width <= 50:
                ax.set_xticks(range(0, width, max(1, width // 10)))
                ax.set_xticklabels([str(x) for x in range(0, store.size, max(1, store.size // 10))])
            else:
                ax.set_xticks([])
        plt.tight_layout()
        return fig

    def visualize_connections(self):
        """Visualize connections between stores."""
        G = nx.DiGraph()
        for name, store in self.stores.items():
            G.add_node(name, size=store.size, speed=store.speed)
        for (source, target), strength in self.connections.items():
            G.add_edge(source, target, weight=strength)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        sizes = [store.size/50 for store in self.stores.values()]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')
        edge_widths = [strength * 3 for strength in self.connections.values()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                              arrowsize=15, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(G, pos)
        edge_labels = {(s, t): f"{st:.1f}" for (s, t), st in self.connections.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title('Store Connections')
        plt.axis('off')
        return plt.gcf()

    def visualize_activation_history(self, window=None):
        """Visualize activation history across stores."""
        if not self.store_activations:
            return None
        fig, ax = plt.subplots(figsize=(12, 6))
        times = sorted(self.store_activations.keys())
        if window is not None:
            start_time = max(0, times[-1] - window)
            times = [t for t in times if t >= start_time]
        activations = {}
        for name in self.stores.keys():
            activations[name] = [np.mean(self.store_activations[t][name]) for t in times]
        for name, values in activations.items():
            ax.plot(times, values, label=name)
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
