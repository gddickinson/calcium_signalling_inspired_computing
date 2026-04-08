"""
HybridComputing -- multi-store computing task implementations.

Demonstrates hierarchical memory, cross-store processing, and
specialized computation using the MultiStoreComputer.
"""

import numpy as np
from .computer import MultiStoreComputer


class HybridComputing:
    def __init__(self, computer):
        self.computer = computer

    def hierarchical_memory(self, data_sequence, retrieval_pattern, steps=50):
        """Demonstrate hierarchical memory with data moving between stores."""
        self.computer = MultiStoreComputer()
        write_results = []
        for i, data in enumerate(data_sequence):
            if i % 3 == 0:
                priority = 'high'
            elif i % 3 == 1:
                priority = 'medium'
            else:
                priority = 'low'
            success, store = self.computer.write_data(data, priority=priority)
            write_results.append({
                'data': data, 'priority': priority,
                'success': success, 'store': store
            })
        self.computer.run_simulation(steps=steps)
        retrieval_results = []
        for query in retrieval_pattern:
            results = self.computer.query(query)
            retrieval_results.append({'query': query, 'results': results})
        return {
            'write_results': write_results,
            'retrieval_results': retrieval_results,
            'store_activations': self.computer.store_activations
        }

    def cross_store_processing(self, input_data, processing_steps=30):
        """Demonstrate computation that requires multiple stores."""
        self.computer = MultiStoreComputer()
        if isinstance(input_data, (list, tuple)) and len(input_data) >= 3:
            fast_component = input_data[0]
            medium_component = input_data[1]
            slow_component = input_data[2]
        else:
            fast_component = 0.8
            medium_component = 0.5
            slow_component = 0.3

        self.computer.write_data(fast_component, priority='high')
        self.computer.write_data(medium_component, priority='medium')
        self.computer.write_data(slow_component, priority='low')

        processing_states = []
        for _ in range(processing_steps):
            state = {
                name: {
                    'activation': np.copy(store.activation),
                    'content': np.copy(store.contents)
                } for name, store in self.computer.stores.items()
            }
            processing_states.append(state)
            self.computer.update()

        final_result = self.computer.query([fast_component, medium_component, slow_component])
        return {
            'processing_states': processing_states,
            'final_result': final_result
        }

    def specialized_computation(self, task_data):
        """Demonstrate specialized computation in different stores."""
        self.computer = MultiStoreComputer()
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
        for store_name, task in tasks.items():
            store = self.computer.stores[store_name]
            if isinstance(task['data'], (list, tuple, np.ndarray)) and len(task['data']) > 1:
                store.write(slice(0, min(len(task['data']), store.size)), task['data'])
            else:
                store.write(0, task['data'])

        results = {}

        # Primary: Fast lookup
        primary = self.computer.stores['primary']
        if len(primary.activation) > 0:
            max_idx = np.argmax(primary.activation)
            lookup_result = primary.read(max_idx)
            results['primary'] = {
                'task': 'rapid_lookup',
                'result': lookup_result,
                'latency': 1.0 / primary.speed
            }

        # Secondary: Pattern matching
        secondary = self.computer.stores['secondary']
        if len(secondary.activation) > 0:
            active = secondary.activation > 0.3
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
            patterns = []
            for region in pattern_regions:
                if len(region) > 1:
                    pattern_data = secondary.read(slice(region[0], region[-1] + 1))
                    patterns.append({'region': region, 'data': pattern_data})
            results['secondary'] = {
                'task': 'pattern_matching',
                'patterns': patterns,
                'latency': 1.0 / secondary.speed
            }

        # Tertiary: Complex analysis
        tertiary = self.computer.stores['tertiary']
        if len(tertiary.activation) > 0:
            weights = tertiary.activation
            weighted_sum = np.zeros(3)
            for i in range(tertiary.size):
                weighted_sum += weights[i] * tertiary.contents[i]
            weight_total = weights.sum()
            integrated_result = weighted_sum / weight_total if weight_total > 0 else np.zeros(3)
            results['tertiary'] = {
                'task': 'complex_analysis',
                'integrated_result': integrated_result,
                'latency': 1.0 / tertiary.speed
            }

        self.computer.run_simulation(steps=20)

        # Combine results
        store_weights = {'primary': 0.5, 'secondary': 0.3, 'tertiary': 0.2}
        combined_result = np.zeros(3)
        weight_total = 0
        for store_name, store_result in results.items():
            w = store_weights.get(store_name, 0)
            if store_name == 'primary' and 'result' in store_result:
                combined_result += w * store_result['result']
                weight_total += w
            elif store_name == 'secondary' and 'patterns' in store_result and store_result['patterns']:
                pattern_avg = np.mean([d for d in store_result['patterns'][0]['data']], axis=0)
                combined_result += w * pattern_avg
                weight_total += w
            elif store_name == 'tertiary' and 'integrated_result' in store_result:
                combined_result += w * store_result['integrated_result']
                weight_total += w
        if weight_total > 0:
            combined_result /= weight_total
        results['combined'] = combined_result
        return results
