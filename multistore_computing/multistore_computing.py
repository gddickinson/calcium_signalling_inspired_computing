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

The classes live in the package modules (store.py, computer.py, hybrid.py); this file
re-exports them and keeps the original end-to-end demo: python multistore_computing/multistore_computing.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .store import CalciumStore
    from .computer import MultiStoreComputer
    from .hybrid import HybridComputing
else:  # run as a script: import the package from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from multistore_computing.store import CalciumStore
    from multistore_computing.computer import MultiStoreComputer
    from multistore_computing.hybrid import HybridComputing

__all__ = ['CalciumStore', 'MultiStoreComputer', 'HybridComputing']

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
