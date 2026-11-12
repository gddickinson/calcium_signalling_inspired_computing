"""Reference numbers for web/static/wave_engine.js from wave_based_computing (deterministic, no random numbers).

Run: python scripts/make_wave_fixture.py [--write]
"""
import numpy as np

from fixture_common import main, quiet

from wave_based_computing import WaveComputing, WaveComputingMedium

NAME = "wave"


def medium_run(boundary):
    m = WaveComputingMedium(size=(40, 30), boundary=boundary)
    m.add_source((8, 15), amplitude=0.5, frequency=0.8)
    m.add_source((32, 15), amplitude=0.5, frequency=0.8, phase=np.pi)
    m.add_obstacle(18, 22, 10, 20)
    m.add_resonant_region(28, 36, 20, 28, frequency=0.8)
    m.set_memory_element((5, 5), 0.6)
    m.run_simulation(steps=40)
    return {"boundary": boundary, "row": m.current[15], "col": m.current[:, 30], "energy": m.energy_history,
            "freq_row": m.frequency[15], "phase_row": m.phase[16], "memory": m.read_memory_element((5, 5))}


def fixture():
    base = WaveComputingMedium(size=(40, 40))
    wc = WaveComputing(base)
    with quiet():
        gates = {op: wc.wave_logic_gates([np.ones((4, 4))] * 2, operation=op, simulation_steps=30)["result_value"]
                 for op in ("and", "or", "xor")}
        mem = wc.wave_memory([0.8, -0.5, 0.3], storage_time=20, retrieval_time=10)["retrieved_data"]
        sig = np.sin(np.linspace(0, 2 * np.pi, 30)) * 0.6
        filt = wc.wave_filter(sig, filter_type="lowpass", cutoff_frequency=0.3, simulation_steps=25)["filtered_signal"]
    return {"media": [medium_run("absorbing"), medium_run("periodic"), medium_run("reflecting")],
            "gates": gates, "memory": mem, "signal": sig, "filtered": filt}


if __name__ == "__main__":
    main(NAME, fixture)
