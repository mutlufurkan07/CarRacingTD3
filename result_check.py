import os.path

import numpy as np
import pathlib
import matplotlib.pyplot as plt

plt.style.use("seaborn")
if __name__ == "__main__":
    smoothing_window_size = 5
    working_dir = pathlib.Path(__file__).parent.resolve()
    result_file_name = "TD3_CarRacing-v0_32.npy"
    result_dir = os.path.join(working_dir, "results", result_file_name)
    results = np.load(result_dir)
    r1 = results[0:1000]
    r2 = results[1000:]
    r2 = r2[(r2 > 200)]
    results = np.concatenate((r1, r2), axis=0)

    if smoothing_window_size > 1:
        results = np.convolve(results, np.ones(smoothing_window_size) / smoothing_window_size, mode="same")

    x_axis = np.arange(0, len(results) * 5e3, 5e3)
    plt.plot(x_axis, results)
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title("evaluation returns over timestep")
    plt.show()

    pass
