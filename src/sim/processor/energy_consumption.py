import matplotlib.pyplot as plt

from utils.plot_style import set_plot_style


def calculate_energy_consumption(time, frequency, is_idle=False):
    '''
        param: time, frequency (GHz)

        Power = P_static + P_dynamic
        P_dynamic = C * V^2 * f 
        P_static = I_leakage * V

        V = k * f   (simplified voltage-frequency scaling)

        Power = I_leakage * V + C * V^2 * f
              = I_leakage * k * f + C * (k * f)^2 * f

        Energy = Power * time
               = [I_leakage * k * f + C * (k * f)^2] * f * (N / f)
               = [I_leakage * k * f + C * (k * f)^2] * N
    '''
    k = 0.2 / 1e9           # V/GHz -> V/Hz
    C = 1e-9                # capacitance (F)
    I_leakage = 3e-1        # leakage current (A)

    time = time / 1e9       # convert ns to s
    frequency *= 1e9        # convert GHz to Hz

    V = k * frequency
    P_static = I_leakage * V
    P_dynamic = C * (V ** 2) * frequency

    if not is_idle:
        Power = P_static + P_dynamic  # in Watts
    else:
        Power = P_static 

    Energy = Power * time  # in Joules

    return Energy


if __name__ == "__main__":
    # plot energy consumption vs frequency
    set_plot_style()
    frequencies = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # GHz
    times = [1000] * len(frequencies)  # ns
    energies = [calculate_energy_consumption(
        times[i], frequencies[i]) for i in range(len(frequencies))]
    idle_energies = [calculate_energy_consumption(
        times[i], frequencies[i], is_idle=True) for i in range(len(frequencies))]
    active_energies = [energies[i] - idle_energies[i]
                       for i in range(len(frequencies))]

    plt.plot(frequencies, energies, marker='o', label="Total energy consumption")
    plt.plot(frequencies, idle_energies, marker='^', label="Idle energy consumption")
    plt.plot(frequencies, active_energies, marker='d', label="Active energy consumption")
    print(idle_energies)
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Energy Consumption (Joules)')
    # plt.title('Energy Consumption vs Frequency')
    plt.tight_layout()
    plt.savefig('outputs/energy_consumption_vs_frequency.png', dpi=300)
    plt.close()
