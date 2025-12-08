from sim.simulator import run_simulation


def main():
    run_simulation(env_type="ROUND_ROBIN", freq=1.0)


if __name__ == "__main__":
    main()
