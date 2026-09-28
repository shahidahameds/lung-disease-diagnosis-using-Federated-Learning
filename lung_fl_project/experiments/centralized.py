from phase3_engine import run_experiment


def run_centralized(epochs=1, smoke_test=False):
    return run_experiment("centralized", local_epochs=epochs, smoke_test=smoke_test)


if __name__ == "__main__":
    run_centralized(smoke_test=True)
