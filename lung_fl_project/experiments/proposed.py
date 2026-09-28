from phase3_engine import run_experiment


def run_proposed(comm_rounds=5, local_epochs=1, smoke_test=False, **_):
    return run_experiment("proposed", rounds=comm_rounds, local_epochs=local_epochs, smoke_test=smoke_test)


if __name__ == "__main__":
    run_proposed(smoke_test=True)
