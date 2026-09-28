from phase3_engine import run_experiment


def run_hierarchical_fl(comm_rounds=5, local_epochs=1, smoke_test=False):
    return run_experiment("hierarchical", rounds=comm_rounds, local_epochs=local_epochs, smoke_test=smoke_test)


if __name__ == "__main__":
    run_hierarchical_fl(smoke_test=True)
