from phase3_engine import run_experiment


def run_basic_fl(comm_rounds=3, local_epochs=1, smoke_test=False, full_data=False, resume=True):
    return run_experiment("basic_fl", rounds=comm_rounds, local_epochs=local_epochs, smoke_test=smoke_test, full_data=full_data, resume=resume)


if __name__ == "__main__":
    run_basic_fl(smoke_test=True)
