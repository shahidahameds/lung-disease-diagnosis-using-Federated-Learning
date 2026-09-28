from phase3_engine import run_experiment


def run_fl_dp(comm_rounds=5, local_epochs=1, smoke_test=False, **_):
    return run_experiment("fl_dp", rounds=comm_rounds, local_epochs=local_epochs, smoke_test=smoke_test)


if __name__ == "__main__":
    run_fl_dp(smoke_test=True)
