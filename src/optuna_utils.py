import optuna


def print_study_results(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print(" Number of finished trials:", len(study.trials))
    print(" Number of pruned trials:", len(pruned_trials))
    print(" Number of complete trials:", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print(" Value:", trial.value)
    print(" Params:")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))
