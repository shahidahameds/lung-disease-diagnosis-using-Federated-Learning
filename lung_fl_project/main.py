import argparse
import sys
from data.dataset_manager import inspect_dataset

def main():
    parser = argparse.ArgumentParser(description="Lung Disease FL Experiment Prototype")
    parser.add_argument('--inspect', action='store_true', help='Phase 1: Inspect dataset and environment')
    parser.add_argument('--experiment', type=str, choices=['centralized', 'basic_fl', 'fl_dp', 'hierarchical', 'proposed', 'all'], help='Run specific experiment')
    parser.add_argument('--smoke-test', action='store_true', help='Run a fast minimal test')
    parser.add_argument('--full-data', action='store_true', help='Use all available train/test images instead of the demo subset')
    parser.add_argument('--no-resume', action='store_true', help='Ignore an existing experiment checkpoint')
    
    args = parser.parse_args()

    if args.inspect:
        success = inspect_dataset()
        if not success:
            sys.exit(1)
        sys.exit(0)
        
    elif args.experiment == 'centralized':
        from experiments.centralized import run_centralized
        run_centralized(epochs=1, smoke_test=args.smoke_test)
        
    elif args.experiment == 'basic_fl':
        from experiments.basic_fl import run_basic_fl
        run_basic_fl(comm_rounds=1 if args.smoke_test else 3, smoke_test=args.smoke_test, full_data=args.full_data, resume=not args.no_resume)
        
    elif args.experiment == 'hierarchical':
        from experiments.hierarchical_fl import run_hierarchical_fl
        run_hierarchical_fl(comm_rounds=1 if args.smoke_test else 3, smoke_test=args.smoke_test)
        
    elif args.experiment == 'fl_dp':
        from experiments.fl_dp import run_fl_dp
        run_fl_dp(comm_rounds=1 if args.smoke_test else 3, smoke_test=args.smoke_test)
        
    elif args.experiment == 'proposed':
        from experiments.proposed import run_proposed
        run_proposed(comm_rounds=1 if args.smoke_test else 3, smoke_test=args.smoke_test)
        
    elif args.experiment == 'all':
        print("Running ALL experiments...")
        from experiments.centralized import run_centralized
        from experiments.basic_fl import run_basic_fl
        from experiments.fl_dp import run_fl_dp
        from experiments.hierarchical_fl import run_hierarchical_fl
        from experiments.proposed import run_proposed
        from privacy.reconstruction import run_reconstruction
        from security.audit import run_tamper_test
        
        smoke = args.smoke_test
        from phase3_engine import run_all
        run_all(smoke_test=smoke)
        return
        run_reconstruction()
        run_tamper_test()
        
        from experiments.plot_convergence import generate_convergence_plot
        generate_convergence_plot()
        
        # Save experiment config
        import json
        from config import NUM_HOSPITALS, DEPARTMENTS_PER_HOSPITAL, IMAGE_SIZE, BATCH_SIZE
        config_data = {
            "NUM_HOSPITALS": NUM_HOSPITALS,
            "DEPARTMENTS_PER_HOSPITAL": DEPARTMENTS_PER_HOSPITAL,
            "IMAGE_SIZE": IMAGE_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "RANDOM_SEED": 42
        }
        with open(os.path.join('results', 'experiment_config.json'), 'w') as f:
            json.dump(config_data, f, indent=4)
            
        print("All experiments completed.")
        
    elif args.experiment:
        print(f"Experiment '{args.experiment}' is not yet implemented.")
    
    elif args.smoke_test:
        print("Running smoke test for Centralized CNN...")
        from experiments.centralized import run_centralized
        run_centralized(epochs=1, smoke_test=True)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
