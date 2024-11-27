import os

def run_script(script_name):
    """
    Execute a Python script in the current environment.
    """
    print(f"Running {script_name}...")
    exit_code = os.system(f"python {script_name}")
    if exit_code != 0:
        print(f"Error occurred while running {script_name}. Exiting...")
        exit(1)
    print(f"Finished {script_name}.\n")

if __name__ == "__main__":
    try:
        # Step 1: Generate synthetic data
        run_script("SyntheticDataGenerationScript.py")

        # Step 2: Preprocess the data
        run_script("PreprocessingData.py")

        # Step 3: Train the model
        run_script("GNNModel.py")

        # Step 4: Evaluate and visualize the results
        run_script("EvaluationAndVisualization.py")

        print("All steps completed successfully!")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
