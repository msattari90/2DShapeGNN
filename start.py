import os

def run_script(script_name):
    """
    Execute a Python script in the current environment.
    
    Args:
        script_name (str): Name of the Python script to run.
    """
    print(f"Running {script_name}...")
    # Execute the script using the current Python interpreter
    exit_code = os.system(f"python {script_name}")
    if exit_code != 0:
        # If the script fails (non-zero exit code), print an error message and exit
        print(f"Error occurred while running {script_name}. Exiting...")
        exit(1)
    print(f"Finished {script_name}.\n")

if __name__ == "__main__":
    """
    Main pipeline orchestrates the execution of the project steps sequentially:
    1. Generate synthetic data.
    2. Preprocess the data into graph format.
    3. Train the Graph Neural Network (GNN) model.
    4. Evaluate and visualize the model's results.
    """
    try:
        # Step 1: Generate synthetic data
        run_script("SyntheticDataGenerationScript.py")

        # Step 2: Preprocess the data into a format suitable for GNN training
        run_script("PreprocessingData.py")

        # Step 3: Train the Graph Neural Network model
        run_script("GNNModel.py")

        # Step 4: Evaluate and visualize the results
        run_script("EvaluationAndVisualization.py")
        
        # Print a success message once all steps are completed
        print("All steps completed successfully!")
    except KeyboardInterrupt:
        # Handle interruption by the user (Ctrl+C)
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        # Handle any unexpected errors during execution
        print(f"An unexpected error occurred: {e}")
