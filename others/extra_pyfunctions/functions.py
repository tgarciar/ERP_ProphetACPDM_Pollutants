### THIS SCRIPT CONTAINS EXTRA FUNCTIONS NEEDED TO RUN THE DIFFERENT SCRIPTS IN A SEQUENCE WAY.



def run_script(script_path, working_dir):
# Function that will run a python script in an specified working directory to mantain the output in the right path.

    import subprocess
    import os
    try:
        print(f"Running {script_path} from {working_dir}...")
        result = subprocess.run(['python', script_path],
                                cwd=working_dir,
                                check=True,
                                text=True,
                                capture_output=True)
        print(f"✅ Finished running {script_path}.")
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running {script_path}: {e}")
        print("Output:")
        print(e.output)
        print("Error Output:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ An error occurred while running {script_path}: {e}")
        return False
