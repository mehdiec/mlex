from datetime import datetime
from pathlib import Path
import yaml


def save_config_and_checkpoints(config, checkpoint_folder, model_name):
    # Generate a unique folder name using model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_name = f"{model_name}_{timestamp}"
    print(checkpoint_folder)
    print(new_folder_name)
    new_folder_path = Path(checkpoint_folder) / new_folder_name

    # Create the new folder
    new_folder_path.mkdir(parents=True, exist_ok=True)

    config["checkpoint_folder"] = str(new_folder_path)

    # Save the updated configuration back to the config file
    updated_config_path = new_folder_path / "updated_config.yaml"
    with open(updated_config_path, "w") as file:
        yaml.safe_dump(config, file)

    print(f"Configuration and checkpoints will be saved in: {new_folder_path}")

    return new_folder_path
