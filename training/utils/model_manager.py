import os
from transformers import AutoModelForImageToText, AutoTokenizer


class ModelManager:
    @staticmethod
    def save_model(model, tokenizer, save_directory, model_name="custom_model"):
        """
        Save a model and its tokenizer using Hugging Face method.

        Args:
            model: The model to save
            tokenizer: Corresponding tokenizer
            save_directory (str): Directory to save the model
            model_name (str): Optional name for the model
        """
        # Ensure save directory exists and is specific to this model
        full_save_path = os.path.join(save_directory, model_name)
        os.makedirs(full_save_path, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(full_save_path)
        tokenizer.save_pretrained(full_save_path)

        print(f"Model saved to {full_save_path}")
        return full_save_path

    @staticmethod
    def load_model(model_path, device="cuda"):
        """
        Load a model and tokenizer flexibly.

        Args:
            model_path (str): Path to saved model
            device (str): Device to load model onto

        Returns:
            tuple: (loaded_model, loaded_tokenizer)
        """
        try:
            # Attempt to load using AutoClasses
            model = AutoModelForImageToText.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Move to specified device
            model = model.to(device)

            return model, tokenizer

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Troubleshooting tips:")
            print("1. Check if model path is correct")
            print("2. Ensure compatible model architecture")
            print("3. Verify tokenizer files are present")
            return None, None

    @staticmethod
    def list_saved_models(base_directory):
        """
        List all saved models in a directory.

        Args:
            base_directory (str): Directory containing saved models

        Returns:
            list: Paths to saved models
        """
        saved_models = []
        for item in os.listdir(base_directory):
            full_path = os.path.join(base_directory, item)
            if os.path.isdir(full_path) and os.path.exists(
                os.path.join(full_path, "pytorch_model.bin")
            ):
                saved_models.append(full_path)
        return saved_models


# Example usage demonstrating model interchangeability
def model_management_workflow():
    # Saving multiple models
    save_directory = "./model_checkpoints"

    # Model 1
    model1 = YourFirstModelClass(...)
    tokenizer1 = YourFirstTokenizerClass(...)
    ModelManager.save_model(model1, tokenizer1, save_directory, "model_variant_1")

    # Model 2
    model2 = YourSecondModelClass(...)
    tokenizer2 = YourSecondTokenizerClass(...)
    ModelManager.save_model(model2, tokenizer2, save_directory, "model_variant_2")

    # List available models
    available_models = ModelManager.list_saved_models(save_directory)
    print("Available models:", available_models)

    # Flexibly load any model
    for model_path in available_models:
        loaded_model, loaded_tokenizer = ModelManager.load_model(model_path)
        if loaded_model:
            print(f"Successfully loaded model from {model_path}")
