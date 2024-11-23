from torch.utils.data import DataLoader
import torch
from typing import Optional


def train_with_periodic_validation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    validate_every: int = 50,
    epochs: Optional[int] = None,
):
    """
    Training loop with periodic validation

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        validate_every: Number of batches between validation checks
        epochs: Number of epochs (if None, runs indefinitely)
    """
    model.train()
    train_iter = iter(train_loader)
    batch_counter = 0
    epoch = 0

    while True:
        try:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator when it's exhausted
                train_iter = iter(train_loader)
                epoch += 1
                if epochs is not None and epoch >= epochs:
                    break
                batch = next(train_iter)

            # Training step
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Periodic validation
            if batch_counter % validate_every == 0:
                model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_inputs, val_targets = val_batch
                        val_outputs = model(val_inputs)
                        val_loss += criterion(val_outputs, val_targets).item()

                        _, predicted = val_outputs.max(1)
                        total += val_targets.size(0)
                        correct += predicted.eq(val_targets).sum().item()

                val_loss /= len(val_loader)
                accuracy = 100.0 * correct / total

                print(f"Epoch {epoch}, Batch {batch_counter}:")
                print(f"Validation Loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%")

                model.train()  # Switch back to training mode

            batch_counter += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break


# Example usage:
"""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = YourModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Run indefinitely
train_with_periodic_validation(train_loader, val_loader, model, criterion, optimizer)

# Or run for specific number of epochs
train_with_periodic_validation(train_loader, val_loader, model, criterion, optimizer, epochs=10)
"""


def quick_validate(model, val_loader, sample_ratio=0.1):
    model.eval()
    total_size = len(val_loader.dataset)
    n_samples = int(total_size * sample_ratio)

    # Randomly sample indices
    all_indices = torch.randperm(total_size)[:n_samples]

    # Create a temporary subset dataset
    temp_dataset = torch.utils.data.Subset(val_loader.dataset, all_indices)
    quick_loader = DataLoader(
        temp_dataset,
        batch_size=val_loader.batch_size,
        num_workers=val_loader.num_workers,
    )

    running_loss = 0
    with torch.no_grad():
        for batch in quick_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    model.train()
    return running_loss / len(quick_loader)


# Usage in training loop:
while True:
    for batch in train_loader:
        # ... training step ...

        if batch_counter % 50 == 0:
            # Random 10% validation check
            quick_loss = quick_validate(model, val_loader, sample_ratio=0.1)
            print(f"Quick validation loss (10% sample): {quick_loss:.4f}")

        batch_counter += 1
