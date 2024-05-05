import torch

from common.data_factory import DataFactory, OgbInfo
from common.model_factory import ModelFactory
from common.utils import CrossEntropyLossSmooth, train_model, evaluate_model
from exp.node_classification_paddle.source.model.gnn_model import GAT
from exp.node_classification_paddle.source.util.load_data import ArixvDataSet


dataset_name = "ogbn-arxiv"
device = "cuda" if torch.cuda.is_available() else "cpu"
root = "dataset/"
dataset = ArixvDataSet(name='arixv', raw_dir='exp/node_classification_paddle/data')

# data_factory = DataFactory(dataset=dataset, device=device)
ogb_info = OgbInfo(dataset_name, root)
data_factory = DataFactory(ogb_info=ogb_info, device=device)

# Model hyperparameters
d_input = data_factory.graph.ndata['x'].shape[1] + data_factory.n_class
d_hidden = 256
d_output = data_factory.n_class
n_head = 3
n_layer = 3
masked = True  # Establish whether or not to include masked label features

# Initialize an instance of the model
model = GAT(d_input, d_hidden, d_output, n_head, n_layer).to(device)
model.masked = masked
# Register the model name and directory so that model_factory will save the best model to "models_dir/name" during training
model_factory = ModelFactory(model, models_dir="saved_models", name="GAT")

# This registers the training loss, valid metric, and test metric so that model_factory will save their values to disk during training
model_factory.add_loss_name("train", mode="min")
model_factory.add_loss_name("valid", mode="max")
model_factory.add_loss_name("test", mode="max")

# Print number of model parameters
model_factory.print_num_params()


# Training hyperparameters
lr = 0.005  # learning rate
criterion = CrossEntropyLossSmooth  # training loss
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=500, verbose=True)

# This registers the optimizer and scheduler so that model_factory will save their state_dict to disk during training
model_factory.set_optimizer(optimizer)
model_factory.set_scheduler(scheduler)


epochs = 500
log_every = 20

for epoch in range(epochs):
    # Train for one epoch
    loss_train = train_model(model_factory, data_factory, criterion)
    # Store last training loss to memory
    model_factory.append_loss("train", loss_train)

    # Validate
    valid_score = evaluate_model(model_factory, data_factory, dataset_name, split_name="valid")
    # Store last validation score to memory
    model_factory.append_loss("valid", valid_score)

    # Test
    test_score = evaluate_model(model_factory, data_factory, dataset_name, split_name="test")
    # Store last test score to memory
    model_factory.append_loss("test", test_score)

    # After each epoch, store the training loss, validation score, test score, optimizer state_dict, and scheduler state_dict to disk
    # If validation score is best, then also save model state_dict to disk
    model_factory.save_best("valid")

    # Log results periodically
    if epoch % log_every == 0:
        # Print results from current epoch
        model_factory.print_last_loss(epoch)
