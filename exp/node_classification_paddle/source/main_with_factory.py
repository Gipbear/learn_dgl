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

data_factory = DataFactory(dataset=dataset, device=device)
# ogb_info = OgbInfo(dataset_name, root)
# data_factory = DataFactory(ogb_info=ogb_info, device=device)

# 模型超参数
d_input = data_factory.graph.ndata['x'].shape[1] + data_factory.n_class
d_hidden = 256
d_output = data_factory.n_class
n_head = 3
n_layer = 3
masked = True  # 是否将 label 加入features

model = GAT(d_input, d_hidden, d_output, n_head, n_layer).to(device)
model.masked = masked
model_factory = ModelFactory(model, models_dir="saved_models", name="GAT")

model_factory.add_loss_name("train", mode="min")
model_factory.add_loss_name("valid", mode="max")
model_factory.add_loss_name("test", mode="max")

model_factory.print_num_params()
model_factory.print_model()


# 训练超参数
lr = 0.005  # learning rate
criterion = CrossEntropyLossSmooth  # training loss
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=500, verbose=True)

model_factory.set_optimizer(optimizer)
model_factory.set_scheduler(scheduler)

epochs = 500
log_every = 20

for epoch in range(epochs):
    loss_train = train_model(model_factory, data_factory, criterion)
    model_factory.append_loss("train", loss_train)

    valid_score = evaluate_model(model_factory, data_factory, dataset_name, split_name="valid")
    model_factory.append_loss("valid", valid_score)

    # test_score = evaluate_model(model_factory, data_factory, dataset_name, split_name="test")
    # model_factory.append_loss("test", test_score)

    # 根据验证集的结果存储最好的参数
    model_factory.save_best("valid")

    if epoch % log_every == 0:
        model_factory.print_last_loss(epoch)

test_score = evaluate_model(model_factory, data_factory, dataset_name, split_name="test")
