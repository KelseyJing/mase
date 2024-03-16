import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import copy
import torch
import torch.nn as nn
# figure out the correct path
machop_path = Path(".").resolve().parent.parent /"mase/machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))
from torchmetrics.classification import MulticlassAccuracy
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger


from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph


from chop.models import get_model_info, get_model
from torch import nn


from chop.passes.graph.utils import get_parent_name
set_logging_verbosity("info")


logger = get_logger("chop")
logger.setLevel(logging.INFO)


batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"




data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()


model_info = get_model_info(model_name)


input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)


dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}


class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )


    def forward(self, x):
        return self.seq_blocks(x)
model = JSC_Three_Linear_Layers()


# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)


def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)




def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError("default value must be provided.")


    for node in graph.fx_graph.nodes:
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
       
        # 确保 node.target 在 graph.modules 中
        if node.target not in graph.modules:
            continue


        ori_module = graph.modules[node.target]


        # 如果当前模块是Linear层，修改其特征维度
        if isinstance(ori_module, nn.Linear):
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias is not None


            if name == "output_only":
                out_features *= config["channel_multiplier"]
            elif name == "both":
                in_features *= config["channel_multiplier"]
                out_features *= config["channel_multiplier"]
            elif name == "input_only":
                in_features *= config["channel_multiplier"]


            # new_module = instantiate_linear(in_features, out_features, bias)
            # parent_name, module_name = get_parent_name(node.target)
            # setattr(graph.modules[parent_name], module_name, new_module)
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
       
        # 对ReLU层的处理可以在这里添加
        # 注意：ReLU层不涉及特征维度的改变，因此您可能想要根据具体需求添加其他类型的修改
            print(f"Modified layer '{name}': in_features={in_features}, out_features={out_features}, bias={bias}")


    return graph, {}




# def redefine_linear_transform_pass(graph, pass_args=None):
#     main_config = pass_args.pop('config')
#     default = main_config.pop('default', None)
#     if default is None:
#         raise ValueError(f"default value must be provided.")
#     i = 0
#     for node in graph.fx_graph.nodes:
#         i += 1
#         # if node name is not matched, it won't be tracked
#         config = main_config.get(node.name, default)['config']
#         name = config.get("name", None)
#         if name is not None:
#             ori_module = graph.modules[node.target]
#             in_features = ori_module.in_features
#             out_features = ori_module.out_features
#             bias = ori_module.bias
#             if name == "output_only":
#                 out_features = out_features * config["channel_multiplier"]
#             elif name == "both":
#                 in_features = in_features * config["channel_multiplier"]
#                 out_features = out_features * config["channel_multiplier"]
#             elif name == "input_only":
#                 in_features = in_features * config["channel_multiplier"]
#             new_module = instantiate_linear(in_features, out_features, bias)
#             parent_name, name = get_parent_name(node.target)
#             setattr(graph.modules[parent_name], name, new_module)


#         print(f"Modified layer '{name}': in_features={in_features}, out_features={out_features}, bias={bias}")
#     return graph, {}




pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}


# this performs the architecture transformation based on the config




channel_multipliers = [1, 2, 3]  # 通道乘数值的列表

search_spaces = []
metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5  # 定义要评估的批次数量
acc_results = []
for multiplier in channel_multipliers:
   
    pass_config['seq_blocks_2']['config']['channel_multiplier'] = multiplier
    pass_config['seq_blocks_4']['config']['channel_multiplier'] = multiplier
    pass_config['seq_blocks_6']['config']['channel_multiplier'] = multiplier
    search_spaces.append(copy.deepcopy(pass_config))
    
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)  


# metric = MulticlassAccuracy(num_classes=5)
# num_batchs = 5  # 定义要评估的批次数量
# acc_results = []
for i, config in enumerate(search_spaces):
    model = JSC_Three_Linear_Layers()  # 重新实例化模型
    mg = MaseGraph(model=model)
    mg, _ = redefine_linear_transform_pass(mg, pass_args={"config": config})
    j = 0

    acc_avg, loss_avg = 0, 0
    accs, losses = [], []

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
       
        if j > num_batchs:  
            break
        j += 1
    # avg_acc = metric.compute().item()  
    # results.append((multiplier, avg_acc))
    acc_avg = sum(accs) / len(accs)
   
    acc_results.append(acc_avg)
    print(f"Channel Multiplier: {channel_multipliers[i]}, Avg Accuracy: {acc_results[-1]}")
best_acc_index = acc_results.index(max(acc_results))
best_multiplier = channel_multipliers[best_acc_index]

print(f"Best Channel Multiplier: {best_multiplier}, Best Avg Accuracy: {acc_results[best_acc_index]}")






   


