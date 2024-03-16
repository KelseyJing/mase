import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp


from chop.passes import quantize_transform_pass
# from lab3 import search_spaces
import copy
# figure out the correct path
machop_path = Path(".").resolve().parent.parent /"mase"/"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))
import torch
from torchmetrics.classification import MulticlassAccuracy
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger
import argparse
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
from torch import nn
from chop.models import get_model_info, get_model

# config_path = '/mnt/c/users/KelseyJing/mase/mase_output/jsc-three-linear-layers_classification_jsc_2024-02-02/software/training_ckpts/best.ckpt'
set_logging_verbosity("info")


logger = get_logger("chop")
logger.setLevel(logging.INFO)


batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


# ./ch search --config configs/examples/JSC_Three_Linear_Layers.toml --load /mnt/c/users/KelseyJing/mase/mase_output/jsc-three-linear-layers_classification_jsc_2024-02-02/software/training_ckpts/best.ckpt --load-type pl

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


from torch import nn
from chop.passes.graph.utils import get_parent_name


def parse_args():
    parser = argparse.ArgumentParser(description="Model search with command line integration.")
    parser.add_argument("--multiplier", type=int, default=1, help="Base multiplier for channel sizes.")
    return parser.parse_args()




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
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            # if name == "output_only":
            #     out_features = out_features * config["channel_multiplier"]
            # elif name == "both":
            #     in_features = in_features * config["channel_multiplier"]
            #     out_features = out_features * config["channel_multiplier"]
            # elif name == "input_only":
            #     in_features = in_features * config["channel_multiplier"]
            # 根据name和channel_multiplier的类型处理不同情况
           
            if name == "output_only":
                multiplier = config["channel_multiplier"] if not isinstance(config["channel_multiplier"], dict) else config["channel_multiplier"].get("output", 3)
                out_features *= multiplier


            elif name == "both":
                input_multiplier = config["channel_multiplier"] if not isinstance(config["channel_multiplier"], dict) else config["channel_multiplier"].get("input", 1)
                output_multiplier = config["channel_multiplier"] if not isinstance(config["channel_multiplier"], dict) else config["channel_multiplier"].get("output", 1)
                in_features *= input_multiplier
                out_features *= output_multiplier


            elif name == "input_only":
                multiplier = config["channel_multiplier"] if not isinstance(config["channel_multiplier"], dict) else config["channel_multiplier"].get("input", 2)
                in_features *= multiplier


            # 使用更新后的特征维度创建新的Linear模块并替换原模块
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # print(f"Modified layer '{name}': in_features={in_features}, out_features={out_features}, bias={bias}")
            if name == "relu":
                relu_layer = nn.ReLU()
                setattr(graph.modules[parent_name], name, relu_layer)


        # print(f"Modified layer '{name}': in_features={in_features}, out_features={out_features}, bias={bias}")  
           
   
   
    return graph, {}










# for multiplier in range(1, 2):


def main():
    channel_multipliers = [1]  # 通道乘数值的列表
    search_spaces = []
    for multiplier in channel_multipliers:
        pass_config = {"by": "name",
            "default": {"config": {"name": None}},
            "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier": {}}},
            "seq_blocks_4": {"config": {"name": "both", "channel_multiplier": {}}},
            "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier": {}}}
        }


       
        pass_config['seq_blocks_2']['config']['channel_multiplier']['output'] = multiplier * 2


        pass_config['seq_blocks_4']['config']['channel_multiplier']['input'] = multiplier * 2
        pass_config['seq_blocks_4']['config']['channel_multiplier']['output'] = multiplier * 4


        pass_config['seq_blocks_6']['config']['channel_multiplier']['input'] = multiplier * 4
       


        search_spaces.append(copy.deepcopy(pass_config))


   
    args = parse_args()
    print(f"Command line argument --multiplier value: {args.multiplier}")
       
   
    model = JSC_Three_Linear_Layers()
    mg = MaseGraph(model=model)
   
   
    print(pass_config)
    print(type(args.multiplier))
    print("Config after applying multiplier:")
    # 使用调整后的配置实例化模型


    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, None)


    print("Before modification:")
    print(model.seq_blocks[2])  # 第一个线性层
    print(model.seq_blocks[4])  # 第二个线性层
    print(model.seq_blocks[6])
    metric = MulticlassAccuracy(num_classes=5)
    num_batchs = 5  # 定义要评估的批次数量
    acc_results = []
    for i, config in enumerate(search_spaces):
        model = JSC_Three_Linear_Layers()  # 重新实例化模型
        # mg = MaseGraph(model=model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": config})
        j = 0
        print("After modification:")
        print(mg.model)


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
    #  results.append((multiplier, avg_acc))
    acc_avg = sum(accs) / len(accs)
   
    acc_results.append(acc_avg)


if __name__ == "__main__":
    main()














