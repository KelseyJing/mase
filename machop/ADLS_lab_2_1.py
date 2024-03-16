import sys
import logging
import os

from pathlib import Path
from pprint import pprint as pp
import torch

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity


from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph
from chop.models import get_model_info, get_model

from chop.passes.graph.utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

set_logging_verbosity("info")


# 模型和数据集名称
batch_size = 8
model_name = "jsc-toy-self"
# model_name = "jsc-tiny"
dataset_name = "jsc"


# 定义你的 JSC_Toy_Self 模型
# class JSC_Toy_Self(nn.Module):
#     def __init__(self, info):
#         super(JSC_Toy_Self, self).__init__()
#         self.seq_blocks = nn.Sequential(
#             # 1st LogicNets Layer
#             nn.BatchNorm1d(16),  # input_quant       # 0
#             nn.ReLU(),  # 1
#             nn.Linear(16, 55),  # linear              # 2
#             nn.BatchNorm1d(55),  # output_quant       # 3
#             nn.ReLU(),  # 4
#             # 2nd LogicNets Layer
#             nn.Linear(55, 26),  # 5
#             nn.BatchNorm1d(26),  # 6
#             nn.ReLU(),  # 7
#             #3rd
#             nn.Linear(26, 15),  # 5
#             nn.BatchNorm1d(15),  # 6
#             nn.ReLU(),
#             #4th
#             nn.Linear(15, 8),  # 5
#             nn.BatchNorm1d(8),  # 6
#             nn.ReLU(),
#             #5th LogicNets Layer
#             nn.Linear(8, 5),  # 8
#              nn.BatchNorm1d(5),  # 9
#              nn.ReLU(),
#         )

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()


# CHECKPOINT_PATH = "/mnt/c/users/KelseyJing/mase/mase_output/jsc-toy_classification_jsc_2024-02-04/software/training_ckpts/best.ckpt"
CHECKPOINT_PATH = "/mnt/c/users/KelseyJing/mase/mase_output/jsc-toy-self_classification_jsc_2024-02-04/software/training_ckpts/best.ckpt"
# CHECKPOINT_PATH = "/mnt/c/users/KelseyJing/mase/mase_output/jsc-tiny_classification_jsc_2024-02-04/software/training_ckpts/best.ckpt"

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)


model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)


input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)


# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)


# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)


mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)






# report graph is an analysis pass that shows you the detailed information in the graph
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)


pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}


mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})


pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 16,
            "weight_frac_width": 2,
            # bias
            "bias_width": 16,
            "bias_frac_width": 4,
        }
    },

}


from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph


ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})


mg, _ = quantize_transform_pass(mg, pass_args)
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")


# def traverse_and_compare(mg, ori_mg):
    
#     for node in mg.fx_graph.nodes:
#         ori_node = next((n for n in ori_mg.fx_graph.nodes if n.name == node.name), None)
#         print(f"Node Operation:{node.op}")
#         print(f"Node: {node.name}")
#         if ori_node:
#             print("Found corresponding node in original graph.")
#         else:
#             print("No corresponding node found in original graph.")

# traverse_and_compare(mg, ori_mg)



for ori_node, node in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    # Get the actual module target of the original node and the quantized node (such as torch.nn.Module object)
    ori_target = get_node_actual_target(ori_node)
    target = get_node_actual_target(node)
    
    # Compare two modules to check if they are of the same type
    if type(ori_target) != type(target):
        print(f"Node {ori_node.name} has different module types between original and quantized models.")
        print(f"Original module type: {type(ori_target)}")
        print(f"Quantized module type: {type(target)}")


    # Compare module weights
        if hasattr(ori_target, 'weight') and hasattr(target, 'weight'):
            print(f"Comparing weights for node {ori_node.name}:")
            print(f"Original weights: {ori_target.weight}")
            print(f"Quantized weights: {target.weight}")

    # Print the weight information of the node after quantization
        if type(ori_target) != type(target):
       
            print(f'Weight of quantized module: {get_node_actual_target(node).weight}')
            test_input = torch.randn(get_node_actual_target(node).in_features)
            print(f'Random generated test input outcomes: {test_input}')

    # Use the original node and the quantized node to process the test random input respectively, and print the output results
            print(f'Output for original modules(ori_node): {get_node_actual_target(ori_node)(test_input)}')
            print(f'Output for quantized modules(node): {get_node_actual_target(node)(test_input)}')



# for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
#     if (type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n))):
#         print(f'difference found at name: {n.name}, mase_type: {get_mase_type(n)}, mase_op: {get_mase_op(n)}\n original module: {type(get_node_actual_target(ori_n))} --> new module {type(get_node_actual_target(n))}')
#         print(f'weight of original module: {get_node_actual_target(ori_n).weight}')
#         print(f'weight of quantized module: {get_node_actual_target(n).weight}')
#         test_input = torch.randn(get_node_actual_target(n).in_features)
#         print(f'random generated test input: {test_input}')
#         print(f'output for original module: {get_node_actual_target(ori_n)(test_input)}')
#         print(f'output for quantized module: {get_node_actual_target(n)(test_input)}')

#
            