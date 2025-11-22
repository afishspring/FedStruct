import os
import sys
import json


pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)


from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedGCN.FedGCN_server import FedGCNServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.FedPub.fedpub_server import FedPubServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    fedGCN_partitioning,
    partition_graph,
)


def set_up_system():
    LOGGER.info(json.dumps(config.config, indent=4))

    graph = define_graph(config.dataset.dataset_name)

    if config.model.smodel_type in ["DGCN", "CentralDGCN"]:
        graph.calc_abar(
            config.structure_model.DGCN_layers,
            method=config.model.smodel_type,
            pruning=config.subgraph.prune,
        )

    graph.add_masks(
        train_ratio=config.subgraph.train_ratio,
        test_ratio=config.subgraph.test_ratio,
    )

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )

    # # [START - 新增的绘图代码]
    # try:
    #     LOGGER.info("Generating label distribution plot...")
    #     output_dir = "figures/"
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     num_subgraphs = len(subgraphs)
    #     num_classes = graph.num_classes
        
    #     # 收集每个子图的标签计数
    #     all_counts = []
    #     for subgraph in subgraphs:
    #         labels = subgraph.y.numpy()  # 转到CPU和Numpy以便bincount
    #         if len(labels) > 0:
    #             # 确保bincount的长度至少为num_classes
    #             counts = np.bincount(labels, minlength=num_classes)
    #         else:
    #             # 处理空子图
    #             counts = np.zeros(num_classes, dtype=int)
    #         all_counts.append(counts)
        
    #     # 转换为Numpy数组 (num_subgraphs, num_classes)
    #     distributions = np.array(all_counts)
        
    #     # --- 创建堆叠条形图 ---
    #     fig, ax = plt.subplots(figsize=(16, 9))
    #     subgraph_indices = np.arange(num_subgraphs)
    #     bottoms = np.zeros(num_subgraphs)
        
    #     # 使用一个颜色映射表
    #     colors = plt.cm.get_cmap('gist_rainbow', num_classes)
        
    #     for i in range(num_classes):
    #         counts_for_class_i = distributions[:, i]
    #         ax.bar(
    #             subgraph_indices, 
    #             counts_for_class_i, 
    #             bottom=bottoms, 
    #             label=f'Class {i}', 
    #             color=colors(i / num_classes) # 为每个类别应用不同颜色
    #         )
    #         bottoms += counts_for_class_i
            
    #     ax.set_xlabel('Subgraph ID')
    #     ax.set_ylabel('Number of Nodes')
    #     ax.set_title(f'Label Distribution across {num_subgraphs} Subgraphs ({config.dataset.dataset_name}, {config.subgraph.partitioning} partitioning)')
    #     ax.set_xticks(subgraph_indices) # 确保每个子图都有一个刻度
    #     ax.set_xticklabels([str(i) for i in subgraph_indices])
        
    #     # 将图例放在图表外部
    #     ax.legend(title='Class', bbox_to_anchor=(1.04, 1), loc='upper left')
        
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    #     # 调整布局以便图例能完整显示
    #     plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
    #     # 保存图像
    #     plot_filename = f"label_dist_{config.dataset.dataset_name}_{config.subgraph.partitioning}_{num_subgraphs}subgraphs_{now}.png"
    #     save_file_path = os.path.join(output_dir, plot_filename)
    #     plt.savefig(save_file_path)
    #     LOGGER.info(f"Saved label distribution plot to {save_file_path}")
    #     plt.close(fig)  # 关闭图像，释放内存
        
    # except Exception as e:
    #     LOGGER.error(f"Failed to generate label distribution plot: {e}")
    # # [END - 新增的绘图代码]

    # MLP_server = MLPServer(graph)
    # for subgraph in subgraphs:
    #     MLP_server.add_client(subgraph)

    GNN_server = GNNServer(graph)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    # GNN_server2 = GNNServer(graph)
    # for subgraph in subgraphs:
    #     mend_graph = create_mend_graph(subgraph, graph, 0)
    #     GNN_server2.add_client(mend_graph)

    # GNN_server3 = GNNServer(graph)
    # for subgraph in subgraphs:
    #     mend_graph = create_mend_graph(subgraph, graph, 1)
    #     GNN_server3.add_client(mend_graph)

    # fedsage_server = FedSAGEServer(graph)
    # for subgraph in subgraphs:
    #     fedsage_server.add_client(subgraph)

    # fedpub_server = FedPubServer(graph)
    # for subgraph in subgraphs:
    #     fedpub_server.add_client(subgraph)

    # fedgcn_server = FedGCNServer(graph)
    # fedgcn_subgraphs = fedGCN_partitioning(
    #     graph,
    #     config.subgraph.num_subgraphs,
    #     method=config.subgraph.partitioning,
    #     num_hops=config.fedgcn.num_hops,
    # )
    # for subgraph in fedgcn_subgraphs:
    #     fedgcn_server.add_client(subgraph)

    results = {}

    # LOGGER.info("MLP")
    # res = MLP_server.train_local_model()
    # results[f"server MLP"] = round(res["Test Acc"], 4)

    # res = MLP_server.joint_train_g(FL=False)
    # results[f"local MLP"] = round(res["Average"]["Test Acc"], 4)

    # res = MLP_server.joint_train_g(FL=True)
    # results[f"flga MLP"] = round(res["Average"]["Test Acc"], 4)

    # LOGGER.info("GNN")
    # res = GNN_server.train_local_model(data_type="feature", fmodel_type="GNN")
    # results[f"Server F GNN"] = round(res["Test Acc"], 4)

    # res = GNN_server.train_local_model(data_type="structure")
    # results[f"Server S GNN"] = round(res["Test Acc"], 4)

    # res = GNN_server.train_local_model(data_type="f+s")
    # results[f"Server F+S GNN"] = round(res["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="feature", FL=False)
    # results[f"Local F GNN"] = round(res["Average"]["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="structure", FL=False)
    # results[f"Local S GNN"] = round(res["Average"]["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="f+s", FL=False)
    # results[f"Local F+S GNN"] = round(res["Average"]["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="feature", FL=True)
    # results[f"FL F GNN"] = round(res["Average"]["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="structure", FL=True)
    # results[f"FL S GNN"] = round(res["Average"]["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="feature", FL=True)
    # res = GNN_server.joint_train_g(data_type="structure", FL=True)
    res = GNN_server.joint_train_g(data_type="f+s", FL=True,spectral_len=0)
    results[f"FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    # res = fedsage_server.train_fedSage_plus()
    # results[f"fedsage WA"] = round(res["WA"]["Average"]["Test Acc"], 4)
    # results[f"fedsage GA"] = round(res["GA"]["Average"]["Test Acc"], 4)

    # res = fedpub_server.start()
    # results[f"fedpub"] = round(res["Average"]["Test Acc"], 4)
    # res = GNN_server3.joint_train_w(data_type="feature", fmodel_type="GNN", FL=True)
    # results[f"FedSage Ideal"] = round(res["Average"]["Test Acc"], 4)

    # res = fedgcn_server.joint_train_w()
    # results[f"fedgcn"] = round(res["Average"]["Test Acc"], 4)

    LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    set_up_system()
    # plt.show()
