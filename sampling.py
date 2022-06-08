import numpy as np
import torch

def sampling(src_nodes, sample_num, neighbor_table):
    """

    :param src_nodes: 源节点
    :param sample_num: 采样指定数量的邻居节点，有放回的采样，防止节点邻居数量不足够采样
    :param neighbor_table: 节点与邻居连接表
    :return:
    """
    results = []
    for src in src_nodes:
        if src not in neighbor_table:
            # res = np.array([src])
            pass
        else:
            res = np.random.choice(neighbor_table[src], size=(sample_num,))
            # res = np.append(res, [src])
        # print(res)
            results.extend(list(res))
    return np.asarray(results).flatten()


def multi_hop_sampling(src_nodes, num_sample, table1, table2):
    """
    多级采样
    :param src_nodes: 源节点
    :param sample_num: int 采样指定数量的邻居节点，有放回的采样，防止节点邻居数量不足够采样
    :param table1: 节点与邻居连接表
    :return:
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(num_sample):
        if k % 2 == 0:
            hopk_result = sampling(sampling_result[k], hopk_num, table1)
        else:
            hopk_result = sampling(sampling_result[k], hopk_num, table2)
        sampling_result.append(hopk_result)
    return sampling_result



if __name__ == "__main__":
    from kkmusic_data import KKMuicData

    data = KKMuicData()
    (user2item, item2user), x_user, x_item, x_train_edge, x_test_edge = data.get_data()

    sampling_src_id = multi_hop_sampling([1, 2, 3], [5, 5], user2item, item2user)
    sampling_dst_id = multi_hop_sampling([1, 2, 3], [5, 5], item2user, user2item)
    print(sampling_src_id)
    sampling_src_x = []
    sampling_dst_x = []
    for i, nodes_id in enumerate(sampling_src_id):
        if i % 2 == 0:
            sampling_src_x.append(torch.from_numpy(x_user[nodes_id]).float())
            # sampling_dst_x.append(torch.from_numpy(x_item[sampling_dst_id[i]]).float())
        else:
            sampling_src_x.append(torch.from_numpy(x_item[nodes_id]).float())
            # sampling_dst_x.append(torch.from_numpy(x_user[sampling_dst_id[i]]).float())


    print(sampling_src_x)
    # print(sampling_dst_x)