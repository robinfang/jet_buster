import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
# import pdb
import sys


def batch2rows(batch):
    jet_id = batch[0][-1]
    rs = np.concatenate([i[0:-1] for i in batch])
    # print(len(rs))
    return np.concatenate(([jet_id], np.pad(rs, (0, 115 * 19 - len(rs)), 'constant', constant_values=0)))



def p2j(particle_df):
    particle_df = particle_df.sort_values(
        ["jet_id", "particle_mass","particle_energy"]).reset_index(drop=True)

    len_cate = len(particle_df["particle_category"].unique())
    print(len_cate)
    category_list = [-2212, -2112, -321, -211, -13, -11, 11, 13, 22, 130, 211, 321, 2112, 2212]
    assert len_cate == len(category_list)
    particle_df["particle_category"] = particle_df["particle_category"].apply(category_list.index)
    particle_df["particle_category"] = particle_df["particle_category"].apply(lambda x: to_categorical(x, num_classes=len(category_list), dtype='bool'))
    t = pd.DataFrame([list(x) for x in particle_df["particle_category"]], index=particle_df.index).add_prefix("g")
    # print(t.head)
    # print(t.dtypes)
    # ["particle_category_{}".format(i) for i in range(14)]
    # 最多的喷注有115个粒子
    particle_df.drop(["particle_category"], inplace=True, axis=1)
    particle_df = pd.concat([t, particle_df.astype(np.float32, errors="ignore")], axis=1)
    particle_df = particle_df.values
    print(particle_df[:10])
    result_ds = []
    pid = None
    current_batch = None
    len_ds = len(particle_df)
    for idx, row in enumerate(particle_df):
        # pdb.set_trace()
        if pid is None or pid != row[-1]:
            if current_batch is not None:
                result_ds.append(batch2rows(current_batch))
            # 处理上一批的
            current_batch = [row]  # 新建本批次的
        else:
            current_batch.append(row)
        pid = row[-1]
        if idx == len_ds - 1:
            result_ds.append(batch2rows(current_batch))
            # 到达最后一个元素
    result_ds = np.array(result_ds)
    print(result_ds.shape)
    return result_ds



def j2e(jet_df, grouped_particle_df):
    jet_df = jet_df.sort_values(["event_id", "jet_mass", "jet_energy"]).reset_index(drop=True)

    pass


if __name__ == '__main__':
    df = pd.read_csv("e:/data/complex_test_R04_particle.csv")
    df = p2j(df)
    print(sys.getsizeof(df))

# import featuretools as ft


# def feature_dfs(df):
#     """
#     featuretools自动生成特征
#     """
#     es = ft.EntitySet()
#     es = es.entity_from_dataframe(entity_id='jets', dataframe=df, index='jet_id')
#     es = es.normalize_entity(base_entity_id='jets', new_entity_id='events', index='event_id')
#     feature_matrix, _ = ft.dfs(entityset=es, target_entity='jets')
#     return feature_matrix
