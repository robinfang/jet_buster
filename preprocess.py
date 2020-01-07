import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import pdb
import sys
import gc
import pickle
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

def batch2rows(batch):
    jet_id = batch[0][-1]
    # pdb.set_trace()
    rs = np.array([i[0:-1] for i in batch])
    # print(len(rs))
    return np.array([jet_id, rs])


def p2j(particle_df):
    particle_df = particle_df.sort_values(
        ["jet_id", "particle_mass", "particle_energy"]).reset_index(drop=True).astype(np.float32, errors="ignore")
    unique_jet_count = particle_df["jet_id"].nunique()
    logging.info(particle_df.shape)
    p_count = particle_df.shape[0]
    logging.info(unique_jet_count)
    logging.info(particle_df.dtypes)
    logging.info(gc.collect())
    len_cate = len(particle_df["particle_category"].unique())
    # print(len_cate)
    category_list = [-2212, -2112, -321, -211, -13, -11, 11, 13, 22, 130, 211, 321, 2112, 2212]
    assert len_cate == len(category_list)
    particle_df["particle_category"] = particle_df["particle_category"].apply(category_list.index)
    particle_df["particle_category"] = particle_df["particle_category"].apply(lambda x: to_categorical(x, num_classes=len(category_list), dtype=bool))
    logging.info(gc.collect())
    # print(t.head)
    # print(t.dtypes)
    # ["particle_category_{}".format(i) for i in range(14)]
    # 最多的喷注有115个粒子
    logging.info(sys.getsizeof(particle_df)/1024/1024)
    particle_df = pd.concat([pd.DataFrame([list(x) for x in particle_df["particle_category"]], index=particle_df.index).add_prefix("g"), particle_df], axis=1)
    # particle_df.drop(["particle_category"], inplace=True, axis=1)
    # logging.info("particle_category dropped")
    particle_df = particle_df.values
    logging.info(gc.collect())
    # print(particle_df[:10])
    result = []
    pid = None
    current_batch = None
    len_ds = len(particle_df)
    for idx, row in enumerate(particle_df):
        # pdb.set_trace()
        if pid is None or pid != row[-1]:
            if current_batch is not None:
                yield pd.DataFrame(batch2rows(current_batch))
            # 处理上一批的
            current_batch = [row]  # 新建本批次的
        else:
            current_batch.append(row)
        pid = row[-1]
        if idx == len_ds - 1:
            yield pd.DataFrame(batch2rows(current_batch))
            # 到达最后一个元素


def j2e(jet_df, grouped_particle_df):
    jet_df = jet_df.sort_values(["event_id", "jet_mass", "jet_energy"]).reset_index(drop=True)
    
    pass


if __name__ == '__main__':

    test_or_train = "train"
    particle_df = pd.read_csv("data/complex_{}_R04_particle.csv".format(test_or_train))
    logging.info(gc.collect())
    logging.info(sys.getsizeof(particle_df)/1024/1024)
    particle_df = p2j(particle_df)
    logging.info(sys.getsizeof(particle_df)/1024/1024)
    store = pd.HDFStore("data/{}.h5".format(test_or_train))
    for item in particle_df:
        store.append("data", item)
    store.close()
    # with open("e:/data/{}_p.pickle".format(test_or_train), "wb") as f:
    #     pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    # df.to_csv("e:/data/{}_particle_group_by_jet.csv".format(test_or_train), index=False)

    # df2 = pd.read_csv("d:/pyworkspace/jet_buster/data/complex_test_R04_jet.csv")
    # print(sys.getsizeof(df2)/1024/1024)
    # df3 = pd.merge(df, df2, how="inner", on="jet_id")
    # print(sys.getsizeof(df3)/1024/1024)
    # df3.to_csv("d:/pyworkspace/jet_buster/data/test_jet_combined.csv")

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
