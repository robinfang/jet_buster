import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import pdb
import sys
import os
import gc
import pickle
import threading, queue
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)


def batch2rows(batch):
    jet_id = batch[0][-1]
    result = []
    for i in batch:
        cat = to_categorical(i[0], num_classes=14, dtype=np.uint8)
        result.append(np.concatenate((cat, i[1:-1]), axis=0))
    # pdb.set_trace()
    rs = np.array(result)
    # print(len(rs))
    return pd.DataFrame([[jet_id, rs]])


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
    # particle_df["particle_category"] = particle_df["particle_category"].apply(lambda x: to_categorical(x, num_classes=len(category_list), dtype=np.uint8))
    logging.info(gc.collect())
    # print(t.head)
    # print(t.dtypes)
    # ["particle_category_{}".format(i) for i in range(14)]
    # 最多的喷注有115个粒子
    logging.info(sys.getsizeof(particle_df)/1024/1024)
    # particle_df = pd.concat([pd.DataFrame([list(x) for x in particle_df["particle_category"]], index=particle_df.index).add_prefix("g"), particle_df], axis=1).drop(["particle_category"], axis=1)
    # logging.info("particle_category dropped")
    particle_df = particle_df.values
    logging.info(gc.collect())
    # print(particle_df[:10])
    result = []
    pid = None     # previous jet id
    current_batch = None
    len_ds = len(particle_df)
    for idx, row in tqdm(enumerate(particle_df), total=len_ds):
        # pdb.set_trace()
        if pid is None or pid != row[-1]:
            if current_batch is not None:
                result.append(batch2rows(current_batch))
            # 处理上一批的
            current_batch = [row]  # 新建本批次的
        else:
            current_batch.append(row)
        pid = row[-1]
        if idx == len_ds - 1:
            result.append(batch2rows(current_batch))
            # 到达最后一个元素
    return pd.concat(result).reset_index(drop=True)


def j2e(jet_df, grouped_particle_df):
    jet_df = jet_df.sort_values(["event_id", "jet_mass", "jet_energy"]).reset_index(drop=True)
    pass


def load_ds(filename):
    return pd.read_hdf(filename)


def combine_j_p(jds, pds):
    jds = jds.sort_values(["jet_id"]).reset_index(drop=True)
    pds["label"] = jds["label"] if ("label" in jds) else 0
    pds["event_id"] = jds["event_id"]
    return pds


def batch2rows_group_p(batch):
    event_id = batch[0][-1]
    label = batch[0][-2]
    rs = pd.concat([pd.DataFrame(i[1]) for i in batch]).values
    return pd.DataFrame([[event_id, rs, label]])

def group_p_by_event(ds):
    """将combine_j_p得到的以jet为键的数据转为按event为键的数据
    
    Args:
        ds (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    assert "label" in ds
    ds = ds.sort_values(["event_id"])
    ds = ds.values
    result = []
    pid = None     # previous event id
    current_batch = None
    len_ds = len(ds)
    i = 0
    for idx, row in tqdm(enumerate(ds), total=len_ds):
        # pdb.set_trace()
        if pid is None or pid != row[-1]:
            if current_batch is not None:
                result.append(batch2rows_group_p(current_batch))
            # 处理上一批的
            current_batch = [row]  # 新建本批次的
        else:
            current_batch.append(row)
        pid = row[-1]
        if idx == len_ds - 1:
            result.append(batch2rows_group_p(current_batch))
            # 到达最后一个元素
    return pd.concat(result).reset_index(drop=True)


if __name__ == '__main__':

    test_or_train = "train"
    if os.path.exists("data/{}.h5".format(test_or_train)):
        particle_df = load_ds("data/{}.h5".format(test_or_train))
    else:
        particle_df = pd.read_csv("data/complex_{}_R04_particle.csv".format(test_or_train))
        logging.info(gc.collect())
        logging.info(sys.getsizeof(particle_df)/1024/1024)
        particle_df = p2j(particle_df)
        logging.info(sys.getsizeof(particle_df)/1024/1024)
        particle_df.to_hdf("data/{}.h5".format(test_or_train), "data")
    jet_df = pd.read_csv("data/complex_{}_R04_jet.csv".format(test_or_train))
    df = combine_j_p(jet_df, particle_df)
    del jet_df
    del particle_df
    logging.info(gc.collect())
    df = group_p_by_event(df)
    logging.info(df.shape)
    # logging.info(sys.getsizeof(df)/1024/1024)
    df.to_hdf("data/{}_grouped_by_event.h5".format(test_or_train), "data")
    # to_write = queue.Queue()

    # def writer():
    #     # Call to_write.get() until it returns None
    #     for k, df in iter(to_write.get, None):
    #         df.to_hdf("data/{}.h5".format(test_or_train), "jet_id_{}".format(k), mode="a")
    # threading.Thread(target=writer).start()

    # for item in particle_df:
    #     to_write.put(item)
    # to_write.put(None)


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
