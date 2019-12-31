import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from keras.utils.np_utils import to_categorical


def lbz(y):
    return label_binarize(y, classes=[0, 1, 2, 3])


def p2j(particle_df):
    particle_df = particle_df.sort_values(
        ["jet_id", "particle_energy"]).reset_index(drop=True)

    len_cate = len(particle_df["particle_category"].unique())
    print(len_cate)
    category_list = [-2212, -2112, -321, -211, -13, -11, 11, 13, 22, 130, 211, 321, 2112, 2212]
    assert len_cate == len(category_list)
    particle_df["particle_category"] = particle_df["particle_category"].apply(category_list.index)
    # particle_df["particle_category"] = particle_df["particle_category"].apply(lambda x: to_categorical(x, num_classes=len(category_list), dtype='int'))
    print(particle_df.dtypes)
    print(particle_df["particle_category"])
    pass


if __name__ == '__main__':
    df = pd.read_csv("data/complex_test_R04_particle.csv")
    p2j(df)


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
