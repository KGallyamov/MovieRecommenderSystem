import numpy as np
import pandas as pd
from fire import Fire
from catboost import CatBoostRegressor


def evaluation(data_path='data/processed/test_df.csv', ckpt_path='models/catboost_model'):
    """
    Calculate RMSE on the given data
    :param data_path: path to a csv file in the same format as ones from data/processed
    :param ckpt_path: path to catboost regression checkpoint
    :return: None, print RMSE to stdout
    """
    data = pd.read_csv(data_path, sep='|', index_col=0)
    data, gt_rating = data.drop(['rating'], axis=1), data.rating.values

    model = CatBoostRegressor()
    model.load_model(ckpt_path)
    predictions = model.predict(data)
    print(np.sqrt(np.mean((predictions.reshape(-1) - gt_rating.reshape(-1)) ** 2)))


if __name__ == '__main__':
    Fire(evaluation)
