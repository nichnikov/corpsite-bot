# https://stackoverflow.com/questions/839994/extracting-a-url-in-python
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def data_labled(df: pd.DataFrame, data_column: str, label_column: str):
    """тегирование данных для обучения"""
    df.fillna("нет_значения", inplace=True)
    labels, data = df[label_column], df[data_column]
    unique_labels_tg = [(tg_tx, num) for num, tg_tx in enumerate(sorted([str(x) for x in set(labels)]))]
    unique_labels_tg_df = pd.DataFrame(unique_labels_tg, columns=[label_column, "label"])
    data_lb_df = pd.merge(df[[data_column, label_column]], unique_labels_tg_df, on=label_column)
    return data_lb_df


def pattern_changed(text: str, patterns: [()]):
    """заменяет паттерн из списка на маску
    (список кортежей: (паттерн, маска))"""

    def pattern_finding(myString: str, pattern: str):
        """выбирает url из текстовой строки"""
        return re.findall(pattern, myString)

    for ptrn, msk in patterns:
        found_urls = pattern_finding(text, ptrn)
        if found_urls:
            for found_url in found_urls:
                text = text.replace(found_url, msk)
    return text


def column_text_changed(df: pd.DataFrame, column: str, patterns: []):
    """изменяет текст в указанном столбце датафрейма"""
    df[column] = df[column].apply(lambda x: pattern_changed(x, patterns))
    return df

def train_test_datasets_prepare(df: pd.DataFrame, data_column: str, label_column: str):
    """"""
    patterns = [("(?P<url>https?://[^\s]+)", " http_url "), (r"\d+", " number_mask "),
                (r"[\w.+-]+@[\w-]+\.[\w.-]+", " email "), (r"(?P<url>\\[^\s]+)", " url_internal ")]

    data_lb_df = data_labled(df, data_column, label_column)
    data_lb_df = column_text_changed(data_lb_df, data_column, patterns)
    train_datasets = []
    test_datasets = []
    for lb in set(data_lb_df["label"]):
        if data_lb_df[data_lb_df["label"] == lb].shape[0] > 1:
            train_df, test_df = train_test_split(data_lb_df[data_lb_df["label"] == lb], test_size=0.2)
            train_datasets.append(train_df)
            test_datasets.append(test_df)
    train_df_all = pd.concat(train_datasets)
    test_df_all = pd.concat(test_datasets)
    return train_df_all, test_df_all


df = pd.read_csv(os.path.join("data", "support_calls.csv"), sep="\t")
train_df, test_df = train_test_datasets_prepare(df, "Description", "ProductName")
train_df.to_csv(os.path.join("data", "train_df.csv"), sep="\t", index=False)
test_df.to_csv(os.path.join("data", "test_df.csv"), sep="\t", index=False)
