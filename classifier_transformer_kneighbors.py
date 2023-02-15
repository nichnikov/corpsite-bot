import json
import os
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from data_prepare import train_test_datasets_prepare
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")

dir_name = "".join([date_time, "_transformer_knn"])
if not os.path.exists(os.path.join("results", dir_name)):
    os.mkdir(os.path.join("results", dir_name))

n_nbrs = 5
test_parameters = {}

# df = pd.read_csv(os.path.join("data", "support_calls.csv"), sep="\t")
df = pd.read_excel(os.path.join("data", "support_calls.xlsx"))

feature = "tag"
train_df, test_df = train_test_datasets_prepare(df, "Description", feature)

test_parameters["train_shape"] = str(train_df.shape)
test_parameters["test_shape"] = str(test_df.shape)
print(train_df.shape, test_df.shape)

test_df.rename(columns={"Description": "text"}, inplace=True)
train_df.rename(columns={"label": "y", "Description": "etalon"}, inplace=True)
test_df.rename(columns={"Description": "text"}, inplace=True)


print("unique y:\n", set(train_df["y"]))
test_parameters["unique_y"] = str(set(train_df["y"]))
train_balanced_df = pd.DataFrame({})
for y in set(train_df["y"]):
    train_y_df = train_df[train_df["y"] == y]
    train_y_df = train_y_df.sample(frac=1)
    train_balanced_df = pd.concat((train_balanced_df, train_y_df))

test_parameters["train_balanced_shape"] = str(train_balanced_df.shape)
# train_balanced_df.to_csv("train_balanced_df_ + str(feature) +  .csv", sep='\t')
print("df_train_balanced:\n", train_balanced_df)

vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')

train_queries = list(train_balanced_df["etalon"])
y = list(train_balanced_df["y"])
train_vectors = vectorizer.encode([x.lower() for x in train_queries])


test_parameters["n_neighbors"] = n_nbrs

X = train_vectors
neigh = KNeighborsClassifier(n_neighbors=n_nbrs)
neigh.fit(X, y)


test_texts = list(test_df["text"])
uniq_y = sorted(list(set(y)))

results = []
for num, t_texts in enumerate(test_texts):
    test_vector = vectorizer.encode([t_texts.lower()])
    test_results = neigh.predict_proba(test_vector)
    test_results_y_sort = sorted(zip(uniq_y, test_results[0]), key=lambda x: x[1], reverse=True)
    results.append((num, t_texts, test_results_y_sort[0][0], test_results_y_sort[0][1]))
    print(num, "/", len(test_texts))

results_df = pd.DataFrame(results, columns=["text_number", "text", "predict_id", "score"])
print(results_df)
results_with_true_df = pd.merge(results_df, test_df, on="text")

# results_df.to_csv(os.path.join("data", "test_results_transformer.csv"), sep="\t", index=False)
results_with_true_df.to_csv(os.path.join("results", dir_name, "test_results_with_true_transformer_" +
                                         str(feature) + ".csv"), sep="\t", index=False)
results_with_true_df_ = results_with_true_df[results_with_true_df["score"] >= 0.6]

y_pred = results_with_true_df_["predict_id"]
y_true = results_with_true_df_["label"]
accuracy = accuracy_score(y_true, y_pred)
test_parameters["recall"] = len(y_true) / len(results_with_true_df)
test_parameters["accuracy"] = accuracy

print(test_parameters)

with open(os.path.join("results", dir_name, "parameters.json"), "w") as j_f:
    json.dump(test_parameters, j_f)

print(len(y_true), "recall:", len(y_true) / len(results_with_true_df))
print("accuracy:", accuracy)