import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from data_prepare import train_test_datasets_prepare
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from texts_processing import TextsTokenizer, TokensVectorsTfIdf

df = pd.read_csv(os.path.join("data", "support_calls.csv"), sep="\t")

feature = "ReasonName"
train_df, test_df = train_test_datasets_prepare(df, "Description", feature)

print(train_df.shape, test_df.shape)

test_df.rename(columns={"Description": "text"}, inplace=True)
train_df.rename(columns={"label": "y", "Description": "etalon"}, inplace=True)
test_df.rename(columns={"Description": "text"}, inplace=True)


print("unique y:\n", set(train_df["y"]))
train_balanced_df = pd.DataFrame({})
for y in set(train_df["y"]):
    train_y_df = train_df[train_df["y"] == y]
    train_y_df = train_y_df.sample(frac=1)
    train_balanced_df = pd.concat((train_balanced_df, train_y_df))

# train_balanced_df.to_csv("train_balanced_df_ + str(feature) +  .csv", sep='\t')
print("df_train_balanced:\n", train_balanced_df)

# vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')
tokenizer = TextsTokenizer()
vectorizer = TokensVectorsTfIdf(15000)

train_queries = list(train_balanced_df["etalon"])
y = list(train_balanced_df["y"])
tokens = tokenizer(train_queries)
train_vectors = vectorizer(tokens)

train_matrix = hstack(train_vectors).T
# X = np.concatenate([v.toarray() for v in train_vectors])
X = train_matrix.toarray()
print(X.shape)
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X, y)


test_texts = list(test_df["text"])
uniq_y = sorted(list(set(y)))

results = []
test_tokens = tokenizer(test_texts)
num = 1
for t_text, t_tokens in zip(test_texts, test_tokens):
    test_vectors = vectorizer([t_tokens])
    test_matrix = hstack(test_vectors).T
    test_results = neigh.predict_proba(test_matrix.toarray())
    test_results_y_sort = sorted(zip(uniq_y, test_results[0]), key=lambda x: x[1], reverse=True)
    results.append((t_text, t_tokens, test_results_y_sort[0][0], test_results_y_sort[0][1]))
    print(num, "/", len(test_texts))
    num += 1

results_df = pd.DataFrame(results, columns=["text", "tokens", "predict_id", "score"])
print(results_df)
results_with_true_df = pd.merge(results_df, test_df, on="text")

# results_df.to_csv(os.path.join("data", "test_results_transformer.csv"), sep="\t", index=False)
results_with_true_df.to_csv(os.path.join("data", "test_results_with_true_tfidf_" +
                                         str(feature) + ".csv"), sep="\t", index=False)
results_with_true_df_ = results_with_true_df[results_with_true_df["score"] >= 0.6]

y_pred = results_with_true_df_["predict_id"]
y_true = results_with_true_df_["label"]
accuracy = accuracy_score(y_true, y_pred)
print(len(y_true), "recall:", len(y_true) / len(results_with_true_df))
print("accuracy:", accuracy)
