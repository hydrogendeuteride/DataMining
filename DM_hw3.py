import latent_factor
from recommendation import *
from latent_factor import *

import csv
import torch


def calculate_rmse(predictions, true_ratings):
    squared_errors = [(pred - true) ** 2 for pred, true in zip(predictions, true_ratings)]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return rmse


def validation(args):
    train_file = args.train
    user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating = train(train_file)

    pdata = latent_factor.get_train_dataset(train_file)
    train_data, val_data = train_test_split(pdata, test_size=0.1, random_state=42)
    model, user_to_index, item_to_index = train_model(train_data)

    ##################################################################################################

    val_users = [row[0] for row in val_data]
    val_items = [row[1] for row in val_data]
    val_ratings = [float(row[2]) for row in val_data]

    tfidf_predictions = []
    lfm_predictions = []

    for user, item, true_rating in zip(val_users, val_items, val_ratings):
        tfidf_score = predict_rating(user, item, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings,
                                     global_avg_rating)
        tfidf_predictions.append((tfidf_score))

        # LFM prediction
        user_idx = user_to_index.get(user, user_to_index["UNK"])
        item_idx = item_to_index.get(item, item_to_index["UNK"])
        model.eval()
        with torch.no_grad():
            lfm_score = model(torch.tensor([user_idx]), torch.tensor([item_idx])).item()
        lfm_predictions.append((lfm_score))

    tfidf_rmse= calculate_rmse(tfidf_predictions, val_ratings)
    lfm_rmse = calculate_rmse(lfm_predictions, val_ratings)

    combined_predictions = [(tfidf + lfm) / 2.0 for tfidf, lfm in zip(tfidf_predictions, lfm_predictions)]
    combined_rmse = calculate_rmse(combined_predictions, val_ratings)

    print("Validation Results:")
    print(f"TF-IDF Model: RMSE = {tfidf_rmse:.4f}")
    print(f"LFM Model: RMSE = {lfm_rmse:.4f}")
    print(f"Combined Model: RMSE = {combined_rmse:.4f}")

    ##################################################################################################

    test_file = args.test
    tfidf_predictions = predict(test_file, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings,
                                global_avg_rating)
    lfm_predictions = test(model, user_to_index, item_to_index, test_file)

    print("RID, TF-IDF Score, LFM Score")
    tfidf_dict = {rid: score for rid, score in tfidf_predictions}
    for rid, lfm_score in lfm_predictions:
        tfidf_score = tfidf_dict.get(rid, None)
        if tfidf_score is not None:
            print(f"{rid}, {tfidf_score:.2f}, {lfm_score:.2f}")


def main(args):
    train_file = args.train
    user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating = train(train_file)
    pdata = latent_factor.get_train_dataset(train_file)
    model, user_to_index, item_to_index = train_model(pdata)

    test_file = args.test
    tfidf_predictions = predict(test_file, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings,
                                global_avg_rating)
    lfm_predictions = test(model, user_to_index, item_to_index, test_file)

    print("RID, TF-IDF Score, LFM Score")
    tfidf_dict = {rid: score for rid, score in tfidf_predictions}
    for rid, lfm_score in lfm_predictions:
        tfidf_score = tfidf_dict.get(rid, None)
        if tfidf_score is not None:
            print(f"{rid}, {tfidf_score:.2f}, {lfm_score:.2f}")

    tfidf_dict = {rid: score for rid, score in tfidf_predictions}
    lfm_dict = {rid: score for rid, score in lfm_predictions}

    combined_predictions = []
    for rid in tfidf_dict:
        if rid in lfm_dict:
            tfidf_score = tfidf_dict[rid]
            lfm_score = lfm_dict[rid]
            combined_score = (tfidf_score + lfm_score) / 2.0
            combined_predictions.append((rid, combined_score))

    with open("submission.csv", mode="w", newline="", encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["RID", "rating"])
        writer.writerows(combined_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store")
    parser.add_argument("--test", dest="test", action="store")
    args = parser.parse_args()

    main(args)
