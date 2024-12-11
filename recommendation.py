import argparse
from collections import defaultdict
import csv
import math


def get_train_dataset(file):
    with open(file, newline='', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        data = []
        next(reader)
        for row in reader:
            data.append([row[0], row[1], row[3], row[5]])

    return data


def preprocess_genre(dataset):
    genres_set = set()

    for data in dataset:
        genres = data[2].split('|')
        genres_set.update(genres)

    return genres_set


def tf_idf_rated(dataset, genres_set):
    genre_count = {genre: 0 for genre in genres_set}
    for data in dataset:
        genres = set(data[2].split('|'))
        for genre in genres:
            genre_count[genre] += 1

    N = len(dataset)

    idf = {genre: math.log(N / (1 + genre_count[genre])) for genre in genres_set}

    tf_idf_matrix = []
    for data in dataset:
        item_id = data[1]
        item_genres = data[2].split('|')
        rating = float(data[3])

        total_genres = len(item_genres)

        tf = {genre: (item_genres.count(genre) / total_genres) * rating for genre in genres_set}

        tf_idf_vector = {genre: tf[genre] * idf[genre] for genre in genres_set}
        tf_idf_matrix.append((item_id, tf_idf_vector))

    return tf_idf_matrix


def group_by_user(dataset):
    user_data = defaultdict(list)
    for data in dataset:
        user_id = data[0]
        item_id = data[1]
        genres = data[2]
        rating = float(data[3])
        user_data[user_id].append((item_id, genres, rating))
    return user_data


def create_user_profile(user_data, tf_idf_matrix, user_avg_ratings):
    user_profiles = {}
    for user_id, items in user_data.items():
        user_profile = {genre: 0 for genre in tf_idf_matrix[0][1].keys()}
        total_rating = 0

        for item_id, _, rating in items:
            centered_rating = rating - user_avg_ratings[user_id]
            item_vector = None

            for tf in tf_idf_matrix:
                if tf[0] == item_id:
                    item_vector = tf[1]
                    break

            if item_vector is not None:
                total_rating += abs(centered_rating)
                for genre, value in item_vector.items():
                    user_profile[genre] += value * centered_rating

        for genre in user_profile:
            if total_rating > 0:
                user_profile[genre] /= total_rating

        user_profiles[user_id] = user_profile
    return user_profiles


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[genre] * vec2[genre] for genre in vec1)
    norm_vec1 = math.sqrt(sum(vec1[genre] ** 2 for genre in vec1))
    norm_vec2 = math.sqrt(sum(vec2[genre] ** 2 for genre in vec2))
    return dot_product / (norm_vec1 * norm_vec2 + 1e-10)


def calculate_item_avg_ratings(dataset):
    item_ratings = defaultdict(list)
    for data in dataset:
        item_id = data[1]
        rating = float(data[3])
        item_ratings[item_id].append(rating)
    item_avg_ratings = {item_id: sum(ratings) / len(ratings) for item_id, ratings in item_ratings.items()}
    return item_avg_ratings


def calculate_global_avg_rating(dataset):
    total_rating = sum(float(data[3]) for data in dataset)
    global_avg_rating = total_rating / len(dataset)
    return global_avg_rating


def calculate_user_avg_ratings(dataset):
    user_ratings = defaultdict(list)
    for data in dataset:
        user_id = data[0]
        rating = float(data[3])
        user_ratings[user_id].append(rating)

    user_avg_rating = {user_id: sum(ratings) / len(ratings) for user_id, ratings in user_ratings.items()}
    return user_avg_rating


def predict_rating(user_id, item_id, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings,
                   global_avg_rating):
    b_u = user_avg_ratings.get(user_id, global_avg_rating) - global_avg_rating
    b_i = item_avg_ratings.get(item_id, global_avg_rating) - global_avg_rating

    user_profile = user_profiles.get(user_id, None)
    if user_profile is None:
        return global_avg_rating + b_u + b_i

    item_vector = None
    for tf in tf_idf_matrix:
        if tf[0] == item_id:
            item_vector = tf[1]
            break

    if item_vector is None:
        return global_avg_rating + b_u + b_i

    similarity = cosine_similarity(user_profile, item_vector)
    alpha = 1.0
    predicted_rating = global_avg_rating + b_u + b_i + similarity * alpha

    predicted_rating = min(max(predicted_rating, 1.0), 5.0)
    return predicted_rating


def train(file):
    dataset = get_train_dataset(file)

    genres_set = preprocess_genre(dataset)
    tf_idf_matrix = tf_idf_rated(dataset, genres_set)

    user_data = group_by_user(dataset)
    user_avg_ratings = calculate_user_avg_ratings(dataset)
    item_avg_ratings = calculate_item_avg_ratings(dataset)
    global_avg_rating = calculate_global_avg_rating(dataset)

    user_profiles = create_user_profile(user_data, tf_idf_matrix, user_avg_ratings)

    return user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating


def calculate_rmse(predictions_file, ground_truth_file):
    with open(predictions_file, mode="r", encoding="utf-8") as pred_file:
        reader = csv.reader(pred_file)
        next(reader)
        predictions = {int(row[0]): float(row[1]) for row in reader}

    with open(ground_truth_file, mode="r", encoding="utf-8") as truth_file:
        reader = csv.reader(truth_file)
        next(reader)
        ground_truth = {int(row[0]): float(row[1]) for row in reader}

    squared_errors = []
    absolute_errors = []

    for rid, true_rating in ground_truth.items():
        predicted_rating = predictions.get(rid, None)
        if predicted_rating is not None:
            squared_errors.append((predicted_rating - true_rating) ** 2)
            absolute_errors.append(abs(predicted_rating - true_rating))

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    return rmse


def predict(test_file, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating,
            batch_size=64):
    def process_test_data(file):
        with open(file, newline='', encoding='utf-8') as csvFile:
            reader = csv.reader(csvFile)
            data = []
            next(reader)
            for row in reader:
                rid = int(row[0])
                user_id = row[1]
                item_id = row[2]
                data.append((rid, user_id, item_id))
        return data

    def create_test_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    test_data = process_test_data(test_file)
    all_predictions = []

    for batch in create_test_batches(test_data, batch_size):
        for rid, user_id, item_id in batch:
            pred_rating = predict_rating(
                user_id, item_id, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating
            )
            all_predictions.append((rid, pred_rating))

    return all_predictions


def main(args):
    train_file = args.train
    user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating = train(train_file)

    test_file = args.test
    predictions = predict(test_file, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating)

    with open("submissions.csv", mode="w", newline="", encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["RID", "rating"])
        writer.writerows(predictions)

    # predictions_file = "predictions.csv"

    # ground_truth_file = "submissions1.csv"
    # rmse = calculate_rmse(predictions_file, ground_truth_file)
    #
    # print(f"RMSE: {rmse:.4f}")

    for rid, pred in predictions:
        print(f"RID: {rid}, Predicted Rating: {pred:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store")
    parser.add_argument("--test", dest="test", action="store")
    args = parser.parse_args()

    main(args)
