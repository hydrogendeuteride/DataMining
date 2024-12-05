from collections import defaultdict

import torch
import numpy as np
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


def create_user_profile(user_data, tf_idf_matrix):
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
    norm_vec1 = math.sqrt(sum(vec1[genre]**2 for genre in vec1))
    norm_vec2 = math.sqrt(sum(vec2[genre]**2 for genre in vec2))
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


def predict_rating(user_id, item_id, user_profiles, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating):
    b_u = user_avg_ratings.get(user_id, global_avg_rating) - global_avg_rating
    b_i = item_avg_ratings.get(item_id, global_avg_rating) - global_avg_rating

    user_profile = user_profiles[user_id]
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


d = get_train_dataset("train.csv")
g = preprocess_genre(d)
tf_idf_matrix = tf_idf_rated(d, g)

user_data = group_by_user(d)
user_avg_ratings = calculate_user_avg_ratings(d)
item_avg_ratings = calculate_item_avg_ratings(d)
global_avg_rating = calculate_global_avg_rating(d)
u = create_user_profile(user_data, tf_idf_matrix)

user_id = "1"
item_id1 = "151"
item_id2 = "2115"
predicted_rating1 = predict_rating(user_id, item_id1, u, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating)
predicted_rating2 = predict_rating(user_id, item_id2, u, tf_idf_matrix, user_avg_ratings, item_avg_ratings, global_avg_rating)

print(f"Predicted Rating for User {user_id} and Item {item_id1}: {predicted_rating1}")
print(f"Predicted Rating for User {user_id} and Item {item_id2}: {predicted_rating2}")
