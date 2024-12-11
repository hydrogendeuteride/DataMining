import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv


def get_train_dataset(file):
    with open(file, newline='', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        data = []
        next(reader)
        for row in reader:
            data.append([row[0], row[1], row[5]])

    return data


def construct_matrix(data):
    users = sorted(set(int(row[0]) for row in data))
    items = sorted(set(int(row[1]) for row in data))

    user_to_index = {user: i for i, user in enumerate(users)}
    user_to_index["UNK"] = len(users)

    item_to_index = {item: i for i, item in enumerate(items)}
    item_to_index["UNK"] = len(items)

    rating_mtx = torch.zeros((len(users) + 1, len(items) + 1), dtype=torch.float32)

    for user_id, item_id, rating in data:
        uidx = user_to_index.get(int(user_id), user_to_index["UNK"])
        iidx = item_to_index.get(int(item_id), item_to_index["UNK"])
        rating_mtx[uidx, iidx] = float(rating)

    return rating_mtx, user_to_index, item_to_index


def compute_user_bias(rating_mtx):
    user_bias = torch.zeros(rating_mtx.size(0))
    for user_idx in range(rating_mtx.size(0)):
        user_ratings = rating_mtx[user_idx, :]
        rated_items = user_ratings.nonzero(as_tuple=True)[0]
        if len(rated_items) > 0:
            user_bias[user_idx] = user_ratings[rated_items].mean().item()
    return user_bias


class CF(nn.Module):
    def __init__(self, n_users, n_items, rating_mtx):
        super(CF, self).__init__()

        self.user_similarity = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(n_users + 1, n_users + 1, dtype=torch.float32)))
        global_mean = torch.mean(rating_mtx[rating_mtx > 0])

        self.user_bias = nn.Parameter(0 * torch.ones(n_users + 1, dtype=torch.float32))
        self.item_bias = nn.Parameter(0 * torch.ones(n_items + 1, dtype=torch.float32))

        self.global_bias = nn.Parameter(global_mean.clone())

    def forward(self, user, item, rating_mtx):
        user_bias_fixed = compute_user_bias(rating_mtx)
        user_bias_fixed = user_bias_fixed.to(rating_mtx.device)

        adjusted_ratings = rating_mtx - user_bias_fixed.unsqueeze(1)

        similarities = self.user_similarity[user]
        item_ratings = adjusted_ratings[:, item]

        user_scores = torch.sum(similarities * item_ratings.t(), dim=1)

        bias = self.user_bias[user] + self.item_bias[item] + self.global_bias
        pred_ratings = user_scores + bias

        return F.sigmoid(pred_ratings) * 5.0


def data_loader(data, batch_size, user_to_index, item_to_index):
    user_indices = [user_to_index.get(int(row[0]), user_to_index.get("UNK", len(user_to_index))) for row in data]
    item_indices = [item_to_index.get(int(row[1]), item_to_index.get("UNK", len(item_to_index))) for row in data]
    ratings = [float(row[2]) for row in data]

    dataset_size = len(ratings)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    for start_idx in range(0, dataset_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield (
            torch.tensor([user_indices[i] for i in batch_indices], dtype=torch.long),
            torch.tensor([item_indices[i] for i in batch_indices], dtype=torch.long),
            torch.tensor([ratings[i] for i in batch_indices], dtype=torch.float32)
        )


def train(train_file, num_epochs=32, batch_size=256, learning_rate=0.001):
    device = torch.device('cpu')

    pdata = get_train_dataset(train_file)

    mtx, u_i, i_i = construct_matrix(pdata)
    mtx = mtx.to(device)

    model = CF(len(u_i) - 1, len(i_i) - 1, mtx).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for user_batch, item_batch, rating_batch in data_loader(pdata, batch_size, u_i, i_i):
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            rating_batch = rating_batch.to(device)

            optimizer.zero_grad()
            predicted_ratings = model(user_batch, item_batch, mtx)
            loss = criterion(predicted_ratings, rating_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * rating_batch.size(0)

        avg_epoch_loss = epoch_loss / len(pdata)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")

    return model, mtx, u_i, i_i


def process_test_data(test_file):
    with open(test_file, newline='', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        data = []
        next(reader)
        for row in reader:
            rid = int(row[0])
            user_id = int(row[1])
            item_id = float(row[2])
            data.append([rid, user_id, item_id])

    return data


def predict_ratings(model, mtx, user_to_index, item_to_index, test_file, batch_size=64):
    device = torch.device('cpu')

    test_data = process_test_data(test_file)

    model.eval()
    predictions = []

    with torch.no_grad():
        for start_idx in range(0, len(test_data), batch_size):
            batch_data = test_data[start_idx:start_idx + batch_size]
            rids = [row[0] for row in batch_data]
            user_batch = torch.tensor([user_to_index.get(row[1], user_to_index["UNK"]) for row in batch_data],
                                      dtype=torch.long).to(device)
            item_batch = torch.tensor([item_to_index.get(row[2], item_to_index["UNK"]) for row in batch_data],
                                      dtype=torch.long).to(device)

            pred_ratings = model(user_batch, item_batch, mtx).cpu().numpy()
            predictions.extend(zip(rids, pred_ratings))

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store")
    parser.add_argument("--test", dest="test", action="store")
    args = parser.parse_args()

    train_file = args.train
    test_file = args.test

    model, mtx, user_to_index, item_to_index = train(train_file)

    predictions = predict_ratings(model, mtx, user_to_index, item_to_index, test_file)

    print("Predicted Ratings:")
    for pred in predictions:
        print(pred)

    with open("submission.csv", mode='w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["RID", "rating"])

        for rid, pred in predictions:
            writer.writerow([rid, pred])
