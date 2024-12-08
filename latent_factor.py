import torch
import torch.nn as nn
import numpy as np
import csv
from collections import defaultdict
import argparse


def get_train_dataset(file):
    with open(file, newline='', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        data = []
        next(reader)
        for row in reader:
            user_id = int(row[0])
            item_id = int(row[1])
            rating = float(row[5])
            data.append([user_id, item_id, rating])

    return data


def preprocess_data(data):
    user_ids = list(set(row[0] for row in data))
    item_ids = list(set(row[1] for row in data))

    user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
    item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

    processed_data = [
        [user_to_index[row[0]], item_to_index[row[1]], row[2]] for row in data
    ]

    return processed_data, user_to_index, item_to_index


def calculate_biases(data):
    global_mean = np.mean([row[2] for row in data])

    user_ratings = defaultdict(list)
    for row in data:
        user_ratings[row[0]].append(row[2])
    user_means = {user: np.mean(ratings) - global_mean for user, ratings in user_ratings.items()}

    item_ratings = defaultdict(list)
    for row in data:
        item_ratings[row[1]].append(row[2])
    item_means = {item: np.mean(ratings) - global_mean for item, ratings in item_ratings.items()}

    return global_mean, user_means, item_means


class LFM(nn.Module):
    def __init__(self, n_users, n_items, n_rank):
        super(LFM, self).__init__()
        self.P = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_users, n_rank)))
        self.Q = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_items, n_rank)))

        self.mu = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Parameter(torch.zeros(n_users))
        self.item_bias = nn.Parameter(torch.zeros(n_items))

        self.hidden = nn.Linear(n_rank, 1)
        self.act = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, user_ids, item_ids):
        global_bias = self.mu
        user_bias = self.user_bias[user_ids]
        item_bias = self.item_bias[item_ids]

        user_factors = self.P[user_ids]
        item_factors = self.Q[item_ids]

        interaction = user_factors * item_factors
        interaction = self.act(interaction)
        interaction = self.hidden(interaction).squeeze()

        pred_rating_raw = global_bias + user_bias + item_bias + interaction
        pred_rating = torch.clamp(pred_rating_raw, min=1.0, max=5.0)

        return pred_rating


def create_batches(data, batch_size):
    data = torch.tensor(data, dtype=torch.float)
    shuffled_data = data[torch.randperm(len(data))]

    batches = []
    for i in range(0, len(shuffled_data), batch_size):
        batch = shuffled_data[i:i + batch_size]
        user_ids = batch[:, 0].long()
        item_ids = batch[:, 1].long()
        ratings = batch[:, 2]
        batches.append((user_ids, item_ids, ratings))

    return batches


def train_model(train_file):
    pdata = get_train_dataset(train_file)
    processed_data, user_to_index, item_to_index = preprocess_data(pdata)
    num_users = len(user_to_index)
    num_items = len(item_to_index)
    batch_size = 256
    n_rank = 64

    global_mean, user_means, item_means = calculate_biases(processed_data)

    model = LFM(num_users, num_items, n_rank)

    model.mu.data.fill_(global_mean)
    for user, bias in user_means.items():
        if user + 1 in user_to_index:
            index = user_to_index[user + 1]
            model.user_bias.data[index] = bias

    for item, bias in item_means.items():
        if item + 1 in item_to_index:
            index = item_to_index[item + 1]
            model.item_bias.data[index] = bias

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    epochs = 256

    for epoch in range(epochs):
        batches = create_batches(processed_data, batch_size)
        epoch_loss = 0.0
        for user_ids, item_ids, ratings in batches:
            optimizer.zero_grad()
            pred_ratings = model(user_ids, item_ids)
            pred_ratings = torch.clamp(pred_ratings, min=1.0, max=5.0)

            pred_loss = criterion(pred_ratings, ratings)
            loss = pred_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size

        avg_loss = np.sqrt(epoch_loss / len(pdata))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training complete")
    return model, user_to_index, item_to_index


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


def create_test_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def test(model, user_to_index, item_to_index, test_file, batch_size=64):
    tdata = process_test_data(test_file)
    model.eval()

    all_predictions = []

    for batch in create_test_batches(tdata, batch_size):
        rids = [row[0] for row in batch]
        user_ids = [user_to_index.get(row[1], -1) for row in batch]
        item_ids = [item_to_index.get(row[2], -1) for row in batch]

        rids = torch.tensor(rids)
        user_ids = torch.tensor(user_ids)
        item_ids = torch.tensor(item_ids)

        with torch.no_grad():
            predictions = model(user_ids, item_ids)

        all_predictions.extend(zip(rids.cpu().numpy(), predictions.cpu().numpy()))

    for rid, pred in all_predictions:
        print(f"RID: {rid}, Predicted Rating: {pred:.2f}")

    return all_predictions


def test_eval(preds, valid_file):
    actual_ratings = {}
    with open(valid_file, newline='', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)  # 헤더 스킵
        for row in reader:
            rid = int(row[0])
            rating = float(row[1])
            actual_ratings[rid] = rating

    errors = []
    for rid, pred_rating in preds:
        if rid in actual_ratings:
            actual_rating = actual_ratings[rid]
            errors.append((pred_rating - actual_rating) ** 2)

    if errors:
        rmse = np.sqrt(np.mean(errors))
        print(f"RMSE: {rmse:.4f}")
        return rmse
    else:
        print("No matching RIDs found between predictions and actual ratings.")
        return None


def save_output(preds):
    with open("submissions.csv", mode='w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["RID", "rating"])

        for rid, pred in preds:
            writer.writerow([rid, pred])


def main(args):
    model, user_to_index, item_to_index = train_model(args.train)

    preds = test(model, user_to_index, item_to_index, args.test)

    # test_eval(preds, "submission.csv")

    save_output(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store")
    parser.add_argument("--test", dest="test", action="store")
    args = parser.parse_args()

    main(args)
