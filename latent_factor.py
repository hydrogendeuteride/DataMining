import torch
import torch.nn as nn
import numpy as np
import csv
from collections import defaultdict
import argparse
from sklearn.model_selection import train_test_split


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
    user_to_index["UNK"] = len(user_ids)
    item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
    item_to_index["UNK"] = len(item_ids)

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
        self.P = nn.Parameter(nn.init.kaiming_normal_(torch.empty(n_users + 1, n_rank, dtype=torch.float32)))
        self.Q = nn.Parameter(nn.init.kaiming_normal_(torch.empty(n_items + 1, n_rank, dtype=torch.float32)))

        self.mu = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.user_bias = nn.Parameter(torch.zeros(n_users + 1, dtype=torch.float32))
        self.item_bias = nn.Parameter(torch.zeros(n_items + 1, dtype=torch.float32))

        self.act = nn.SELU()

    def forward(self, user_ids, item_ids):
        global_bias = self.mu
        user_bias = self.user_bias[user_ids]
        item_bias = self.item_bias[item_ids]

        user_factors = self.P[user_ids]
        item_factors = self.Q[item_ids]

        interaction = torch.sum(self.act(user_factors * item_factors), dim=1)

        pred_rating = global_bias + user_bias + item_bias + interaction

        return pred_rating


def create_batches(data, batch_size):
    data = torch.tensor(data, dtype=torch.float32)
    shuffled_data = data[torch.randperm(len(data))]

    batches = []
    for i in range(0, len(shuffled_data), batch_size):
        batch = shuffled_data[i:i + batch_size]
        user_ids = batch[:, 0].long()
        item_ids = batch[:, 1].long()
        ratings = batch[:, 2]
        batches.append((user_ids, item_ids, ratings))

    return batches


def train_model(pdata):
    train_data, val_data = train_test_split(pdata, test_size=0.1, random_state=42)

    train_processed, user_to_index, item_to_index = preprocess_data(train_data)
    val_processed, _, _ = preprocess_data(val_data)

    num_users = len(user_to_index) - 1
    num_items = len(item_to_index) - 1
    batch_size = 256
    n_rank = 50

    global_mean, user_means, item_means = calculate_biases(train_processed)

    model = LFM(num_users, num_items, n_rank)

    model.mu.data.fill_(global_mean)
    for user, bias in user_means.items():
        model.user_bias.data[user] = bias
    for item, bias in item_means.items():
        model.item_bias.data[item] = bias

    model.user_bias.data[-1] = 0
    model.item_bias.data[-1] = 0

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 256

    for epoch in range(epochs):
        model.train()
        train_batches = create_batches(train_processed, batch_size)
        epoch_train_loss = 0.0

        for user_ids, item_ids, ratings in train_batches:
            optimizer.zero_grad()
            pred_ratings = model(user_ids, item_ids)
            train_loss = criterion(pred_ratings, ratings)
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            epoch_train_loss += train_loss.item() * len(user_ids)

        avg_train_loss = np.sqrt(epoch_train_loss / len(train_data))

        model.eval()
        val_batches = create_batches(val_processed, batch_size)
        epoch_val_loss = 0.0

        with torch.no_grad():
            for user_ids, item_ids, ratings in val_batches:
                pred_ratings = torch.clamp(model(user_ids, item_ids), 1.0, 5.0)
                val_loss = criterion(pred_ratings, ratings)
                epoch_val_loss += val_loss.item() * len(user_ids)

        avg_val_loss = np.sqrt(epoch_val_loss / len(val_data))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
        user_ids = [user_to_index.get(row[1], user_to_index["UNK"]) for row in batch]
        item_ids = [item_to_index.get(row[2], user_to_index["UNK"]) for row in batch]

        rids = torch.tensor(rids)
        user_ids = torch.tensor(user_ids)
        item_ids = torch.tensor(item_ids)

        with torch.no_grad():
            predictions = torch.clamp(model(user_ids, item_ids), 1.0, 5.0)

        all_predictions.extend(zip(rids.cpu().numpy(), predictions.cpu().numpy()))

    for rid, pred in all_predictions:
        print(f"RID: {rid}, Predicted Rating: {pred:.2f}")

    return all_predictions


def save_output(preds):
    with open("submissions.csv", mode='w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["RID", "rating"])

        for rid, pred in preds:
            writer.writerow([rid, pred])


def main(args):
    pdata = get_train_dataset(args.train)
    model, user_to_index, item_to_index = train_model(pdata)

    preds = test(model, user_to_index, item_to_index, args.test)

    save_output(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store")
    parser.add_argument("--test", dest="test", action="store")
    args = parser.parse_args()

    main(args)
