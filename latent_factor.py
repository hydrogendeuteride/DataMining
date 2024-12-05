import torch
import torch.nn as nn
import numpy as np
import csv
from collections import defaultdict


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

    return processed_data, len(user_to_index), len(item_to_index)


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

    def forward(self, user_ids, item_ids):
        global_bias = self.mu
        user_bias = self.user_bias[user_ids]
        item_bias = self.item_bias[item_ids]

        user_factors = self.P[user_ids]
        item_factors = self.Q[item_ids]

        pred_rating = global_bias + user_bias + item_bias + torch.sum(item_factors * user_factors,
                                                                      dim=1)
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


file = 'train.csv'
pdata = get_train_dataset(file)

processed_data, num_users, num_items = preprocess_data(pdata)

batch_size = 128

n_rank = 50

global_mean, user_means, item_means = calculate_biases(processed_data)

model = LFM(num_users, num_items, n_rank)

model.mu.data.fill_(global_mean)
for user, bias in user_means.items():
    model.user_bias.data[user] = bias

for item, bias in item_means.items():
    model.item_bias.data[item] = bias

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 128

lambda_reg = 0.000005

for epoch in range(epochs):
    batches = create_batches(processed_data, batch_size)

    epoch_loss = 0.0
    for user_ids, item_ids, ratings in batches:
        optimizer.zero_grad()

        pred_ratings = model(user_ids, item_ids)
        pred_ratings = torch.clamp(pred_ratings, min=1.0, max=5.0)

        pred_loss = criterion(pred_ratings, ratings)
        reg_loss = lambda_reg * (
                torch.norm(model.P, p=2) ** 2 + torch.norm(model.Q, p=2) ** 2 +
                torch.norm(model.user_bias, p=2) ** 2 + torch.norm(model.item_bias, p=2) ** 2
        )
        loss = pred_loss + reg_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_size

    avg_loss = np.sqrt(epoch_loss / len(pdata))
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

user_batch = torch.tensor([0, 1, 2])
item_batch = torch.tensor([2, 4, 1])

model.eval()
with torch.no_grad():
    predictions = model(user_batch, item_batch)

print("Predicted Ratings:")
print(predictions.numpy())
