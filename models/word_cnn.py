import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvModel(nn.Module):
    """ cnn model  """
    def __init__(self, max_len, num_emotions, vocab_size,
            filter_sizes, num_filters, embedding_size,
            dropout_prob=None, embedding_matrix=None):

        super(ConvModel, self).__init__()

        self.max_len = max_len
        self.num_emotions = num_emotions
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes
        self.num_filters  = num_filters

        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob

        # self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList([nn.Conv1d(self.embedding_size, self.num_filters,
            filter_size) for filter_size in filter_sizes])
        # self.conv2 = nn.Conv1d(self.num_filters, )

        self.dropout = nn.Dropout(self.dropout_prob)

        self.fc = nn.Linear(num_filters * len(filter_sizes), num_emotions)

        # self.inv_fc = nn.Linear(num_emotions, num_filters * len(filter_sizes))
        # self.deconv = nn.ConvTranspose1d(self.num_filters, 1, filter_sizes[0])

    def load_model_from_ny(self, weights_file):

        trained_weights = np.load(weights_file, encoding='latin1').item()

        # pretrained_embedding = np.pad(trained_weights['embedding']['weights'], ((0,1),(0,0)), 'constant')
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # print("LOAD CNN MODEL WITH DIMENTIO")
        for i, conv in enumerate(self.convs):
            # print(conv.weight.data.size(), torch.from_numpy(trained_weights['conv1d_'+str(i+1)]['weights']).size())
            conv.weight.data.copy_(torch.from_numpy(trained_weights['conv1d_'+str(i+1)]['weights']).transpose(0,2))
            conv.bias.data.copy_(torch.from_numpy(trained_weights['conv1d_'+str(i+1)]['bias']))

        # print(self.fc.weight.data.size(), torch.from_numpy(trained_weights['dense_1']['weights']).size())
        self.fc.weight.data.copy_(torch.from_numpy(trained_weights['dense_1']['weights']).transpose(0,1))
        self.fc.bias.data.copy_(torch.from_numpy(trained_weights['dense_1']['bias']))

    def forward(self, x_emb):

        # x_emb = self.embedding(x)
        x_emb = x_emb.transpose(1,2)
        feat_maps = [F.relu(conv(x_emb)) for conv in self.convs]
        pooled_maps = torch.cat([F.max_pool1d(feat_map, feat_map.size()[2],
            stride=1) for feat_map in feat_maps], 1).squeeze()

        pooled_maps = self.dropout(pooled_maps)

        logit = self.fc(pooled_maps)

        return logit

    def predict(self, x):

        return F.sigmoid(self.forward(x))


