import torch
import torch.nn as nn
import torchvision.models as models


class resnet(nn.Module):
    def __init__(self, embed_size):
        super(resnet, self).__init__()

        # Using pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Number of input features to the last layer
        fc_input_feat_n = self.resnet.fc.in_features

        # Changes the last layer of the pretrained model
        self.resnet.fc = nn.Linear(fc_input_feat_n, embed_size)

        for name, param in self.resnet.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def __call__(self, img):
        features = self.forward(img)
        return features

    def forward(self, img):
        features = self.resnet(img)
        return features

        # make sure to normalize output data (writeup says to transform image data)


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, max_len, num_layers=2):  # must change code
        super().__init__()

        # Defining the emedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Defining the LSTM model
        self.lstm = nn.LSTM(
            input_size=embed_size*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.max_len = max_len  # max length of caption

    def __call__(self, features, captions):
        return self.forward(features, captions)

    def train(self, features, captions):
        captions = captions[:, :-1]

        # pad captions to ensure same length: required to put all captions across a minibatch in one tensor
        captions = torch.nn.functional.pad(
            input=captions, pad=(1, 0), mode='constant', value=0)

        # embed captions to convert sparse one-hot-encoded matrix to something else
        captions = self.embedding(captions)

        # reshape features from 2d to 3d: lstm requires a 3d tensor
        features = features.unsqueeze(1).repeat(1, captions.shape[1], 1)

        # combine features and captions to feed the LSTM model
        combined_input = torch.cat((features, captions), dim=2)

        lstm_output, (hidden_state, cell_state) = self.lstm(combined_input)

        linear_output = self.linear(lstm_output)

        return linear_output


class baseline:
    def __init__(self, hidden_size, embedding_size, vocab_size, num_layers, max_len):
        # We suggest as a baseline model to use 2 layers of 512 LSTM units each, followed by a fully-connected layer to the softmax output.

        hidden_state = cell_state = 0

        self.encoder = resnet(embedding_size)
        self.decoder = LSTM(embedding_size, hidden_size,
                            vocab_size, num_layers)
        self.max_len = max_len

    def __call__(self, image, captions):
        self.train(image, captions)

    def train(self, image, captions):
        features = self.encoder(image)
        return self.decoder.train(features, captions)

    def deterministic_caption_generator(self, image, captions):
        token_ids = []

        N = image.shape[0]

        captions = self.decoder.embedding(torch.tensor([0]).cuda())
        captions = captions.unsqueeze(1)
        captions = captions.repeat(N, 1, 1)

        features = self.encoder(image)
        features = features.unsqueeze(1).repeat(1, captions.shape[1], 1)

        for _ in range(self.max_len):
            combined_input = torch.cat((features, captions), dim=2)
            decoder_output, hidden_state, cell_state = self.decoder.lstm(
                embed, states)

            prediction = decoder_output.argmax(1).cpu().detach().numpy()

            ids_storage.append(prediction)

            captions = self.decoder.embedding(prediction)

        return token_ids

    def stochastic_caption_generator(self, image, captions):
        encoder_output = self.encoder(image)
        decoder_output = self.decoder(encoder_output, captions)

        softmax_output = nn.functional.softmax(
            decoder_output.div(temperature)).multinomial(1).view(-1)


#         token_ids = []
#         while len(token_ids) <= max_len:
#             self.decoder(image, captions)


def get_start(img):
    """
    gets the 'start'(i.e. first) word of a caption.

    img: 256 X 256 input image
    """
    lstm = LSTM()
    lstm(img)
    start = softmax(img)
    return start

# Build and return the model here based on the configuration.


def get_model(config_data, vocab, num_layers=2):
    """
    # pseudo code structure
    resnet_data = resnet()
    resnet_data_norm = normalize(resnet_data)
    LSTM(resnet_data_norm)
    """
    # unravel configurations
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_size = vocab.idx
    max_len = config_data['generation']['max_length']

    if model_type == 'RNN':
        return baseline(hidden_size, embedding_size, vocab_size, num_layers, max_len)

    # only use hidden step to generate the output
