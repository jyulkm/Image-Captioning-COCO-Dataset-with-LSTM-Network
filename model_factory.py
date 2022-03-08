import torch
import torch.nn as nn
import torchvision.models as models
from coco_dataset import collate_fn
import numpy as np


class resnet(nn.Module):
    def __init__(self, embed_size):
        super(resnet, self).__init__()

        # Using pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Number of input features to the last layer
        fc_input_feat_n = self.resnet.fc.in_features

        # Changes the last layer of the pretrained model
        self.resnet.fc = nn.Linear(fc_input_feat_n, embed_size)

        # freeze weights
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

    def __call__(self, features, captions=None, hidden_state=None, cell_state=None):
        return self.forward(features, captions, hidden_state=None, cell_state=None)

    def transform_features(self, features, word_count):
        """
        reshapes features to make it ready for the LSTM model
        """
        return features.unsqueeze(1).repeat(1, word_count, 1)

    def transform_captions(self, captions, image_count):
        """
        reshapes captions to make it ready for the LSTM model
        image_count: number of images (ie rows)
        """

        lengths = [len(caption) for caption in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return self.embedding(targets.cuda())

    def forward(self, features, captions, hidden_state=None, cell_state=None):
        # reshape features
        word_count = captions.shape[1]
        features = self.transform_features(features, word_count)

        # reshape captions
        image_count = features.shape[0]
        captions = self.transform_captions(captions, image_count)

        # combine features and captions to feed the LSTM model
        combined_input = torch.cat((features, captions), dim=2)

        if hidden_state == None:
            lstm_output, (hidden_state, cell_state) = self.lstm(combined_input)
        else:
            lstm_output, (hidden_state, cell_state) = self.lstm(
                combined_input, hidden_state, cell_state)

        linear_output = self.linear(lstm_output)

        return linear_output, (hidden_state, cell_state)


class baseline:
    def __init__(self, config_data, hidden_size, embedding_size, vocab_size, num_layers, max_len):
        # We suggest as a baseline model to use 2 layers of 512 LSTM units each, followed by a fully-connected layer to the softmax output.

        hidden_state = cell_state = 0

        self.config_data = config_data
        self.embedding_size = embedding_size
        self.encoder = resnet(embedding_size)
        self.decoder = LSTM(embedding_size, hidden_size,
                            vocab_size, num_layers)
        self.max_len = max_len

    def __call__(self, image, captions=None):
        if captions != None:
            return self.train(image, captions)
        else:
            return self.test(image)

    def train(self, image, captions):
        """
        calls on forward() in one-go to perform teacher-forcing
        """
        features = self.encoder(image)
        output, _ = self.decoder(features, captions)

        return output

    def test(self, image):
        """
        uses for-loop; no teacher-forcing 
        """
        deterministic = self.config_data['generation']['deterministic']
        temperature = self.config_data['generation']['temperature']
        num_of_images = image.shape[0]

        features = self.encoder(image)
        captions = torch.zeros(num_of_images, 1).cuda()

        token_ids = []
        token_ids_t = torch.empty(
            num_of_images, self.max_len, self.embedding_size)
        hidden_state = cell_state = None
        for t in range(self.max_len):
            output, (hidden_state, cell_state) = self.decoder(
                features, captions, hidden_state, cell_state)
            output = output.squeeze()

            if deterministic:
                token_id = output.argmax(dim=2).cpu().detach()
            else:
                token_id = nn.functional.softmax(
                    output.div(temperature)).multinomial(1).view(-1)

            token_ids.append(token_id.cpu().numpy())
            captions = token_id.unsqueeze(-1)
#             token_ids_t[t, :, :] = output

        return np.array(token_ids)

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
        model = baseline(config_data, hidden_size,
                         embedding_size, vocab_size, num_layers, max_len)
        return model
