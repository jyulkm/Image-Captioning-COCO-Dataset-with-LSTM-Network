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

    def __call__(self, img):
        self.forward(img)

    def forward(self, img):
        ...
        # make sure to normalize output data


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):  # must change code
        super().__init__()

        # Defining the emedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # Defining the LSTM model
        self.lstm = nn.LSTM(
            input_size=embed_size*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def __call__(self, features, captions, vocab):
        self.generate_caption()

    def generate_caption():
        caption = []
        self.generate_first_token(caption, img)
        self.generate_susequet_tokens(caption)

    def generate_first_token(caption, img):

        word = ...
        caption.append(word)

    def generate_subsequent_tokens(caption):
        eos = False

        while not eos:
            eos = generate_subsequent_token()

    def generate_subsequent_token(caption):
        softmax = self.forward()
        word = argmax(softmax)

        if word != 'EOS':
            return True

        else:
            caption.append(word)
            return False

    def forward(self, features, captions, vocab):
        if len(captions) == 0:
            first = self.lstm(input, (h0, c0))
            return first

        captions.append(...)

        return eos

        ...


def baseline(img):
    # We suggest as a baseline model to use 2 layers of 512 LSTM units each, followed by a fully-connected layer to the softmax output.
    captions = []

    hidden_state = cell_state = 0
    hidden_states = (hidden_state, cell_state)

    # first step
    EMBED_SIZE = 300

    cnn = resnet(EMBED_SIZE)
    feature_vec = cnn(img)

    initial_generator = LSTM(embed_size, hidden_size,
                             vocab_size, num_layers)  # first word of caption
    caption[0] = initial_generator(features, captions, vocab)

    caption

    # second step
    input = start
    output = first word


def loss_fnc():
    """
    automatically combines both softmax and negative log likelihood loss
    """
    CrossEntropyLoss


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


def get_model(config_data, vocab):
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

    # only use hidden step to generate the output
