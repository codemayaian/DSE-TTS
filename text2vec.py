import torch

class Txt2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for the text encoder
        self.embedding = nn.Embedding(10000, 256)
        self.conv1d_layer = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pooling_layer = nn.AdaptiveMaxPool1d(1)
        self.linear_layer = nn.Linear(256, 256)
    
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        text_embed = self.embedding(text)
        conv1d_out = self.conv1d_layer(text_embed.permute(0, 2, 1))
        max_pool_out = self.max_pooling_layer(conv1d_out).squeeze(-1)
        linear_out = self.linear_layer(max_pool_out)
        text_features = torch.tanh(linear_out)
        return text_features

def text_to_tensor(text, char2idx):
    """
    Convert text to tensor based on character-to-index mapping.
    """
    tensor = []
    for char in text:
        if char in char2idx:
            tensor.append(char2idx[char])
        else:
            tensor.append(char2idx['<unk>'])
    return torch.tensor(tensor).unsqueeze(0)

def tensor_to_text(tensor, idx2char):
    """
    Convert tensor to text based on index-to-character mapping.
    """
    text = ''
    for idx in tensor:
        if idx.item() == 0:
            break
        text += idx2char[idx.item()]
    return text

def synthesize_text(txt2vec, text, char2idx, idx2char):
    """
    Convert input text to text features using the Txt2Vec model.
    """
    txt2vec.eval()
    with torch.no_grad():
        tensor = text_to_tensor(text, char2idx)
        text_features = txt2vec(tensor)
        text = tensor_to_text(tensor.squeeze(0), idx2char)
    return text_features, text