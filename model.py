import torch
from torch import nn
from huggingface.transformers import Wav2Vec2Model

class ConvolutionalPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvolutionalPredictor, self).__init__()
        
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        
    def forward(self, X):
        return self.conv(X)
        
class CausalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        """
        x: input tensor of shape (batch_size, seq_len, dim)
        mask: optional mask tensor of shape (batch_size, seq_len)
        """
        # Compute query, key, and value tensors
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        
        # Apply mask to scores, if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        # Apply softmax to obtain attention weights
        weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to value tensor
        y = torch.matmul(weights, value)
        
        return y
    
    
class VQFeatureExtractor(nn.Module):
    def __init__(self, wav2vec_pretrained_model_name, num_codewords=320, codebook_dim=256):
        super(VQFeatureExtractor, self).__init__()
        
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_pretrained_model_name)
        self.codebook1 = nn.Embedding(num_codewords, codebook_dim)
        self.codebook2 = nn.Embedding(num_codewords, codebook_dim)
        self.predictor1 = ConvolutionalPredictor(codebook_dim, num_codewords)
        self.predictor2 = ConvolutionalPredictor(codebook_dim, num_codewords)
        

    def forward(self, speech_input):
        features = self.wav2vec(speech_input)
        
        codeword_indices1 = features.argmax(dim=-1)
        codeword_indices2 = features.argmax(dim=-1)
        
        codewords1 = self.codebook1(codeword_indices1)
        codewords2 = self.codebook2(codeword_indices2)
        
        predicted_indices1 = self.predictor1(codewords1)
        predicted_indices2 = self.predictor2(codewords2)
        
        return predicted_indices1, predicted_indices2
    

class VQModel(nn.Module):
    def __init__(self, num_classes=320):
        super(VQModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class VQClassifier(nn.Module):
    def __init__(self, num_classes=320):
        super(VQClassifier, self).__init__()
        self.num_classes = num_classes
        self.classifier1 = VQModel()
        self.classifier2 = VQModel()
    
    def forward(self, x):
        # predict the index of each codebook separately
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2