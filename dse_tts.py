 # txt2vec acoustic model 
import torch
import torch.nn as nn
import dse_speaker_embeddings 
import vq_xvector


class DSE_TTS:
    def __init__(self, num_speakers, embedding_dim):
        self.acoustic_model = AcousticModel()
        self.txt2vec = Txt2Vec()
        self.vocoder = Vocoder()
        self.vec2wav = Vec2Wav()
        self.dual_speaker_embed = DualSpeakerEmbedding(num_speakers, embedding_dim)
    
    def synthesize(self, text, input_lang, target_speaker):
        """
        Synthesize speech waveform from given text using the specified input language and target speaker.

        Parameters:
            text (Tensor): The input text to be synthesized. Shape: [batch_size, seq_len].
            input_lang (Tensor): The input language of the text. Shape: [batch_size].
            target_speaker (Tensor): The target speaker of the synthesized speech. Shape: [batch_size].

        Returns:
            Tensor: The synthesized speech waveform. Shape: [batch_size, waveform_length].
        """
        # text shape: [batch_size, seq_len]
        # input_lang shape: [batch_size]
        # target_speaker shape: [batch_size]
        txt_embed = self.txt2vec(text)
        acoustic_speaker_embed, vocoder_speaker_embed = self.dual_speaker_embed(input_lang, target_speaker)
        acoustic_features = self.acoustic_model(txt_embed, acoustic_speaker_embed)
        vocoder_features = self.vocoder(acoustic_features, vocoder_speaker_embed)
        waveform = self.vec2wav(vocoder_features)
        return waveform
    
    def train(self, train_dataset, num_epochs, batch_size, optimizer):
        # train_dataset: PyTorch Dataset object
        # num_epochs: int
        # batch_size: int
        # optimizer: PyTorch optimizer object
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_idx, (text, input_lang, target_speaker, waveform) in enumerate(train_loader):
                optimizer.zero_grad()
                txt_embed = self.txt2vec(text)
                acoustic_speaker_embed, vocoder_speaker_embed = self.dual_speaker_embed(input_lang, target_speaker)
                acoustic_features = self.acoustic_model(txt_embed, acoustic_speaker_embed)
                vocoder_features = self.vocoder(acoustic_features, vocoder_speaker_embed)
                predicted_waveform = self.vec2wav(vocoder_features)
                loss = nn.MSELoss()(predicted_waveform, waveform)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Epoch {} Loss: {}'.format(epoch+1, total_loss / len(train_loader)))



class DualSpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers, embedding_dim):
        super().__init__()
        self.speaker_embed = nn.Embedding(num_speakers, embedding_dim)
        self.vocoder_embed = nn.Embedding(num_speakers, embedding_dim)
    
    def forward(self, speaker_id, target_speaker_id):
        # speaker_id shape: [batch_size]
        # target_speaker_id shape: [batch_size]
        acoustic_speaker_embed = self.speaker_embed(speaker_id)
        vocoder_speaker_embed = self.vocoder_embed(target_speaker_id)
        return acoustic_speaker_embed, vocoder_speaker_embed
    
    def merge(self):
        Vocoder_embed = self.vocoder_embed
        Acoustic_embed = self.speaker_embed
        return Vocoder_embed
    


class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for the acoustic model
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=3, batch_first=True)
        self.linear_layer = nn.Linear(512, 80)
        self.relu = nn.ReLU()
    
    def forward(self, text_embed, speaker_embed):
        # text_embed shape: [batch_size, seq_len, 256]
        # speaker_embed shape: [batch_size, 256]
        concat_embed = torch.cat((text_embed, speaker_embed.unsqueeze(1).repeat(1, text_embed.size(1), 1)), dim=-1)
        lstm_out, _ = self.lstm(concat_embed)
        linear_out = self.linear_layer(lstm_out)
        acoustic_features = self.relu(linear_out)
        return acoustic_features


class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for the vocoder
        self.conv1d_layer = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_layer = nn.Linear(2048, 512)
    
    def forward(self, acoustic_features, speaker_embed):
        # acoustic_features shape: [batch_size, seq_len, 80]
        # speaker_embed shape: [batch_size, 256]
        concat_embed = torch.cat((acoustic_features, speaker_embed.unsqueeze(1).repeat(1, acoustic_features.size(1), 1)), dim=-1)
        conv1d_out = self.conv1d_layer(concat_embed.permute(0, 2, 1))
        lstm_out, _ = self.lstm(conv1d_out.permute(0, 2, 1))
        linear_out = self.linear_layer(lstm_out.permute(0, 2, 1))
        vocoder_features = torch.tanh(linear_out)
        return vocoder_features


class Vec2Wav(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for the waveform generator
        self.conv1d_layer1 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer3 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.tanh = nn.Tanh()
    
    def forward(self, vocoder_features):
        # vocoder_features shape: [batch_size, seq_len, 512]
        conv1d_out = self.conv1d_layer1(vocoder_features.permute(0, 2, 1))
        conv1d_out = self.conv1d_layer2(conv1d_out)
        conv1d_out = self.conv1d_layer3(conv1d_out)
        waveform = self.tanh(conv1d_out)
        return waveform.squeeze()

