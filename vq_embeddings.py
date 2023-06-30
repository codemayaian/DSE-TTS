import torch
import torch.nn as nn
import torchaudio
import numpy as np
from scipy import stats
from jiwer import wer
from sklearn.preprocessing import StandardScaler

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook_indices = None

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        distances = (x.unsqueeze(2) - self.embeddings.weight.unsqueeze(0)) ** 2
        encoding_indices = torch.argmin(distances, dim=2)
        self.codebook_indices = encoding_indices.detach().cpu().numpy()
        quantized = self.embeddings(encoding_indices)
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return quantized, loss

class DualSpeakerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.speaker_embed = nn.Embedding(num_embeddings, embedding_dim)
        self.style_embed = VQEmbedding(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x, speaker_id):
        # x shape: [batch_size, seq_len, feature_dim]
        style_embed, style_loss = self.style_embed(x)
        speaker_embed = self.speaker_embed(speaker_id)
        return style_embed + speaker_embed.unsqueeze(1), style_loss

class DSETTS(nn.Module):
    def __init__(self, input_dim, output_dim, num_speakers, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, embedding_dim, 5, stride=1, padding=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, output_dim, 5, stride=1, padding=2)
        )
        self.speaker_embed = nn.Embedding(num_speakers, embedding_dim)
        self.style_embed = VQEmbedding(num_speakers, embedding_dim, commitment_cost)

    def forward(self, x, speaker_id):
        style_embed, style_loss = self.style_embed(x)
        speaker_embed = self.speaker_embed(speaker_id)
        z = style_embed + speaker_embed.unsqueeze(1)
        z = self.encoder(z.transpose(1, 2))
        z = self.decoder(z).transpose(1, 2)
        return z, style_loss

class Evaluation:

    def nmos(self, samples):
        """Collect NMOS ratings for a list of samples"""
        scores = []

        # Recruit raters and collect ratings
        for sample in samples:
            rating = self.collect_rating("How native is the speech?", 1, 5, 0.5) 
            scores.append(rating)

        return np.mean(scores), stats.sem(scores)

    def smos(self, samples, speakers):
        """Collect SMOS ratings for samples spoken by speakers"""
        scores = []

        # Recruit raters and collect ratings
        for i in range(len(samples)):
            sample = samples[i]
            speaker = speakers[i]
            rating = self.collect_rating(f"How similar is the speech to {speaker}?", 1, 5, 0.5) 
            scores.append(rating)

        return np.mean(scores), stats.sem(scores)

    def collect_rating(self, prompt, min_score, max_score, increment):
        """Collect a single rating from a rater"""
        print(prompt)
        print(f"Please rate on a scale from {min_score} to {max_score} with {increment}-point increments.")
        rating = None
        while rating is None:
            try:
                rating = float(input("Rating: "))
                if rating < min_score or rating > max_score:
                    print(f"Rating must be between {min_score} and {max_score}.")
                    rating = None
                elif (rating - min_score) % increment != 0:
                    print(f"Rating must be a multiple of {increment}.")
                    rating = None
            except ValueError:
                print("Invalid rating. Please enter a number.")
                rating = None
        return rating

    def mcd(self, ref, syn):
        """Calculate mel-cepstral distortion between reference and synthesized speech"""
        # Extract mel-cepstral coefficients from ref and syn
        ref_mfcc = torchaudio.transforms.MFCC(sample_rate=16000)(ref).detach().cpu().numpy()
        syn_mfcc = torchaudio.transforms.MFCC(sample_rate=16000)(syn).detach().cpu().numpy()

        # Normalize the features
        scaler = StandardScaler()
        ref_mfcc = scaler.fit_transform(ref_mfcc)
        syn_mfcc = scaler.transform(syn_mfcc)

        # Calculate MCD between the two
        mcd = np.mean(np.sqrt(np.mean((ref_mfcc - syn_mfcc) ** 2, axis=1)))
        return mcd

    def wer(self, ref_text, syn_text):
        """Calculate word error rate between reference and synthesized text"""
        wer_score = wer(ref_text, syn_text)
        return wer_score
    
    
        # Normalize
class SpeakerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        return self.embedding(inputs)


class DualSpeakerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.timbre_embedding = SpeakerEmbedding(num_embeddings, embedding_dim)
        self.speaker_embedding = SpeakerEmbedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        timbre = self.timbre_embedding(inputs)
        speaker = self.speaker_embedding(inputs)
        return timbre, speaker


class DSE_TTS(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super().__init__()
        self.dual_speaker_embedding = DualSpeakerEmbedding(num_embeddings, embedding_dim)
        self.rnn = nn.GRU(embedding_dim * 2, hidden_size, num_layers=2, bidirectional=True)

    def forward(self, inputs):
        timbre, speaker = self.dual_speaker_embedding(inputs)
        embeddings = torch.cat((timbre, speaker), dim=-1)
        output, _ = self.rnn(embeddings)
        return output


class SANE_TTS(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super().__init__()
        self.speaker_embedding = SpeakerEmbedding(num_embeddings, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=2, bidirectional=True)

    def forward(self, inputs):
        embeddings = self.speaker_embedding(inputs)
        output, _ = self.rnn(embeddings)
        return output