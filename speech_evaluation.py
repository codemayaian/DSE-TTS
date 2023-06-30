import numpy as np
from scipy import stats
from jiwer import wer

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
        ref_mfcc = librosa.feature.mfcc(ref, sr=16000, n_mfcc=13, hop_length=160, n_fft=512)
        syn_mfcc = librosa.feature.mfcc(syn, sr=16000, n_mfcc=13, hop_length=160, n_fft=512)

        # Calculate MCD between the two
        mcd = np.mean(np.sqrt(np.mean((ref_mfcc - syn_mfcc) ** 2, axis=0)))
        return mcd

    def wer(self, ref_text, syn_text):
        """Calculate word error rate between reference and synthesized text"""
        wer_score = wer(ref_text, syn_text)
        return wer_score

def subjective_evaluation(synthesized_samples, native_language, reference_speaker, num_raters=5):
    """
    Computes NMOS and SMOS scores for a set of synthesized speech samples using subjective evaluation.
    Args:
        synthesized_samples (list of str): List of synthesized speech samples.
        native_language (str): The native language of the synthesized speech samples.
        reference_speaker (str): The name of the reference speaker used for cross-lingual synthesis.
        num_raters (int, optional): The number of raters for evaluation. Defaults to 5.
    Returns:
        nmos_score (float): The mean nativeness mean opinion score.
        smos_score (float): The mean speaker similarity mean opinion score.
    """
    nmos_total = 0
    smos_total = 0
    for sample in synthesized_samples:
        # Present the synthesized sample to the raters
        print("Synthesized speech:\n", sample)
        print("Please rate the nativeness and speaker similarity on a scale from 1 to 5.")
        nmos = 0
        smos = 0
        for i in range(num_raters):
            nmos += float(input("Nativeness rating (1-5): "))
            smos += float(input("Speaker similarity rating (1-5): "))
        nmos /= num_raters
        smos /= num_raters
        print("Nativeness mean opinion score:", nmos)
        print("Speaker similarity mean opinion score:", smos)
        nmos_total += nmos
        smos_total += smos
    nmos_score = nmos_total / len(synthesized_samples)
    smos_score = smos_total / len(synthesized_samples)
    print("Overall nativeness mean opinion score:", nmos_score)
    print("Overall speaker similarity mean opinion score:", smos_score)
    return nmos_score, smos_score