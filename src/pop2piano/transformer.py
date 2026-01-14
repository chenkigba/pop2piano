"""Transformer model wrapper for Pop2Piano (inference only)."""

import os
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration

from pop2piano.tokenizer import MidiTokenizer, extrapolate_beat_times
from pop2piano.layers.mel import LogMelSpectrogram, ConcatEmbeddingToMel
from pop2piano.beat import extract_rhythm, interpolate_beat_times
from pop2piano.audio import get_stereo


DEFAULT_COMPOSERS = {"various composer": 2052}


class TransformerWrapper(nn.Module):
    """T5-based transformer for music to piano conversion."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.t5config = T5Config.from_pretrained("t5-small")

        for k, v in config.t5.items():
            self.t5config.__setattr__(k, v)

        # Ensure is_gated_act is set correctly for gated activations
        if "gated" in str(self.t5config.feed_forward_proj):
            self.t5config.is_gated_act = True

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.use_mel = self.config.dataset.use_mel
        self.mel_is_conditioned = self.config.dataset.mel_is_conditioned
        self.composer_to_feature_token = config.composer_to_feature_token

        if self.use_mel and not self.mel_is_conditioned:
            self.composer_to_feature_token = DEFAULT_COMPOSERS

        if self.use_mel:
            self.spectrogram = LogMelSpectrogram()
            if self.mel_is_conditioned:
                n_dim = 512
                composer_n_vocab = len(self.composer_to_feature_token)
                embedding_offset = min(self.composer_to_feature_token.values())
                self.mel_conditioner = ConcatEmbeddingToMel(
                    embedding_offset=embedding_offset,
                    n_vocab=composer_n_vocab,
                    n_dim=n_dim,
                )
        else:
            self.spectrogram = None

    @classmethod
    def from_checkpoint(cls, ckpt_path, config, device="cpu"):
        """Load model from checkpoint.

        Args:
            ckpt_path: Path to checkpoint file
            config: Model configuration
            device: Device to load model on

        Returns:
            TransformerWrapper instance
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls(config)

        # Handle pytorch-lightning checkpoint format
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(device)
        return model

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    @torch.no_grad()
    def single_inference(
        self,
        feature_tokens=None,
        audio=None,
        beatstep=None,
        max_length=256,
        max_batch_size=64,
        n_bars=None,
        composer_value=None,
    ):
        """Generate MIDI from audio.

        Args:
            feature_tokens: Feature tokens (optional)
            audio: Audio tensor (time,)
            beatstep: Beat time points
            max_length: Maximum sequence length
            max_batch_size: Maximum batch size
            n_bars: Number of bars per batch
            composer_value: Composer style token

        Returns:
            relative_tokens, notes, PrettyMIDI object
        """
        assert feature_tokens is not None or audio is not None
        assert beatstep is not None

        if feature_tokens is not None:
            assert len(feature_tokens.shape) == 1

        if audio is not None:
            assert len(audio.shape) == 1

        config = self.config
        PAD = self.t5config.pad_token_id
        n_bars = config.dataset.n_bars if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            beatstep = beatstep - beatstep[0]

        if self.use_mel:
            input_ids = None
            inputs_embeds, ext_beatstep = self.prepare_inference_mel(
                audio,
                beatstep,
                n_bars=n_bars,
                padding_value=PAD,
                composer_value=composer_value,
            )
            batch_size = inputs_embeds.shape[0]
        else:
            raise NotImplementedError

        # Generate in batches to handle GPU memory
        relative_tokens = list()
        for i in range(0, batch_size, max_batch_size):
            start = i
            end = min(batch_size, i + max_batch_size)

            if input_ids is None:
                _input_ids = None
                _inputs_embeds = inputs_embeds[start:end]
            else:
                _input_ids = input_ids[start:end]
                _inputs_embeds = None

            _relative_tokens = self.transformer.generate(
                input_ids=_input_ids,
                inputs_embeds=_inputs_embeds,
                max_length=max_length,
            )
            _relative_tokens = _relative_tokens.cpu().numpy()
            relative_tokens.append(_relative_tokens)

        max_length = max([rt.shape[-1] for rt in relative_tokens])
        for i in range(len(relative_tokens)):
            relative_tokens[i] = np.pad(
                relative_tokens[i],
                [(0, 0), (0, max_length - relative_tokens[i].shape[-1])],
                constant_values=PAD,
            )
        relative_tokens = np.concatenate(relative_tokens)

        pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
            relative_tokens,
            beatstep=ext_beatstep,
            bars_per_batch=n_bars,
            cutoff_time_idx=(n_bars + 1) * 4,
        )

        return relative_tokens, notes, pm

    def prepare_inference_mel(
        self, audio, beatstep, n_bars, padding_value, composer_value=None
    ):
        """Prepare mel spectrogram for inference."""
        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        sample_rate = self.config.dataset.sample_rate
        ext_beatstep = extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            batch = []
            for i in range(0, n_target_step, n_steps):
                start_idx = i
                end_idx = min(i + n_steps, n_target_step)
                start_sample = int(ext_beatstep[start_idx] * sample_rate)
                end_sample = int(ext_beatstep[end_idx] * sample_rate)
                feature = audio[start_sample:end_sample]
                batch.append(feature)
            return batch

        def pad_and_stack_batch(batch):
            batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
            return batch

        batch = split_audio(audio)
        batch = pad_and_stack_batch(batch)

        inputs_embeds = self.spectrogram(batch).transpose(-1, -2)
        if self.mel_is_conditioned:
            composer_value = torch.tensor(composer_value).to(self.device)
            composer_value = composer_value.repeat(inputs_embeds.shape[0])
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer_value)
        return inputs_embeds, ext_beatstep

    @torch.no_grad()
    def generate(
        self,
        audio_path=None,
        composer=None,
        steps_per_beat=2,
        n_bars=2,
        max_batch_size=None,
        beatsteps=None,
        audio_y=None,
        audio_sr=None,
    ):
        """Generate piano MIDI from audio.

        Args:
            audio_path: Path to audio file
            composer: Composer style (e.g., "composer1" to "composer21")
            steps_per_beat: Steps per beat for quantization
            n_bars: Number of bars per batch
            max_batch_size: Maximum batch size
            beatsteps: Pre-computed beat steps (optional)
            audio_y: Pre-loaded audio waveform (optional)
            audio_sr: Sample rate of audio_y (optional)

        Returns:
            PrettyMIDI object
        """
        config = self.config
        device = self.device

        max_batch_size = 64 // n_bars if max_batch_size is None else max_batch_size
        composer_to_feature_token = self.composer_to_feature_token

        if composer is None:
            composer = random.sample(list(composer_to_feature_token.keys()), 1)[0]

        composer_value = composer_to_feature_token[composer]

        LIBROSA_SAMPLERATE = 44100

        if beatsteps is None:
            y, sr = librosa.load(audio_path, sr=LIBROSA_SAMPLERATE)
            (
                bpm,
                beat_times,
                confidence,
                estimates,
                beat_intervals,
            ) = extract_rhythm(audio_path, y=y)
            beat_times = np.array(beat_times)
            beatsteps = interpolate_beat_times(beat_times, steps_per_beat, extend=True)
        else:
            y = None

        if self.use_mel:
            if audio_y is None and config.dataset.sample_rate != LIBROSA_SAMPLERATE:
                if y is not None:
                    y = librosa.core.resample(
                        y,
                        orig_sr=LIBROSA_SAMPLERATE,
                        target_sr=config.dataset.sample_rate,
                    )
                    sr = config.dataset.sample_rate
                else:
                    y, sr = librosa.load(audio_path, sr=config.dataset.sample_rate)
            elif audio_y is not None:
                if audio_sr != config.dataset.sample_rate:
                    audio_y = librosa.core.resample(
                        audio_y, orig_sr=audio_sr, target_sr=config.dataset.sample_rate
                    )
                    audio_sr = config.dataset.sample_rate
                y = audio_y
                sr = audio_sr

            start_sample = int(beatsteps[0] * sr)
            end_sample = int(beatsteps[-1] * sr)
            _audio = torch.from_numpy(y)[start_sample:end_sample].to(device)
            fzs = None
        else:
            raise NotImplementedError

        relative_tokens, notes, pm = self.single_inference(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            max_length=config.dataset.target_length
            * max(1, (n_bars // config.dataset.n_bars)),
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            composer_value=composer_value,
        )

        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        return pm
