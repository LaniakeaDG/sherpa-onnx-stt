#!/usr/bin/env python3sounddevicesd

"""
This script shows how to use Python APIs for speaker identification with
a microphone.

Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Run this script

Assume the filename of the text file is speaker.txt.

python3 ./python-api-examples/speaker-identification.py \
  --speaker-file ./speaker.txt \
  --model ./wespeaker_zh_cnceleb_resnet34.onnx
"""
import argparse
import queue
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf



speaker_file = './speaker_file.txt'
model = './3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
num_threads = 1
threshold = 0.6
debug = False
provider = 'cuda'





def load_speaker_embedding_model():
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return extractor


def load_speaker_file() -> Dict[str, List[str]]:
    
    ans = defaultdict(list)
    with open(speaker_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, filename = fields
            ans[speaker_name].append(filename)
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_speaker_embedding(
    filenames: List[str],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    assert len(filenames) > 0, "filenames is empty"

    ans = None
    for filename in filenames:
        print(f"processing {filename}")
        samples, sample_rate = load_audio(filename)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        if ans is None:
            ans = embedding
        else:
            ans += embedding

    return ans / len(filenames)


g_buffer = queue.Queue()
g_stop = False
g_sample_rate = 16000
g_read_mic_thread = None





def main():
    extractor = load_speaker_embedding_model()
    speaker_file = load_speaker_file()

    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    for name, filename_list in speaker_file.items():
        embedding = compute_speaker_embedding(
            filenames=filename_list,
            extractor=extractor,
        )
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")
        
    filename = './chinese.mp3'
    print(f"input: {filename}")
    samples, sample_rate = load_audio(filename)
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()


    # stream = extractor.create_stream()
    # while not g_buffer.empty():
    #     samples = g_buffer.get()
    #     stream.accept_waveform(sample_rate=g_sample_rate, waveform=samples)
    # stream.input_finished()

    embedding = extractor.compute(stream)
    embedding = np.array(embedding)
    name = manager.search(embedding, threshold=threshold)
    print(name)
    # if not name:
    #     status = manager.add("0", embedding)
    #     print("Add speker 0")

    


if __name__ == "__main__":
    main()