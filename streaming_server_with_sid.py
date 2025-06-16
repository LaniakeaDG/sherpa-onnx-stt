import argparse
import asyncio
import http
import json
import logging
import re
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional, Tuple

import numpy as np
import sherpa_onnx
import websockets

def is_english(text):
    for ch in text:
        if ch !=' ':
            return re.match(r'[a-zA-Z]',ch)
    return None

def log_with_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {message}")

def create_recognizer():
    tokens ='./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt'
    encoder ='./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx'
    decoder ='./sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder-epoch-99-avg-1.onnx'
    joiner = './sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner-epoch-99-avg-1.onnx'
    provider= 'cuda'
    num_threads = 1
    
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        max_active_paths=4,
        hotwords_score=1.5,
        hotwords_file="",
        blank_penalty=0.0,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=0.4,
        rule3_min_utterance_length=10,
        provider=provider,
        modeling_unit="", #cjkchar+bpe #cjkchar
        bpe_vocab=''
    )

    return recognizer

def load_speaker_embedding_model():
    model = './3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
    num_threads = 1
    debug = False
    provider = 'cuda'
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
    
extractor = load_speaker_embedding_model()
manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

def format_timestamps(timestamps: List[float]) -> List[str]:
    return ["{:.3f}".format(t) for t in timestamps]


class StreamingServer(object):
    def __init__(
        self,
        recognizer: sherpa_onnx.OnlineRecognizer,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
    ):
        """
        Args:
          recognizer:
            An instance of online recognizer.
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          max_batch_size:
            Max batch size for inference.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          beam_search_params:
            Dictionary containing all the parameters for beam search.
          online_endpoint_config:
            Config for endpointing.
          doc_root:
            Path to the directory where files like index.html for the HTTP
            server locate.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        self.recognizer = recognizer


        self.nn_pool_size = nn_pool_size
        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.stream_queue = asyncio.Queue()

        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

        self.sample_rate = int(recognizer.config.feat_config.sampling_rate)
        # 说话人识别和VAD相关初始化
        self.speakers = {}  # 每个连接的说话人embedding
        self.vad_config = sherpa_onnx.VadModelConfig()
        self.vad_config.silero_vad.model = "./silero_vad.onnx"  # 请确保路径正确
        self.vad_config.silero_vad.min_silence_duration = 0.25
        self.vad_config.silero_vad.min_speech_duration = 0.25
        self.vad_config.sample_rate = self.sample_rate
        self.vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=100)
        # self.window_size = self.vad_config.silero_vad.window_size #512-32ms
        self.window_size = 480 #30ms

        # 对话总结初始化
        self.summary = {}

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the neural network model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()

                    assert self.recognizer.is_ready(item[0])

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.recognizer.decode_streams,
                stream_list,
            )

            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(
        self,
        stream: sherpa_onnx.OnlineStream,
    ) -> None:
        """Put the stream into the queue and wait it to be processed by the
        consumer task.

        Args:
          stream:
            The stream to be processed. Note: It is changed in-place.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.stream_queue.put((stream, future))
        await future

    async def run(self, port: int):
        tasks = []
        for i in range(self.nn_pool_size):
            tasks.append(asyncio.create_task(self.stream_consumer_task()))

        TRANSLATE_SERVER_URI = "ws://localhost:8765"
        try:
            async with websockets.connect(TRANSLATE_SERVER_URI) as translate_socket:
                print("[INFO] Connected to translate server")
                # 发送注册消息（客户端启动时自动发送）
                register_msg = {
                    "id": "client_000",
                    "type": "Server_STT",
                    "des": "STT"
                }
                await translate_socket.send(json.dumps(register_msg))
                print(f"[INFO] Sent registration: {register_msg}")

                #对话总结
                SUMMARY_SERVER_URI = "ws://localhost:8764"
                async with websockets.connect(SUMMARY_SERVER_URI) as summary_socket:
                    print("[INFO] Connected to summary server")
                    # 发送注册消息（客户端启动时自动发送）
                    register_msg = {
                        "id": "client_000",
                        "type": "Server_STT",
                        "des": "STT"
                    }
                    await summary_socket.send(json.dumps(register_msg))
                    print(f"[INFO] Sent registration: {register_msg}")

                    async with websockets.serve(
                        lambda ws: self.handle_connection(ws, translate_socket, summary_socket),
                        host="0.0.0.0",
                        port=port,
                    ):
                        print(f"[INFO] WebSocket server started on ws://localhost:{port}")
                        await asyncio.Future()  # run forever

                    await asyncio.gather(*tasks)  # not reachable

        except ConnectionRefusedError:
            print("Connection refused. Is the WebSocket server running at ws://localhost:8765 and 8764?")
        except websockets.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")
        except Exception as e:
            print(f"Error: {e}")

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
        translate_socket,
        summary_socket
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket,translate_socket,summary_socket)
        except websockets.exceptions.ConnectionClosedError:
            print(f"{socket.remote_address} disconnected")
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            print(
                f"Disconnected: {socket.remote_address}. ")
            print(
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
        translate_socket,
        summary_socket        
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        self.current_active_connections += 1
        print(
            f"Connected: {socket.remote_address}. ")
        print(
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )
        # init
        stream = self.recognizer.create_stream()
        segment = 0
         
        buffer = np.array([], dtype=np.float32) #vad处理音频流
        buffer_comp = np.array([], dtype=np.float32) #完整音频流，修复vad检测精度差问题

        #注册消息 保存客户端信息
        message = await socket.recv()
        data = json.loads(message)
        log_with_timestamp(f"Received message from {socket.remote_address}")
                
        if "id" in data and "type" in data and "des" in data:
            client_info = {
                "websocket": socket,
                "id": data["id"],
                "type": data["type"],
                "des": data["des"]
            }
            print(f"[INFO] Client registered: {client_info}")
            conn_id = data["id"]
            if conn_id not in self.speakers:
                self.speakers[conn_id] = None  
            if conn_id not in self.summary:
                self.summary[conn_id] = []

        while True:
            code, data = await self.recv_audio_samples(socket)

            if code == 1:
                action = data
                if action == 'change_host':
                    self.speakers[conn_id] = None
                    log_with_timestamp(f"[INFO] Host changed by {conn_id}")
                if action == 'generate_summary':
                    message = {
                        "id" : "client_000",
                        "content": ';'.join(self.summary[conn_id])
                    }
                    log_with_timestamp(f"[INFO] Meeting summary generation requested by {conn_id}")
                    await summary_socket.send(json.dumps(message))
                    self.summary[conn_id] = []
            else:
                samples = data
                request_time = time.time()
                
                if samples is None or len(samples) == 0:
                    continue  # 跳过空音频段
                # VAD分段
                buffer = np.concatenate([buffer, samples])
                buffer_comp = np.concatenate([buffer, samples])
                while len(buffer) > self.window_size:
                    self.vad.accept_waveform(buffer[:self.window_size])
                    buffer = buffer[self.window_size:]
                log_with_timestamp(f"VAD accepted, buffer size:{len(buffer)}")
                
                stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
                while self.recognizer.is_ready(stream):
                    if not self.recognizer.is_endpoint(stream):
                        await self.compute_and_decode(stream)
                        result = self.recognizer.get_result(stream)
                        log_with_timestamp(result)
                    # 只在端点检测时做说话人识别
                    else:
                        log_with_timestamp(f"Detect endpoint, result:{result}")
                        
                        self.recognizer.reset(stream)
                        
                        if result != '':
                            # 处理VAD分段
                            vad_segments = []
                            while not self.vad.empty():
                                vad_samples = self.vad.front.samples
                                print(f"vad_samples: {len(vad_samples)}")
                                self.vad.pop()
                                # 过滤静音和过短段
                                vad_segments.append(vad_samples)
                            log_with_timestamp(f"VAD processing ended, vad_segment length: {len(vad_segments)}")
                            if len(vad_segments) == 0:
                                vad_segments.append(buffer_comp)
                                buffer_comp = np.array([], dtype=np.float32)
                            # 说话人识别逻辑
                            if len(vad_segments) > 0:
                                all_samples = np.concatenate(vad_segments)
                                stream_sp = extractor.create_stream()
                                stream_sp.accept_waveform(sample_rate=self.sample_rate, waveform=all_samples)
                                stream_sp.input_finished()
                                embedding = extractor.compute(stream_sp)
                                embedding = np.array(embedding)
                                log_with_timestamp("stream_sp compute")
                                if self.speakers[conn_id] is None:
                                    # 第一次注册
                                    self.speakers[conn_id] = embedding
                                    manager.add(conn_id, embedding)
                                    print(f"[INFO] Registered speaker: {conn_id}")
                                    message = {
                                        "type": "host_changed",
                                        "message": "Host has been successfully changed."
                                    }
                                    await socket.send(json.dumps(message))
                                else:
                                    name = manager.search(embedding, threshold=0.3)
                                    if not name or name != conn_id:
                                        log_with_timestamp("[INFO] Speaker change detected")
                                        message = {
                                            "id" : "client_000",
                                            "content": result,
                                            "msg_id": segment,
                                            "process_time" : time.time() -request_time,
                                            "speaker" : 1
                                        }
                                        print(message)
                                        await translate_socket.send(json.dumps(message))
                                        await socket.send(json.dumps(message))
                                        if is_english(result):
                                            message = {
                                                "id" : "client_000",
                                                "content": "<|EN|>",
                                                "msg_id": segment,
                                                "process_time" : time.time() -request_time,
                                                "speaker" : 1
                                            }
                                        else:
                                            message = {
                                                "id" : "client_000",
                                                "content": "<|ZH|>",
                                                "msg_id": segment,
                                                "process_time" : time.time() -request_time,
                                                "speaker" : 1
                                            }
                                        await socket.send(json.dumps(message))
                                        
                                        segment += 1                       
                                    else:
                                        log_with_timestamp("[INFO] Speaker remains the same")  
                                        message["speaker"] = 0
                                        print(message)
                                        await socket.send(json.dumps(message))
                                        if is_english(result):
                                            message = {
                                                "id" : "client_000",
                                                "content": "<|EN|>",
                                                "msg_id": segment,
                                                "process_time" : time.time() -request_time,
                                                "speaker" : 0
                                            }
                                        else:
                                            message = {
                                                "id" : "client_000",
                                                "content": "<|ZH|>",
                                                "msg_id": segment,
                                                "process_time" : time.time() -request_time,
                                                "speaker" : 0
                                            }
                                    self.summary[conn_id].append(result)
                        
  
    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive a tensor from the client.

        Each message contains either a bytes buffer containing audio samples
        in 16 kHz or contains "Done" meaning the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D np.float32 tensor containing the audio samples or
          return None.
        """
        message = await socket.recv()
        data = json.loads(message)
        # print(data)
        if "action" in data:
            action = data["action"]
            if action:
                log_with_timestamp(f"Received message from {socket.remote_address}, action: {action}")
                return 1, action
        msg_id = data["msg_id"]
        log_with_timestamp(f"Received message from {socket.remote_address}, mes_id: {msg_id}")
        samples = data["samples"]

        return 0, np.array(samples, dtype=np.float32)


def main():
    
    recognizer = create_recognizer()

    max_wait_ms = 100 #ms
    max_batch_size = 3
    nn_pool_size = 1 #Number of threads for NN computation and decoding.
    nn_pool = ThreadPoolExecutor(
        max_workers=nn_pool_size,
        thread_name_prefix="nn",
    )

    server = StreamingServer(
        recognizer=recognizer,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=10,
        max_queue_size=20,
        max_active_connections=20
    )
    asyncio.run(server.run(10003))


if __name__ == "__main__":
    main()