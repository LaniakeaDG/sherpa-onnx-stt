import argparse
import asyncio
import json
import logging
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import time

import numpy as np
import sherpa_onnx
import websockets

clients =[]
max_wait_ms = 100 #ms
max_batch_size = 3
nn_pool_size = 1 #Number of threads for NN computation and decoding.
nn_pool = ThreadPoolExecutor(
    max_workers=nn_pool_size,
    thread_name_prefix="nn",
)
sample_rate = 16000
current_active_connections = 0

#sperker_identification

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
speaker = None


# 根据 id 查找客户端
def find_clients_by_id(client_id):
    return [client for client in clients if client["id"] == client_id]

# 向特定 ID 的客户端发送消息
async def send_message_to_client_by_id(client_id, message):
    targets = find_clients_by_id(client_id)
    if targets:
        json_message = json.dumps(message)
        for target in targets:
            try:
                await target["websocket"].send(json_message)
                print(f"Message sent to {target['id']} ({target['websocket'].remote_address}): {json_message}")
            except websockets.ConnectionClosed:
                print(f"Failed to send to {target['id']} ({target['websocket'].remote_address}), connection closed")
    else:
        print(f"No client found with ID: {client_id}")


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
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
        provider=provider,
        modeling_unit="", #cjkchar+bpe #cjkchar
        bpe_vocab=''
    )

    return recognizer

# recognizer = create_recognizer()

class StreamingServer(object):
    def __init__(
        self,
        recognizer: sherpa_onnx.OnlineRecognizer,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int
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


    async def run(self, port: int):

        TRANSLATE_SERVER_URI = "ws://localhost:8765"
        try:
            async with websockets.connect(TRANSLATE_SERVER_URI) as translate_socket:
                print("Connected to translate server!")
                # 发送注册消息（客户端启动时自动发送）
                register_msg = {
                    "id": "client_000",
                    "type": "Server_STT",
                    "des": "STT"
                }
                await translate_socket.send(json.dumps(register_msg))
                print(f"Sent registration: {register_msg}")

                tasks = []
                # for i in range(self.nn_pool_size):
                #     tasks.append(asyncio.create_task(self.stream_consumer_task()))
                receive_task = asyncio.create_task(self.receive_data(translate_socket))
                tasks.append(receive_task)

                async with websockets.serve(
                    lambda ws: self.handle_connection(ws, translate_socket),
                    host="0.0.0.0",
                    port=port
                ):
                    print(f"WebSocket server started on ws://localhost:{port}")
                    await asyncio.Future()  # run forever

                await asyncio.gather(*tasks)  # not reachable

        except ConnectionRefusedError:
            print("Connection refused. Is the WebSocket server running at ws://localhost:8765?")
        except websockets.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")
        except Exception as e:
            print(f"Error: {e}")


        
    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
        translate_socket
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        global speaker
        try:
            await self.handle_connection_impl(socket,translate_socket)
        except websockets.exceptions.ConnectionClosedError:
            logging.info(f"{socket.remote_address} disconnected")
        finally:
            # Decrement so that it can accept new connections
            speaker = None
            manager.remove("speaker0")
            print("Remove speaker0")
            self.current_active_connections -= 1
            for client in clients[:]:  # 使用副本以避免修改时遍历问题
                if client["websocket"] == socket:
                    clients.remove(client)
                    print(f"Client {client['id']} ({socket.remote_address}) removed from list")
            logging.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
        translate_socket
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        global speaker
        self.current_active_connections += 1
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        stream = self.recognizer.create_stream()
        stream_sp = extractor.create_stream()
        segment = 0

        #注册消息
        message = await socket.recv()
        data = json.loads(message)
        print(f"Received message from {socket.remote_address}")
                
        # 如果是注册消息，保存客户端信息
        if "id" in data and "type" in data and "des" in data:
            client_info = {
                "websocket": socket,
                "id": data["id"],
                "type": data["type"],
                "des": data["des"]
            }
            clients.append(client_info)
            print(f"Client registered: {client_info}")

        while True:
            samples = await self.recv_audio_samples(socket)
            request_time = time.time()

            if samples is None:
                self.recognizer.decode_streams([stream])
                result = self.recognizer.get_result(stream)
                print(result)
                if result == '':
                    print("[DEBUG] No result after flush, skipping.")
                    continue
                message = {
                    "id" : "client_000",
                    "msg_id" : segment,
                    "content": result,
                    "process_time" : time.time() -request_time
                }
                print(f"[DEBUG] Flushed final message: {message}")
                self.recognizer.reset(stream)
                segment += 1

                stream_sp.input_finished()

                if speaker is None: 
                    print("[DEBUG] No speaker registered, creating embedding.")
                    embedding = extractor.compute(stream_sp)
                    embedding = np.array(embedding)
                    speaker = embedding   
                    status = manager.add("speaker0", embedding)
                    if not status:
                        raise RuntimeError(f"Failed to register speaker 0")
                    else:
                        print("Speaker 0 has successfully registered.")
                else:
                    print("[DEBUG] Checking speaker identity.")
                    embedding = extractor.compute(stream_sp)
                    embedding = np.array(embedding)
                    name = manager.search(embedding, threshold=0.6)
                    stream_sp = extractor.create_stream()
                    if not name:
                        print("Unrecognized speaker detected.")
                        await send_message_to_client_by_id("client_001",message)
                        await translate_socket.send(json.dumps(message))
                    else:
                        print("Detect speaker 0.")
            else:
                stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
                stream_sp.accept_waveform(sample_rate=self.sample_rate, waveform=samples)

                # while self.recognizer.is_ready(stream):
                #     await self.compute_and_decode(stream)
                #     result = self.recognizer.get_result(stream)
                #     print(result)
                #     if self.recognizer.is_endpoint(stream):
                #         result = self.recognizer.get_result(stream)
                #         message = {
                #             "id" : "client_000",
                #             "msg_id" : segment,
                #             "content": result,
                #             "process_time" : time.time() -request_time
                #         }
                #         self.recognizer.reset(stream)
                #         segment += 1
                #         print(message)
                #         await translate_socket.send(json.dumps(message))
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_streams([stream])
                    result = self.recognizer.get_result(stream)
                    print(result)
                    if result == '':
                        break

                    if self.recognizer.is_endpoint(stream):
                        print("[DEBUG] Is Endpoint.")
                        result = self.recognizer.get_result(stream)
                        message = {
                            "id" : "client_000",
                            "msg_id" : segment,
                            "content": result,
                            "process_time" : time.time() -request_time
                        }
                        self.recognizer.reset(stream)
                        segment += 1
                        print(message)
                        stream_sp.input_finished()
                        if speaker is None: #未注册
                            # await send_message_to_client_by_id("client_001",message)
                            # await translate_socket.send(json.dumps(message))
                            if extractor.is_ready(stream_sp):
                                embedding = extractor.compute(stream_sp)
                                stream_sp = extractor.create_stream()
                                embedding = np.array(embedding)
                                speaker = embedding
                                status = manager.add("speaker0", embedding)
                                if not status:
                                    raise RuntimeError(f"Failed to register speaker 0")
                                else:
                                    print("Speaker 0 has successfully registered.")
                        else:
                            embedding = extractor.compute(stream_sp)
                            stream_sp = extractor.create_stream()
                            embedding = np.array(embedding)
                            name = manager.search(embedding, threshold=0.6)
                            if not name:
                                print("Unrecognized speaker detected.")
                                await send_message_to_client_by_id("client_001",message)
                                await translate_socket.send(json.dumps(message))
                            else:
                                print("Detect speaker 0.")
                                
                        # await send_message_to_client_by_id("client_001",message)
                        # await translate_socket.send(json.dumps(message))

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[np.ndarray]:
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
        msg_id = data["msg_id"]
        print(f"Received message from {socket.remote_address}, mes_id: {msg_id}")
        if msg_id==-1:
            return None
        
        samples = data["samples"]
        return np.array(samples, dtype=np.float32)

    # 处理中英互译接收数据
    async def receive_data(
            self,
            websocket):
        try:
            while True:
                response = await websocket.recv()
                try:
                    data = json.loads(response)
                    print(f"Received from server: {data}")
                    msg_id = data.get("msg_id", "unknown")
                    content = data.get("content", "")
                    print(f"Parsed - msg_id: {msg_id}, content: {content}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {response}")
        except websockets.ConnectionClosed as e:
            print(f"Connection closed during receive: {e}")
        except Exception as e:
            print(f"Error while receiving data: {e}")  

def main():
    recognizer = create_recognizer()
    
    server = StreamingServer(
        recognizer=recognizer,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=10,
        max_queue_size=20,
        max_active_connections=20
    )
    asyncio.run(server.run(10007))


if __name__ == "__main__":
    main()