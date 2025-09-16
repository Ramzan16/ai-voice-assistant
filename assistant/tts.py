import torch
import sounddevice as sd
from kokoro import KPipeline
from loguru import logger
from collections.abc import Generator
import time
import queue
import threading

class TextToSpeech:
    """
    Handles converting text into spoken audio using the Kokoro TTS model.
    Includes logic for handling and chunking streamed text for more natural speech.
    """
    def __init__(self):
        # ... existing __init__ code ...
        logger.info("Initializing Kokoro Text-to-Speech engine...")
        try:
            # Determine the device to run the model on (GPU if available)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Let the pipeline create and manage the model on the correct device.
            self.pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a', device=self.device)
            self.sample_rate = 24000
            
            # --- FIX: Set up a thread-safe queue and a dedicated playback thread ---
            # This prevents audio chunks from interrupting each other.
            self.audio_queue = queue.Queue()
            self.playback_thread = threading.Thread(target=self._audio_playback_loop, daemon=True)
            self.playback_thread.start()
            
            logger.info("Kokoro TTS engine initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            logger.error("Please ensure you have run 'pip install kokoro-tts torch sounddevice'.")
            self.pipeline = None

    def _audio_playback_loop(self):
        """
        Pulls audio chunks from the queue and plays them sequentially.
        Runs in a separate thread to not block the main synthesis process.
        """
        while True:
            audio_chunk = self.audio_queue.get()
            sd.play(audio_chunk, self.sample_rate)
            sd.wait() # Block this thread until the current chunk is done playing.
            self.audio_queue.task_done()

    def wait_for_audio_to_finish(self):
        """Blocks execution until all items in the audio queue have been processed."""
        self.audio_queue.join()

    def speak(self, text: str, voice: str = 'af_bella'):
        """
        Synthesizes and speaks a complete string of text. This is a blocking operation
        that waits until the audio has finished playing.
        """
        if not self.pipeline:
            logger.error("Cannot speak, Kokoro TTS pipeline is not available.")
            return

        if not text.strip():
            logger.warning("TTS received empty text. Nothing to speak.")
            return

        try:
            log_text = text[:50].strip().replace('\n', ' ')
            logger.info(f"Synthesizing and speaking: '{log_text}...'")
            
            all_audio = []
            for result in self.pipeline(text, voice=voice):
                if result.audio is not None:
                    all_audio.append(result.audio)

            if all_audio:
                waveform = torch.cat(all_audio, dim=0)
                # Put the complete audio into the queue to be played by the playback thread.
                self.audio_queue.put(waveform.numpy())
                # Wait for the queue to be processed to maintain blocking behavior.
                self.wait_for_audio_to_finish()

        except Exception as e:
            logger.error(f"Failed to speak text with Kokoro: {e}")

    def speak_stream(self, text_generator: Generator[str, None, None], voice: str = 'af_bella'):
        """
        Handles streaming text by buffering it into sentences and playing them
        with low latency as they are generated.
        """
        if not self.pipeline:
            logger.error("Cannot speak stream, Kokoro TTS pipeline is not available.")
            return
            
        sentence_buffer = ""

        for token in text_generator:
            sentence_buffer += token
            
            # --- FIX: More intelligent sentence splitting ---
            # Find the earliest position of a valid sentence terminator.
            split_pos = -1
            
            possible_positions = []
            # Newline is an unambiguous sentence break.
            if '\n' in sentence_buffer:
                possible_positions.append(sentence_buffer.find('\n'))
            
            # Other punctuation is only a break if followed by a space, to avoid splitting on "Mr."
            for punc in '.!?':
                pos = sentence_buffer.find(punc + ' ')
                if pos != -1:
                    # We found 'punc ', so the split happens after the punctuation.
                    possible_positions.append(pos)
            
            if possible_positions:
                # Choose the earliest valid split position from the found terminators.
                split_pos = min(possible_positions)
                
                sentence_to_speak = sentence_buffer[:split_pos + 1]
                sentence_buffer = sentence_buffer[split_pos + 1:]
                
                sentence_to_speak = sentence_to_speak.strip()
                if sentence_to_speak:
                    log_chunk = sentence_to_speak.replace('\n', ' ')
                    logger.info(f"Streaming chunk: '{log_chunk}'")
                    for result in self.pipeline(sentence_to_speak, voice=voice):
                        if result.audio is not None:
                            self.audio_queue.put(result.audio.numpy())

        # After the stream is finished, process any remaining text in the buffer.
        if sentence_buffer.strip():
            log_buffer = sentence_buffer.strip().replace('\n', ' ')
            logger.info(f"Playing remaining buffer: '{log_buffer}'")
            # Synthesize and queue the final chunk.
            for result in self.pipeline(sentence_buffer.strip(), voice=voice):
                if result.audio is not None:
                    self.audio_queue.put(result.audio.numpy())


if __name__ == "__main__":
    pass