from RealtimeSTT import AudioToTextRecorder
import logging

class SpeechToText:
    """
    Handles capturing audio from the microphone and transcribing it to text
    using the RealtimeSTT library.
    """
    def __init__(self, stt_model="base.en", microphone_index=None):
        """
        Initializes the RealtimeSTT recorder.

        Args:
            stt_model (str): The name of the Whisper model to use.
            microphone_index (int, optional): The device index of the microphone.
                                              Defaults to None (system default).
        """
        logging.info("Initializing RealtimeSTT...")
        
        self.recorder = AudioToTextRecorder(
            model=stt_model,
            language="en",
            device=microphone_index,
            spinner=False, # To keep logs clean
            wake_words="jarvis",
            wakeword_backend="openwakeword"
        )
        logging.info("RealtimeSTT initialization complete.")


    def listen(self) -> str:
        """
        Listens for a phrase and transcribes it upon detection of silence.

        Returns:
            The transcribed text as a string, or an empty string if it fails.
        """
        try:
            logging.info("Listening...")
            text = self.recorder.text()
            logging.info("Transcription complete.")
            return text.strip()

        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}")
            return ""
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stt = SpeechToText()
    while True:
        transcription = stt.listen()
        if transcription:
            print(f"Transcribed Text: {transcription}")