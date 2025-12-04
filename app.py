# %cd /content/
import sys
import os
chatterbox=f"{os.getcwd()}/chatterbox/src"
sys.path.append(chatterbox)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import tempfile
import random
import numpy as np
import torch

from sentencex import segment
import re
from tqdm.auto import tqdm
import os
import shutil
import soundfile as sf
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
temp_audio_dir="./cloned_voices"
os.makedirs(temp_audio_dir, exist_ok=True)




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.

    This tool synthesizes natural-sounding speech from input text. When a reference audio file
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")

    # Handle optional audio prompt
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")

    wav = current_model.generate(
        text_input,  #max 300 chars
        language_id=language_id,
        **generate_kwargs
    )
    print("Audio generation complete.")
    return current_model.sr, wav.squeeze(0).numpy()



supported_languages = {
    "English": "en",
    "Hindi": "hi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Danish": "da",
    "Dutch": "nl",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Italian": "it",
    "Malay": "ms",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Turkish": "tr"
}
def word_split(text, char_limit=300):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + (1 if current_chunk else 0) <= char_limit:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_into_chunks(text,lang_code, max_char_limit=300):
    global supported_languages
    if len(text)>=300:
      print("⚠️ The text is too long. Breaking it into smaller pieces so the voice generation works correctly.")
      raw_sentences = list(segment(lang_code, text))

      # Flattened list of sentence-level word chunks
      sentence_chunks = []
      for sen in raw_sentences:
          sentence_chunks.extend(word_split(sen, char_limit=max_char_limit))

      chunks = []
      temp_str = ""

      for sentence in sentence_chunks:
          if len(temp_str) + len(sentence) + (1 if temp_str else 0) <= max_char_limit:
              temp_str += (" " if temp_str else "") + sentence
          else:
              chunks.append(temp_str)
              temp_str = sentence

      if temp_str:
          chunks.append(temp_str)

      return chunks
    else:
      return [text]


def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "—": " ",  #
        "-": " ",  # Replace hyphen with space
        "**": " ", # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous symbols and pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and map symbols
        r'[\U0001F700-\U0001F77F]|'  # Alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric shapes extended
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental symbols and pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and pictographs extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U0001F1E0-\U0001F1FF]'   # Flags (iOS)
        r'', flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tts_file_name(text, language="en"):
    global temp_audio_dir
    # Clean and process the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip().replace(" ", "_")

    # Ensure the text is not empty
    if not text:
        text = "audio"

    # Truncate to first 20 characters for filename
    truncated_text = text[:20]

    # Sanitize and format the language tag
    language = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"

    # Generate random suffix
    random_string = uuid.uuid4().hex[:8].upper()

    # Construct the filename
    file_name = f"{temp_audio_dir}/{truncated_text}_{language}_{random_string}.wav"
    return file_name
def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence)
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path

def clone_voice( text,
                audio_prompt_path_input,
    lang_name="English",
    exaggeration_input= 0.5,
    temperature_input= 0.8,
    seed_num_input = 0,
    cfgw_input= 0.5):
    global supported_languages
    language_id=supported_languages.get(lang_name,"en")
    text = clean_text(text)
    chunks = split_into_chunks(text,language_id, max_char_limit=300)
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    temp_files = []
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Generating audio"):

      # print(len(chunk))
      # print(chunk)
      # print(chunk_path)
      try:
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03}.wav")
        sr, audio =generate_tts_audio(
            chunk,
            language_id,
            audio_prompt_path_input,
            exaggeration_input,
            temperature_input,
            seed_num_input,
            cfgw_input
        )
        sf.write(chunk_path, audio, sr)
        print(sr)
        temp_files.append(chunk_path)
      except Exception as e:
        print(f"⚠️ [Chunk {idx}] Generation failed: {e}")
        print(f"Text: {chunk}")
        print(f"Length: {len(chunk)}")
        continue  # Skip failed chunk
            # Merge all valid chunks
    final_audio = []
    for file_path in temp_files:
      try:
        data, _ = sf.read(file_path)
        final_audio.append(data)
      except Exception as e:
        print(f"💀 [Merging] Failed to read chunk: {file_path} ({e})")
    final_path=None
    if final_audio:
      final_audio = np.concatenate(final_audio)
      final_path = tts_file_name(text,language_id)
      sf.write(final_path, final_audio, sr)
    else:
      raise RuntimeError("All audio chunk generations failed.")
    shutil.rmtree(temp_dir)
    return final_path

import torch
import torchaudio
import numpy as np
import soundfile as sf

def remove_noise_high_quality(audio_path,
                              threshold=0.5,
                              min_speech_duration_ms=150,
                              min_silence_duration_ms=100,
                              speech_pad_ms=30,
                              max_gap=1.5,
                              natural_pause=0.02):
    """
    High-quality noise removal:
    - Keeps original sample rate & channels
    - Uses VAD on temporary 16kHz mono copy
    - Collapses long gaps (>max_gap) into short natural pause
    - Preserves original audio clarity
    """
    # Load original audio (full quality)
    orig_audio, orig_sr = torchaudio.load(audio_path)  # stereo or mono preserved
    orig_audio = orig_audio.clone()  # keep original

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_audio = orig_audio.to(device)



    # --------------------------
    # Step 1: Prepare VAD input (16kHz mono)
    # --------------------------
    wav_vad = torch.mean(orig_audio, dim=0, keepdim=True)  # mono

    if orig_sr != 16000:
        wav_vad = wav_vad.cpu()  # move to CPU before resample
        wav_vad = torchaudio.transforms.Resample(orig_sr, 16000)(wav_vad)  # resample
        
    wav_vad = wav_vad.to(device)  # move back to GPU for VAD model
    # --------------------------
    # Step 2: Load Silero-VAD
    # --------------------------
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
    model = model.to(device)

    # Detect speech on VAD audio
    speech_timestamps_vad = get_speech_timestamps(
        wav_vad, model, sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )

    if not speech_timestamps_vad:
        print("No speech detected!")
        output_path = audio_path.replace(".wav", "_remove_noise.wav")
        sf.write(output_path, orig_audio.squeeze().cpu().numpy(), orig_sr)
        return output_path

    # --------------------------
    # Step 3: Map timestamps back to original sample rate
    # --------------------------
    ratio = orig_sr / 16000  # scale factor
    speech_timestamps = []
    for seg in speech_timestamps_vad:
        start = int(seg['start'] * ratio)
        end = int(seg['end'] * ratio)
        speech_timestamps.append({'start': start, 'end': end})

    # --------------------------
    # Step 4: Build cleaned audio
    # --------------------------
    result_audio = []
    gap_list = []

    start_trim = speech_timestamps[0]['start']
    end_trim = speech_timestamps[-1]['end']
    trimmed_audio = orig_audio[:, start_trim:end_trim]

    for i in range(len(speech_timestamps)):
        seg_start = speech_timestamps[i]['start'] - start_trim
        seg_end = speech_timestamps[i]['end'] - start_trim
        result_audio.append(trimmed_audio[:, seg_start:seg_end])

        if i < len(speech_timestamps) - 1:
            next_seg_start = speech_timestamps[i + 1]['start'] - start_trim
            gap_samples = next_seg_start - seg_end

            if gap_samples > 0:
                if gap_samples > int(max_gap * orig_sr):
                    pause_samples = int(natural_pause * orig_sr)
                    result_audio.append(torch.zeros((orig_audio.shape[0], pause_samples), device=device))
                    gap_list.append(pause_samples)
                else:
                    result_audio.append(torch.zeros((orig_audio.shape[0], gap_samples), device=device))
                    gap_list.append(gap_samples)
            else:
                gap_list.append(0)

    clean_audio = torch.cat(result_audio, dim=1)

    # --------------------------
    # Step 5: Save high-quality
    # --------------------------
    output_path = audio_path.replace(".wav", "_remove_noise.wav")
    sf.write(output_path, clean_audio.squeeze().cpu().numpy(), orig_sr, subtype='PCM_24')

  
    cursor = 0
    for i, seg in enumerate(speech_timestamps):
        seg_len = seg['end'] - seg['start']
        cursor += seg_len
        if i < len(speech_timestamps) - 1:
            cursor += gap_list[i]

    return output_path

# -----------------------------
# Example usage
# -----------------------------
# audio_file = "/content/1.wav"
# cleaned_file = remove_noise_high_quality(audio_file,
#                                         threshold=0.5,
#                                         min_speech_duration_ms=150,
#                                         min_silence_duration_ms=100,
#                                         speech_pad_ms=30,
#                                         max_gap=0.9 ,
#                                         natural_pause= 0.2)
# cleaned_file
def clone_voice_streaming(
    text,
    audio_prompt_path_input,
    lang_name="English",
    exaggeration_input=0.5,
    temperature_input=0.8,
    seed_num_input=0,
    cfgw_input=0.5,
    stereo=False,
    remove_silence=False,
    remove_noise=True,
):
    if not os.path.exists(audio_prompt_path_input):
      print("⚠️ Reference Audio File Not Found")
      print(audio_prompt_path_input)
      return None
    if seed_num_input == 0:
        seed_num_input = random.randint(1, 999999)
        print(f"🔑 Auto-generated seed: {seed_num_input}")
    language_id = supported_languages.get(lang_name, "en")
    text = clean_text(text)
    chunks = split_into_chunks(text, language_id, max_char_limit=300)

    final_path = tts_file_name(text, language_id)
    samplerate = 24000  # fixed
    channels = 2 if stereo else 1

    # Open final file for writing, append each chunk
    with sf.SoundFile(final_path, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as f:
        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Generating audio"):
            try:
                sr, audio = generate_tts_audio(
                    chunk,
                    language_id,
                    audio_prompt_path_input,
                    exaggeration_input,
                    temperature_input,
                    seed_num_input,
                    cfgw_input
                )

                # Convert to 2D array if necessary
                if audio.ndim == 1:
                    if stereo:
                        audio = np.stack([audio, audio], axis=1)  # duplicate channel
                    else:
                        audio = audio[:, None]  # mono 2D array

                f.write(audio)
            except Exception as e:
                print(f"⚠️ [Chunk {idx}] Generation failed: {e}")
                continue
    
    if remove_silence:
      final_path=remove_silence_function(final_path,minimum_silence=50)
    if remove_noise:
      final_path= remove_noise_high_quality(final_path,
                                            threshold=0.5,
                                            min_speech_duration_ms=150,
                                            min_silence_duration_ms=100,
                                            speech_pad_ms=30,
                                            max_gap=0.9 ,
                                            natural_pause= 0.2)
    return final_path    


# text = "Elias Thorne lived a life defined by precise lines. As the unofficial, self-appointed cartographer of Port Blossom—a tiny, grey-stone village clinging to the cliff face of the Cornish coast—his days were spent tracing the shifting boundaries of the familiar. His maps were not for navigation, but for contemplation: the subtle drift of the shingle beach after a winter storm, the forgotten network of Roman foundations beneath the old church, the precise length of Mrs. Gable’s prize-winning marrow."  # @param {type: "string"}
# reference_voice = '/content/test.mp3'  # @param {type: "string"}

# output_path=clone_voice_streaming(
#     text,
#     reference_voice,
#     lang_name="English",
#     exaggeration_input=0.5,
#     temperature_input=0.8,
#     seed_num_input=0,
#     cfgw_input=0.5,
#     stereo=False,
#     remove_silence=False,
# )
# from IPython.display import clear_output
# clear_output()
# print(output_path)

# from google.colab import files
# files.download(output_path)






def tts_only(
              text,
              audio_prompt_path_input,
              lang_name="English",
              exaggeration_input=0.5,
              temperature_input=0.8,
              seed_num_input=0,
              cfgw_input=0.5,
              remove_silence=False,
              stereo=False,
          ):
  audio_path=clone_voice_streaming(
      text,
      audio_prompt_path_input,
      lang_name,
      exaggeration_input,
      temperature_input,
      seed_num_input,
      cfgw_input,
      stereo,
      remove_silence,
  )

  return audio_path,audio_path



LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}
def default_audio_for_ui(lang_name):
  lang_code=supported_languages[lang_name]
  return LANGUAGE_CONFIG[lang_code]['audio']


def default_text_for_ui(lang_name):
  lang_code=supported_languages[lang_name]
  return LANGUAGE_CONFIG.get(lang_code, {}).get("text", "")

def resolve_audio_prompt(language_name: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    language_id=supported_languages[language_name]
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")
import gradio as gr
def tts_ui():
  custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
  with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
      gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Chatterbox Multilingual </h1>
        </div>""")
      with gr.Row():
          with gr.Column():
              initial_lang = "English"
              text = gr.Textbox(
                  value=default_text_for_ui(initial_lang),
                  label="Text to synthesize (No Limit)",
                  max_lines=5
              )

              language_id = gr.Dropdown(
                  choices=list(supported_languages.keys()),
                  value=initial_lang,
                  label="Language",
                  info="Select the language for text-to-speech synthesis"
              )

              ref_wav = gr.Audio(
                  sources=["upload", "microphone"],
                  type="filepath",
                  label="Reference Audio File (Optional)",
                  value=default_audio_for_ui(initial_lang)
              )
              run_btn = gr.Button("Generate", variant="primary")
              with gr.Row():
                Remove_Silence_button = gr.Checkbox(label="Remove Silence", value=False)
              with gr.Accordion("More options", open=False):
                gr.Markdown(
                    "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                    elem_classes=["audio-note"]
                )

                exaggeration = gr.Slider(
                    0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
                )
                cfg_weight = gr.Slider(
                    0.2, 1, step=.05, label="CFG/Pace", value=0.5
                )


                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)


          with gr.Column():
              audio_output = gr.Audio(label="Play Audio")
              audio_file = gr.File(label="Download Audio")
          def on_language_change(lang, current_ref, current_text):
              return default_audio_for_ui(lang), default_text_for_ui(lang)

          language_id.change(
              fn=on_language_change,
              inputs=[language_id, ref_wav, text],
              outputs=[ref_wav, text],
              show_progress=False
          )

      run_btn.click(
          fn=tts_only,
          inputs=[
              text,
              ref_wav,
              language_id,
              exaggeration,
              temp,
              seed_num,
              cfg_weight,
              Remove_Silence_button,

          ],
          outputs=[audio_output,
                   audio_file],
      )
  return demo
# demo=tts_ui()
# demo.launch(share=True,debug=True)



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo=tts_ui()
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()    

