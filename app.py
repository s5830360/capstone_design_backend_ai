import os
import uuid
import shutil
import subprocess
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

app = FastAPI()


############################################################
# 1) WavLM 음성 감정 인식 모델 로드
############################################################
WAVLM_DIR = "final_model"
wavlm_model = AutoModelForAudioClassification.from_pretrained(WAVLM_DIR)
wavlm_processor = AutoFeatureExtractor.from_pretrained(WAVLM_DIR)
wavlm_model.eval()
wavlm_labels = list(wavlm_model.config.id2label.values())

device = "cpu"


############################################################
# 2) 환경음 감정 CNN 모델 구조
############################################################
class EnvCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(11776, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


############################################################
# 3) 환경음 모델 로드
############################################################
ENV_MODEL_PATH = "env_cnn_emotion6_final_v1.pth"
env_model = EnvCNN(num_classes=10)
checkpoint = torch.load(ENV_MODEL_PATH, map_location=device)
env_model.load_state_dict(checkpoint["model_state_dict"])
env_model.eval()


############################################################
# 4) Silero VAD 로드
############################################################
print("🔊 Loading Silero VAD JIT...")
SILERO_PATH = "silero_vad/data/silero_vad.jit"
model_vad = torch.jit.load(SILERO_PATH)
model_vad.eval()

from pathlib import Path
import sys
sys.path.append(str(Path("src/silero_vad").absolute()))
from silero_vad.utils_vad import get_speech_timestamps


def is_voice_present_silero(audio, sr=16000):
    """Silero 기반 VAD"""
    audio_t = torch.tensor(audio, dtype=torch.float32)
    speech_ts = get_speech_timestamps(audio_t, model_vad, sampling_rate=sr)
    return len(speech_ts) > 0


############################################################
# 5) Scene → Emotion 매핑 (긍정/중립 중심)
############################################################
SCENES = [
    "airport", "bus", "bus_stop", "metro", "metro_station",
    "park", "public_square", "shopping_mall", "street", "tram"
]

SCENE_TO_EMOTION = {
    "airport": "happy",
    "bus": "neutral",
    "bus_stop": "sadness",
    "metro": "fear",
    "metro_station": "fear",
    "park": "happy",
    "public_square": "neutral",
    "shopping_mall": "happy",
    "street": "happy",
    "tram": "anger",
    "unknown_noise": "surprise",
}

PREDICTION_THRESHOLD = 50.0  # 70 → 50 완화


############################################################
# 6) 학습과 동일한 환경음 전처리
############################################################
def extract_melspectrogram(
        y,
        sr=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        target_length_sec=3
):
    # 1) 길이 맞추기
    target_samples = int(target_length_sec * sr)

    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        pad_len = target_samples - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")

    # 2) mel 생성
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # 3) dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 4) 정규화(학습 동일)
    mean = mel_db.mean()
    std = mel_db.std()
    mel_norm = (mel_db - mean) / (std + 1e-6)

    # 5) shape 맞추기 (1,1,64,T)
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return mel_tensor


############################################################
# 7) ffmpeg 변환
############################################################
def resolve_ffmpeg():
    # 1) OS PATH 검색
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2) Windows fallback 경로
    windows_ffmpeg = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(windows_ffmpeg):
        return windows_ffmpeg

    # 3) Linux fallback (Railway)
    linux_ffmpeg = "/usr/bin/ffmpeg"
    if os.path.exists(linux_ffmpeg):
        return linux_ffmpeg

    raise FileNotFoundError("ffmpeg not found")

def convert_to_wav(input_path, output_path):
    ffmpeg = resolve_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


############################################################
# 🎯 8) 최종 감정 분석 API
############################################################
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            uid = str(uuid.uuid4())
            raw_path = os.path.join(tmpdir, f"{uid}_raw")
            wav_path = os.path.join(tmpdir, f"{uid}.wav")

            # 파일 저장
            contents = await file.read()
            with open(raw_path, "wb") as f:
                f.write(contents)

            # 변환
            convert_to_wav(raw_path, wav_path)

            # 오디오 로드
            audio, _ = librosa.load(wav_path, sr=16000, mono=True)

            # -----------------------------
            # 1) 음성 여부 (VAD)
            # -----------------------------
            has_voice = is_voice_present_silero(audio)

            # -----------------------------
            # 2) WavLM 음성 감정
            # -----------------------------
            wavlm_emotion = None
            wavlm_conf = 0.0

            if has_voice:
                inputs = wavlm_processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=16000 * 4,
                )

                with torch.no_grad():
                    logits = wavlm_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[0]
                    pred_id = int(torch.argmax(probs))
                    wavlm_conf = float(probs[pred_id] * 100)
                    wavlm_emotion = wavlm_labels[pred_id]

            # -----------------------------
            # 3) 환경음 감정
            # -----------------------------
            mel_tensor = extract_melspectrogram(audio)

            with torch.no_grad():
                scene_logits = env_model(mel_tensor)
                scene_probs = torch.softmax(scene_logits, dim=1)[0]

            scene_id = int(torch.argmax(scene_probs))
            scene_conf = float(scene_probs[scene_id] * 100)
            scene_label = SCENES[scene_id]

            # threshold 적용
            if scene_conf < PREDICTION_THRESHOLD:
                scene_label = "unknown_noise"

            env_emotion = SCENE_TO_EMOTION.get(scene_label, "neutral")

            # -----------------------------
            # 4) 최종 감정 결정
            # -----------------------------
            if has_voice:
                final_emotion = wavlm_emotion
                final_conf = wavlm_conf
            else:
                final_emotion = env_emotion
                final_conf = scene_conf

            # -----------------------------
            # 5) 반환
            # -----------------------------
            return {
                "final_emotion": final_emotion,
                "final_confidence": final_conf,
                "voice_detected": bool(has_voice),

                "voice_emotion": wavlm_emotion,
                "voice_confidence": wavlm_conf,

                "scene": scene_label,
                "scene_emotion": env_emotion,
                "scene_confidence": scene_conf
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
