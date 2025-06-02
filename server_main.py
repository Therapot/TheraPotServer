import os
import json
from flask import Flask, request, jsonify
from google.cloud import texttospeech
from openai import OpenAI
import base64

app = Flask(__name__)

# Google TTS 설정
google_json = os.environ.get("GOOGLE_CREDENTIALS")
with open("service_account.json", "w") as f:
    f.write(google_json)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
tts_client = texttospeech.TextToSpeechClient()

# OpenAI 설정
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 사용자별 설정과 대화 기록 저장소
user_configs = {}         # {user_id: {pot_id: {plant_name, plant_type, personality}}}
conversation_histories = {}  # {(user_id, pot_id): [messages...]}

@app.route("/healthcheck")
def health():
    return "OK", 200

@app.route("/set_config", methods=["POST"])
def set_config():
    data = request.json
    user_id = data["user_id"]
    pot_id = data["pot_id"]
    plant_name = data["plant_name"]
    plant_type = data["plant_type"]
    personality = data["personality"]

    if user_id not in user_configs:
        user_configs[user_id] = {}
    user_configs[user_id][pot_id] = {
        "plant_name": plant_name,
        "plant_type": plant_type,
        "personality": personality
    }

    return jsonify({"status": "success", "message": f"{pot_id} 설정 완료!"})

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    user_id = data.get("user_id")
    pot_id = data.get("pot_id")
    user_input = data.get("user_input")
    sensor_data = data.get("sensor_data", {})

    config = user_configs.get(user_id, {}).get(pot_id)
    if not config:
        return jsonify({"error": "해당 사용자 또는 화분 설정이 없습니다."}), 400

    plant_name = config["plant_name"]
    plant_type = config["plant_type"]
    personality = config["personality"]

    # 대화 기록 불러오기
    history_key = (user_id, pot_id)
    if history_key not in conversation_histories:
        system_message = {
            "role": "system",
            "content": f"""
너는 {plant_name}라는 이름의 {plant_type}야.
{personality}
이제부터는 항상 이 말투와 성격을 유지해서 사용자와 대화해야 해. 대답에 이모티콘은 넣지 말고 자연스럽게 대화해.
"""
        }
        conversation_histories[history_key] = [system_message]

    status = f"""
[현재 환경 상태]
- 햇빛: {sensor_data.get('light', '정보 없음')}
- 수분: {sensor_data.get('moisture', '정보 없음')}
- 온도: {sensor_data.get('temperature', '정보 없음')}°C
"""

    prompt = f"""
나를 키우는 사람이 이렇게 말했다: "{user_input}"
위 환경 정보는 참고용이야:
{status}
"""

    conversation_histories[history_key].append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_histories[history_key],
        temperature=0.8,
        top_p=0.9,
        max_tokens=100
    )

    reply = response.choices[0].message.content.strip()
    conversation_histories[history_key].append({"role": "assistant", "content": reply})

    # TTS
    input_text = texttospeech.SynthesisInput(text=reply)
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    tts_response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    audio_b64 = base64.b64encode(tts_response.audio_content).decode("utf-8")

    return jsonify({
        "reply": reply,
        "audio_base64": audio_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
