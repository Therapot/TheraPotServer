import os
import json
import base64
from flask import Flask, request, jsonify
from google.cloud import texttospeech
from openai import OpenAI  # ✅ 최신 방식
from dotenv import load_dotenv

# 환경변수 로드 (로컬 테스트용, Railway에선 무시됨)
load_dotenv()

# ✅ OpenAI 클라이언트 객체 생성 (API 키는 환경변수로 자동 인식)
client = OpenAI()

# Flask 앱 초기화
app = Flask(__name__)

# Google TTS 키 저장
google_json = os.environ.get("GOOGLE_CREDENTIALS")
with open("service_account.json", "w") as f:
    f.write(google_json)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Google TTS 클라이언트
tts_client = texttospeech.TextToSpeechClient()

# 대화 초기 설정
conversation_history = []
PLANT_NAME = "기붕이"
PLANT_TYPE = "물푸레나무"

system_message = f"""
너는 {PLANT_NAME}라는 이름의 {PLANT_TYPE}야.
너는 항상 따뜻하고 친근한 어조로, 때로는 격려와 위로를 주며 사용자와 감정적으로 교감하려고 노력해야 해.
너는 필요할 때는 사용자에게 질문도 하면서 대화를 이어가며, 상황에 맞는 진지하고 도움되는 조언을 적극적으로 해야 해.
하지만 이모티콘이나 특수 문자는 사용하지 말고, 자연스럽고 부드러운 문장으로만 대화해.
"""
conversation_history.append({"role": "system", "content": system_message})

# ✅ 헬스체크
@app.route("/healthcheck")
def health():
    return "OK", 200

# 🌱 메인 API: 대화 처리
@app.route('/process', methods=['POST'])
def process():
    data = request.json
    user_input = data.get('user_input')
    sensor_data = data.get('sensor_data')

    # 센서 상태를 포함한 프롬프트 구성
    status = f"""
    [나의 현재 상태]
    - 햇빛 (조도): {sensor_data['light']}
    - 수분 (습도): {sensor_data['moisture']}
    - 주변 온도: {sensor_data['temperature']}°C
    """
    prompt = f"""
    나는 {PLANT_NAME}라는 이름을 가진 {PLANT_TYPE}야.
    사용자가 현재 햇빛, 수분, 주변 온도에 대해 궁금해할 때만 다음 값을 바탕으로 대답해야 해.
    {status}

    나를 키우는 사람이 나에게 이렇게 말했다: "{user_input}"
    친근하고 자연스럽게 2-3문장 이내로 반응해줘.
    """

    conversation_history.append({"role": "user", "content": prompt})

    # ✅ 최신 OpenAI 방식으로 GPT 응답 생성
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        temperature=0.8,
        top_p=0.9,
        max_tokens=100
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})

    # ✅ Google TTS 변환
    input_text = texttospeech.SynthesisInput(text=reply)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    tts_response = tts_client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    # ✅ MP3를 base64로 변환
    audio_b64 = base64.b64encode(tts_response.audio_content).decode("utf-8")

    return jsonify({
        "reply": reply,
        "audio_base64": audio_b64
    })

# ✅ 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
