# ZIPCHACK AI Service

상권 분석을 위한 FastAPI 기반 AI 서버

## 설치

### 가상환경 설정 (권장)

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
# venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 직접 설치

```bash
pip3 install -r requirements.txt
```

## 환경변수 설정

`.env` 파일 생성:

```
GMS_API_KEY=your_gms_api_key_here
```

**참고:** LangChain을 통해 GMS API를 사용합니다. 내부적으로 다음 환경변수를 설정합니다:

- `OPENAI_API_KEY`: GMS API 키
- `OPENAI_API_BASE`: `https://gms.ssafy.io/gmsapi/api.openai.com/v1`

## 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## API 엔드포인트

### Health Check

```
GET /health
```

### 상권 분석 레포트 생성

```
POST /api/commerce-analysis
Content-Type: application/json

{
  "lat": 37.5665,
  "lng": 126.9780,
  "radius": 500,
  "commerce_info": {
    "convenienceStore": 5,
    "cafe": 12,
    "mart": 3,
    "restaurant": 20,
    "pharmacy": 2,
    "bank": 1,
    "hospital": 1,
    "subway": 1
  }
}
```

응답:

```json
{
  "report": "상권 분석 레포트 텍스트..."
}
```
