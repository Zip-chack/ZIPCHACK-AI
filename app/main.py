from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from app.services.commerce_analysis import CommerceAnalysisService
from app.services.real_estate_chat import RealEstateChatService

load_dotenv()

app = FastAPI(title="ZIPCHACK AI Service", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
commerce_analysis_service = CommerceAnalysisService()
real_estate_chat_service = RealEstateChatService()


class CommerceAnalysisRequest(BaseModel):
    lat: float
    lng: float
    radius: Optional[int] = 500
    commerce_info: dict


class CommerceAnalysisResponse(BaseModel):
    report: str


class RealEstateChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []


class RealEstateChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health_check():
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Health check 요청 수신")
    return {"status": "ok", "timestamp": timestamp}


@app.post("/api/commerce-analysis", response_model=CommerceAnalysisResponse)
async def analyze_commerce(request: CommerceAnalysisRequest):
    """
    주변 상권 정보를 기반으로 AI 분석 레포트 생성
    """
    import sys
    import datetime
    sys.stdout.flush()  # 버퍼 강제 플러시
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] " + "="*80)
    print(f"[{timestamp}] API 요청 수신: /api/commerce-analysis")
    print(f"[{timestamp}] " + "="*80)
    print(f"[{timestamp}] 요청 데이터: lat={request.lat}, lng={request.lng}, radius={request.radius}")
    print(f"[{timestamp}] 상권 정보: {request.commerce_info}")
    sys.stdout.flush()
    
    try:
        print("\n[1/3] generate_report 호출 시작...")
        sys.stdout.flush()
        
        report = await commerce_analysis_service.generate_report(
            lat=request.lat,
            lng=request.lng,
            radius=request.radius,
            commerce_info=request.commerce_info
        )
        
        print("\n[2/3] generate_report 완료")
        print(f"최종 반환 레포트 길이: {len(report)}자")
        sys.stdout.flush()
        
        print("\n[3/3] 응답 반환 중...")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        return CommerceAnalysisResponse(report=report)
    except Exception as e:
        print(f"\n✗ API 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/real-estate-chat", response_model=RealEstateChatResponse)
async def real_estate_chat(request: RealEstateChatRequest):
    """
    부동산 상식 챗봇과 대화
    """
    try:
        response = real_estate_chat_service.chat(
            message=request.message,
            conversation_history=request.conversation_history
        )
        return RealEstateChatResponse(response=response)
    except Exception as e:
        print(f"부동산 챗봇 API 에러: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
