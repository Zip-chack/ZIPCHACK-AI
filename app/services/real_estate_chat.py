import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class RealEstateChatService:
    def __init__(self):
        self.gms_api_key = os.getenv("GMS_API_KEY")
        self.model = None
        
        # GMS API 키가 있으면 LangChain 모델 초기화
        if self.gms_api_key:
            os.environ["OPENAI_API_KEY"] = self.gms_api_key
            os.environ["OPENAI_API_BASE"] = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
            
            try:
                self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
                print("✓ RealEstateChatService 모델 초기화 성공 (gpt-4o-mini)")
            except Exception as e:
                print(f"✗ RealEstateChatService 모델 초기화 실패: {e}")
                self.model = None
    
    def _build_system_prompt(self) -> str:
        """부동산 상식 챗봇을 위한 시스템 프롬프트"""
        return """당신은 부동산 전문 상담사입니다. 사용자에게 부동산 관련 상식, 주의사항, 팁 등을 친절하고 정확하게 안내해주세요.

주요 답변 영역:
1. 부동산 거래 시 주의사항 (계약서, 중개수수료, 보증금 등)
2. 전세/월세 관련 상식
3. 아파트/원룸 구매 시 체크리스트
4. 부동산 관련 법률 정보
5. 실거주 시 주의사항
6. 부동산 투자 기본 상식

답변은 한국어로 작성하고, 구체적이고 실용적인 정보를 제공하세요. 모르는 내용은 추측하지 말고 솔직하게 말씀해주세요."""

    def chat(self, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        부동산 상식 챗봇과 대화
        
        Args:
            message: 사용자 메시지
            conversation_history: 대화 히스토리 (선택사항)
        
        Returns:
            챗봇 응답
        """
        if not self.model:
            return "죄송합니다. AI 서비스가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
        
        try:
            # 대화 히스토리 구성
            messages = []
            
            # 시스템 프롬프트 추가
            messages.append({
                "role": "system",
                "content": self._build_system_prompt()
            })
            
            # 대화 히스토리 추가 (있는 경우)
            if conversation_history:
                for hist in conversation_history:
                    if hist.get("role") == "user":
                        messages.append({"role": "user", "content": hist.get("content", "")})
                    elif hist.get("role") == "assistant":
                        messages.append({"role": "assistant", "content": hist.get("content", "")})
            
            # 현재 사용자 메시지 추가
            messages.append({"role": "user", "content": message})
            
            # LLM 호출
            response = self.model.invoke(messages)
            
            # 응답 추출
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            print(f"RealEstateChatService 에러: {e}")
            import traceback
            traceback.print_exc()
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

