import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class CommerceAnalysisService:
    def __init__(self):
        self.gms_api_key = os.getenv("GMS_API_KEY")
        self.model = None
        
        print("\n" + "="*80)
        print("CommerceAnalysisService 초기화")
        print("="*80)
        print(f"GMS_API_KEY 존재 여부: {self.gms_api_key is not None}")
        if self.gms_api_key:
            print(f"GMS_API_KEY 길이: {len(self.gms_api_key)}")
        
        # GMS API 키가 있으면 LangChain 모델 초기화
        if self.gms_api_key:
            # LangChain에서 GMS를 사용하기 위한 환경변수 설정
            os.environ["OPENAI_API_KEY"] = self.gms_api_key
            os.environ["OPENAI_API_BASE"] = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
            
            print("LangChain 모델 초기화 시도 중...")
            try:
                # gpt-4o 사용
                self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
                print("✓ LangChain 모델 초기화 성공 (gpt-4o-mini)")
            except Exception as e:
                print(f"✗ LangChain 모델 초기화 실패: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
        else:
            print("⚠ GMS_API_KEY가 없어 기본 레포트만 사용합니다.")
        print("="*80 + "\n")
        
    async def generate_report(
        self, 
        lat: float, 
        lng: float, 
        radius: int, 
        commerce_info: Dict[str, Any]
    ) -> str:
        """
        주변 상권 정보를 기반으로 AI 분석 레포트 생성 (LangChain 사용)
        """
        print("\n" + "="*80)
        print("generate_report 호출됨")
        print("="*80)
        print(f"위도: {lat}, 경도: {lng}, 반경: {radius}m")
        print(f"상권 정보: {commerce_info}")
        print(f"GMS_API_KEY 존재: {self.gms_api_key is not None}")
        print(f"모델 존재: {self.model is not None}")
        
        if not self.gms_api_key or not self.model:
            print("⚠ LLM을 사용할 수 없어 기본 레포트를 생성합니다.")
            print("="*80 + "\n")
            return self._generate_default_report(commerce_info, radius)
        
        try:
            # 프롬프트 생성
            prompt = self._build_prompt(commerce_info, radius)
            
            print("\n" + "-"*80)
            print("생성된 프롬프트:")
            print("-"*80)
            print(prompt)
            print("-"*80)
            
            print("\nLLM 호출 시작...")
            import sys
            sys.stdout.flush()  # 버퍼 강제 플러시
            
            # LangChain을 사용한 GMS API 호출
            # invoke는 동기 메서드이므로, 비동기 환경에서는 run_in_executor 사용
            import asyncio
            loop = asyncio.get_event_loop()
            
            print("  → run_in_executor 실행 중... (LLM 응답 대기 중, 시간이 걸릴 수 있습니다)")
            sys.stdout.flush()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.invoke(prompt)
            )
            
            print("✓ LLM 응답 수신 완료")
            sys.stdout.flush()
            print(f"응답 타입: {type(response)}")
            print(f"응답 내용 (원본): {response}")
            
            # LangChain의 응답은 보통 content 속성을 가짐
            report_content = None
            if hasattr(response, 'content'):
                report_content = response.content
                print(f"응답.content 속성 사용: {report_content}")
            elif isinstance(response, str):
                report_content = response
                print(f"응답이 문자열 타입: {report_content}")
            else:
                report_content = str(response)
                print(f"응답을 문자열로 변환: {report_content}")
            
            # LLM 생성 내용을 콘솔에 출력
            print("\n" + "="*80)
            print("LLM 생성 레포트 (최종):")
            print("="*80)
            print(report_content)
            print("="*80 + "\n")
            
            return report_content
                    
        except Exception as e:
            print(f"\n✗ LangChain 레포트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            print("기본 레포트로 대체합니다.\n")
            return self._generate_default_report(commerce_info, radius)
    
    def _build_prompt(self, commerce_info: Dict[str, Any], radius: int) -> str:
        """프롬프트 생성"""
        prompt = f"""다음은 반경 {radius}m 내의 주변 상권 정보입니다:

- 편의점: {commerce_info.get('convenienceStore', 0)}개
- 카페: {commerce_info.get('cafe', 0)}개
- 마트: {commerce_info.get('mart', 0)}개
- 음식점: {commerce_info.get('restaurant', 0)}개
- 약국: {commerce_info.get('pharmacy', 0)}개
- 은행: {commerce_info.get('bank', 0)}개
- 병원: {commerce_info.get('hospital', 0)}개
- 지하철역: {commerce_info.get('subway', 0)}개

위 상권 정보를 바탕으로 이 지역의 상권 특징, 생활 편의성, 거주 적합성에 대해 상세하고 전문적인 분석을 제공해주세요.

다음 형식으로 작성해주세요:

### 상권 분석 결과

(이 지역의 상권 특징, 생활 편의성, 거주 적합성에 대한 상세한 분석을 500-800자 정도로 작성해주세요. 각 시설의 개수와 분포를 고려하여 구체적이고 실용적인 인사이트를 제공해주세요. 예를 들어, 편의점이 많으면 야간 생활의 편의성, 카페가 많으면 주거 환경의 쾌적함, 지하철역이 있으면 교통 접근성 등을 구체적으로 설명해주세요.)

분석은 다음 항목을 포함해주세요:
1. 상권 밀도 및 특징
2. 생활 편의성 평가
3. 교통 접근성
4. 거주 적합성 종합 평가

마크다운 형식으로 작성해주시고, 자연스럽고 읽기 쉬운 문체로 작성해주세요."""
        
        return f"당신은 부동산 상권 분석 전문가입니다. 주변 상권 정보를 분석하여 거주자에게 유용하고 상세한 인사이트를 제공합니다.\n\n{prompt}"
    
    def _generate_default_report(self, commerce_info: Dict[str, Any], radius: int) -> str:
        """기본 레포트 생성"""
        if not commerce_info:
            return f"반경 {radius}m 내의 주변 상권 정보를 조회할 수 없습니다."
        
        total_count = sum([
            commerce_info.get('convenienceStore', 0),
            commerce_info.get('cafe', 0),
            commerce_info.get('mart', 0),
            commerce_info.get('restaurant', 0),
            commerce_info.get('pharmacy', 0),
            commerce_info.get('bank', 0),
            commerce_info.get('hospital', 0),
            commerce_info.get('subway', 0)
        ])
        
        report_parts = [f"반경 {radius}m 내의 주변 상권 분석 결과입니다:\n\n"]
        report_parts.append(f"총 {total_count}개의 시설이 확인되었습니다. ")
        
        # 주요 시설 강조
        subway = commerce_info.get('subway', 0)
        restaurant = commerce_info.get('restaurant', 0)
        cafe = commerce_info.get('cafe', 0)
        convenience_store = commerce_info.get('convenienceStore', 0)
        
        if subway > 0:
            report_parts.append("지하철역이 접근 가능한 위치에 있어 교통이 편리합니다. ")
        if restaurant > 5:
            report_parts.append("다양한 음식점이 있어 식사 옵션이 풍부합니다. ")
        elif restaurant > 0:
            report_parts.append("음식점이 있어 기본적인 식사는 가능합니다. ")
        if cafe > 3:
            report_parts.append("카페가 많아 생활이 편리합니다. ")
        if convenience_store > 2:
            report_parts.append("편의점이 여러 곳 있어 일상생활이 편리합니다.")
        
        report_parts.append("\n\n※ 더 상세한 AI 분석 레포트를 보시려면 유효한 GMS_API_KEY를 설정해주세요.")
        
        return "".join(report_parts)
