"""
customer_classification.py
고객 질문을 분석하여 금융 지식 수준을 분류하는 모듈

Classes
-------
CustomerClassifier
    고객 질문을 기반으로 금융 지식 수준을 분류하는 클래스

Methods
-------
classify(question)
    고객의 질문을 분석하고 금융 지식 수준을 반환
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

class CustomerClassifier:
    def __init__(self, model_name="gpt-4o-mini"):
        # LangChain LLM 초기화
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.8)

        # Few-shot prompting을 위한 예시 추가
        few_shot_examples = """
        [Newcomer] 
        - 질문: ETF가 무엇인가요?
        -> 원래 답변 : ETF는 특정지수 또는 가격의 수익률을 추종하는 인덱스 펀드입니다.
        -> 가공된 답변: ETF는 주식처럼 거래소에서 사고팔 수 있는 펀드입니다. 이 펀드는 주로 특정 시장 지수(예: 코스피나 나스닥 지수)의 움직임을 따라가도록 설계되어 있습니다. 쉽게 말해, ETF를 사면 그 지수를 추종하는 주식을 여러 개 한 번에 사는 것과 비슷합니다.

        - 질문: 기초지수가 뭔가요?
        -> 원래 답변 : ETF가 추종하고자 하는 지수를 말하며, 통상 비교지수, 벤치마크(Benchmark) 등으로도 불립니다. 투자자는 ETF의 수익률과 기초지수의 수익률을 비교해 봄으로써 ETF의 성과를 확인할 수 있습니다.
        -> 가공된 답변: 기초지수는 ETF가 따라가려고 하는 지수를 말합니다. 예를 들어, 코스피 지수를 따라가는 ETF는 코스피가 기초지수입니다. ETF는 이 지수와 비슷한 성과를 내려고 하기 때문에, 기초지수의 움직임이 ETF 성과에 큰 영향을 줍니다.

        [Learner] 
        - 질문: ETF는 배당을 언제 주나요?
        -> 가공된 답변: ETF가 투자하는 기초자산의 종류에 따라 각 ETF별로 배당정책이 결정됩니다. 통상, 국내주식형ETF의 경우 매월 말 또는 분기말을 기준으로 배당재원이 있는 경우 배당금을 지급합니다. 채권형ETF의 경우 보유하고 있는 채권의 이자지급시점을 고려하여 배당금이 지급되며, 통상 매 분기마다 1회 정기적으로 배당금이 지급됩니다.

        - 질문: ETF의 세금 계산은 어떻게 하나요?
        -> 가공된 답변: (1)국내주식형ETF vs 국내주식 직접 투자
                        국내주식 직접 투자 시 주식을 매도할 때 증권거래세 0.23%가 부과되지만, 국내주식형 ETF는 증권거래세가 면제됩니다. 따라서 ETF를 이용하면 거래 시 세금 부담이 줄어드는 장점이 있습니다.
                        (2)국내주식형ETF vs 국내주식형 펀드
                        두 가지 모두 펀드 내에서 주식을 매매할 때 증권거래세 0.03%가 부과됩니다. 그러나, 국내주식형 펀드는 매매회전율이 보통 100~200%로 높아 증권거래세 부담이 큽니다. 반면, 국내주식형 ETF는 매매회전율이 10% 미만으로 낮아 세금 부담이 상대적으로 적습니다.
                        (3)해외주식형ETF vs 해외주식 직접투자 
                        해외주식형 ETF에 투자하면 배당소득세 15.4%가 부과됩니다. 반면, 해외주식 직접 투자 시에는 양도소득세 22%를 신고하고 납부해야 합니다.
                        (4)해외주식형ETF vs 해외주식형펀드 
                        두 가지 모두 배당소득세 15.4%가 적용됩니다. 다만, 해외주식형 펀드는 세금 부과 기준 가격이 하루에 한 번 산출되어 당일 매수 후 매도해도 세금이 부과되지 않을 수 있습니다. 그러나 빈번한 매매는 평가손실의 위험을 동반하므로 주의가 필요합니다.

        [Investigator] 
        - 질문: TIGER 반도체TOP10레버리지 ETF는 환헤지인가요?
        -> 가공된 답변: 아니요, TIGER 반도체TOP10레버리지 ETF는 환헤지를 실시하지 않습니다. 따라서, 투자 시 환율 변동에 따른 영향을 받을 수 있습니다.

        - 질문: TIGER 미국AI빅테크10 ETF의 리밸런싱 주기가 어떻게 되나요?
        -> 가공된 답변: 매년 4회, 3월, 6월, 9월, 12월의 마지막 영업일에 리밸런싱을 실시합니다. 리밸런싱 후, 3영업일 이내에 새로 산출된 비중에 따라 종목 비율을 조정합니다.

        - 질문: TIGER반도체TOP10레버리지 ETF의 과세 기준이 무엇인가요?
        -> 가공된 답변: 매매차익: 배당소득세가 부과되며, 과표 증분이 미미한 경우, 매매차익에 대해 15.4%의 세율이 적용됩니다.
                        분배금: 배당소득세가 부과되며, 현금분배금과 과표 증분 중 적은 금액에 대해 15.4%의 세율이 적용됩니다.

        """

        # PromptTemplate 설정 (Few-shot examples 포함)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "rag_answer", "conversation_history", "level_percentage"],
            template=f"""
            다음은 고객의 질문과 그에 따른 답변입니다.

            당신은 아래 3가지 단계를 거칩니다.
            1. 당신은 고객의 질문을 기반으로 고객의 금융 지식 수준(Necomer/Learner/Investigator)을 분류합니다.
            [Newcomer]: ETF와 같은 기초적인 개념을 묻는 유형으로, 기본적인 용어, 개념 및 특징에 대한 질문을 하는 고객을 위해 최대한 쉽게 설명합니다.
            [Learner]: ETF의 여러 종류 및 파생상품에 대한 이해가 부족한 고객을 대상으로, 기본 용어 지식을 가지고 있다는 전제하에 보다 깊이 있는 답변을 제공합니다.
            [Investigator]: 구체적이고 정밀한 정보를 요구하는 유형으로, 세금, 환율 위험, 구체적 수치의 추적 오차, NAV 등 구체적인 수치 및 상품에 대한 정보를 제공합니다.
            2. 현재까지의 고객 금융 지식 수준에 맞춰 적절하게 RAG 답변을 가공합니다.
            3. 최종적으로 질문에 대한 가공된 답변과 고객의 금융 지식 수준을 도출하세요.

            고객 금융 지식 수준별 가공된 답변 예시는 다음과 같습니다.            
            {few_shot_examples}

            대화 히스토리:
            {{conversation_history}}

            현재 고객의 질문을 통해, 고객의 금융 지식 수준을 다음 중 하나로 분류하세요: [Newcomer, Learner, Investigator]
            답변 포맷은 다음과 같습니다. ("고객의 금융 지식 수준"으로 출력하며, 고객의 금융 지식 수준을 대괄호로 표시하세요.)
            고객의 금융 지식 수준: []

            질문: {{query}}
            RAG 답변: {{rag_answer}}

            이전까지 고객의 금융 지식 수준 비율은 다음과 같습니다: {{level_percentage}}
            현재 고객의 금융 지식 수준과 이전까지 고객의 금융 지식 수준의 비율을 모두 반영해서, 고객 질문에 대한 고객의 금융 지식 수준에 맞는 자세한 가공된 답변을 알려주세요.
            가공된 답변:    
            """
        )
        
        # LLMChain 생성
        self.chain = self.prompt_template | self.llm
        self.levels_count = {"Newcomer": 0, "Learner": 0, "Investigator": 0}
        
    def classify(self, query, rag_answer, chain):
        # 이전 대화 히스토리 생성
        conversation_history = "\n".join(
            [f"질문: {q['질문']}\n답변: {q['가공된 답변']}" for q in chain]
        )

        # 고객 수준 비율 계산
        total = sum(self.levels_count.values())
        if total > 0:
            level_percentage = {
                level: f"{(count / total) * 100:.1f}%" for level, count in self.levels_count.items()
            }
        else:
            level_percentage = {
                "Newcomer": "0%", "Learner": "0%", "Investigator": "0%"
            }

        # LLM이 결합된 Chain을 이용해 결과 생성
        result = self.chain.invoke({
            "query": query,
            "rag_answer": rag_answer,
            "conversation_history": conversation_history,
            "level_percentage": level_percentage
        })

        # result에서 텍스트를 추출하기 위한 코드
        if hasattr(result, 'content'):
            if isinstance(result.content, str):
                # content가 문자열인 경우
                result_text = result.content
            elif isinstance(result.content, dict):
                # content가 딕셔너리인 경우, 적절한 키를 통해 텍스트를 추출
                result_text = result.content.get('text', '')
            else:
                # 다른 타입인 경우 문자열로 변환
                result_text = str(result.content)
        elif isinstance(result, dict):
            # result가 딕셔너리인 경우
            result_text = result.get("content", "")
        else:
            # 그 외의 경우 문자열로 변환
            result_text = str(result)

        # LLMChain 결과에서 가공된 답변과 고객 수준 추출
        refined_answer, customer_level = self.extract_response(result_text)

        if customer_level in self.levels_count:
            self.levels_count[customer_level] += 1
        
        return {
            "refined_answer": refined_answer,
            "customer_level": customer_level
        }
    
    def extract_response(self, response):
        refined_answer = ""
        customer_level = ""

        # 가공된 답변 추출
        match_answer = re.search(r"가공된 답변\s*:\s*(.*?)(?:(고객의 금융 지식 수준\s*:)|$)", response, re.DOTALL)
        if match_answer:
            refined_answer = match_answer.group(1).strip()
        else:
            print("가공된 답변을 찾을 수 없습니다.")

        # 고객 금융 지식 수준 추출
        match_level = re.search(r"고객의 금융 지식 수준\s*:\s*(\w+)", response)
        if match_level:
            customer_level = match_level.group(1).strip()
        else:
            # 대괄호 안에 있는 금융 지식 수준을 찾습니다.
            match_bracket = re.search(r"\[(Newcomer|Learner|Investigator)\]", response)
            if match_bracket:
                customer_level = match_bracket.group(1).strip()
            else:
                print("고객의 금융 지식 수준을 찾을 수 없습니다.")

        return refined_answer, customer_level
    
    def determine_final_level(self):
        return max(self.levels_count, key=self.levels_count.get)