# 예금자보호 법령 RAG 챗봇

예금자보호법과 시행령을 근거로 답하는 RAG 챗봇. LangGraph, pgvector, FastAPI를 쓴다.

- 문제 정의, 문서 범위, 진행 단계, 실행 방법: [README.md](README.md)
- 수집할 법령과 시행일: [data/sources.json](data/sources.json)
- 조문 파싱: [src/rag_chatbot/parsing.py](src/rag_chatbot/parsing.py)

## 지켜야 할 규칙

- 유료 모델 호출(임베딩, 생성) 전에 횟수와 이유를 밝히고 개발자의 답을 기다린다.
- 환경변수 파일(`.env`, `.env.*`)을 읽지 않는다.
- 테스트는 대표 사례만 쓴다. 지금 겪는 문제를 넘어서는 대비 장치나 쓰이지 않는 추상화를 넣지 않는다.
- LangChain 1.x에서 `langchain-classic`으로 옮겨진 레거시 API를 쓰지 않는다. 예: `langchain.retrievers`의 `EnsembleRetriever`, `create_history_aware_retriever` 같은 레거시 체인, 증분 색인 API. 하이브리드 결합(RRF)은 직접 쓴 함수로 한다.
- 평가: 확인용 문항은 마지막에 한 번만 평가한다. 확인용 문항을 보고 파이프라인을 고치지 않는다. 파이프라인을 바꾸면 개발용 문항을 다시 돌리고, 전에 맞던 문항이 틀려진 경우를 기록한다.
- 커밋 메시지는 AngularJS 커밋 컨벤션(`<type>(<scope>): <subject>`)을 따른다. type은 feat, fix, docs, style, refactor, perf, test, chore 중 하나다. 줄은 100자를 넘기지 않고, 공동 저자 줄은 붙이지 않는다. 커밋하기 전에 메시지를 개발자에게 보여 준다.
