# 연말정산 세금 Q&A RAG 시스템

PDF 기반 RAG 질의응답 시스템 구현 및 모델 평가 프로젝트

---

## 팀 구성

| 역할 | 이름 | 담당 |
|------|------|------|
| 팀장 | 이수홍 | 프로젝트 총괄(기획·관리·결과 도출), 발표 자료 제작, LLM 모델 비교 및 Evaluation 모델 작성 |
| 팀원 | 이수민 | Document Processing & Retrieval — PDF 로드, 문서 분할(청킹), 표 전처리, 벡터스토어 구축 |
| 팀원 | 이희영 | Semantic Search & Retrieval — 임베딩 모델 비교, 유사도 검색 알고리즘, 검색 품질 향상 |

※ 역할은 진행 상황에 따라 유동적으로 조정

---

## 주제

**국세청 연말정산 안내 문서 기반 세금 Q&A 챗봇**

국세청이 매년 공개하는 「연말정산 신고안내」 등 공식 PDF 문서를 지식 베이스로,
"월세 살면 얼마나 돌려받아요?" 같은 일상어 질문에 근거 조항과 함께 답변하는
RAG(Retrieval Augmented Generation) 질의응답 시스템을 구현하고,
LLM·임베딩·벡터스토어·검색 파라미터별 성능을 비교 평가한다.

---

## 주제 선정 이유

1. **일상어와 문서 용어의 불일치** — 사용자는 "집 월세 돌려받기"라고 묻지만 문서에는
   "월세액 세액공제"로 기재되어 있다. 키워드 검색으로는 찾기 어려운 구조라서
   **의미 기반 검색(Semantic Search)의 효과가 극명하게 드러나는** 도메인이다.
2. **명확한 정답 존재** — 공제 한도, 요건, 비율 등 답이 숫자와 조건으로 명확해서
   평가 데이터셋(질문·정답·근거 문장)을 체계적으로 구축할 수 있고 BERTScore 비교가 유의미하다.
3. **고품질 공개 데이터** — 국세청 공식 PDF가 매년 갱신·공개되며, 표·목차·조항 등
   문서 구조가 다양해 청킹/전처리 전략 실험에 적합하다.
4. **높은 공감대** — 연말정산은 모든 직장인의 관심사로, 시연 시 누구나 직접 질문을 던져볼 수 있다.

---

## 기능 구조

```
[국세청 연말정산 PDF]
        │  ① 문서 로드 (PyPDFLoader)
        ▼
[텍스트 추출·전처리]  ← 표 전처리, 페이지 메타데이터 부여
        │  ② 문서 분할 (chunk_size / overlap 실험)
        ▼
[문서 청크]
        │  ③ 임베딩 (한국어/다국어 임베딩 모델 비교)
        ▼
[벡터스토어]  ← FAISS / Chroma 비교
        │  ④ 의미 검색 (similarity / MMR, top_k 실험)
        ▼
[관련 문서 청크 검색]
        │  ⑤ LangChain RAG Chain (프롬프트 실험)
        ▼
[LLM 답변 생성]  ← HuggingFace 모델 / OpenAPI 모델 비교
        │  ⑥ 평가 (BERTScore, 응답시간, Hallucination)
        ▼
[답변 + 근거 조항 출력]
```

- 사용자 질문 → 의미 검색으로 관련 조항 검색 → 근거와 함께 답변 생성
- 답변에는 참조한 문서 페이지(근거)를 함께 표시하여 신뢰성 확보

---

## 실험 항목

| 변수 | 후보 |
|------|------|
| LLM 모델 | HuggingFace 공개 모델 2종 이상(예: Qwen, EXAONE, Llama 계열) + OpenAPI(ChatGPT, Claude, Upstage) |
| 임베딩 모델 | ko-sbert(한국어 전용), multilingual-e5(다국어), OpenAI Embedding |
| 벡터스토어 | FAISS, Chroma |
| 유사도 검색 방법 | Similarity Search, MMR(Maximal Marginal Relevance) |
| chunk_size | 300 / 500 / 1000 |
| overlap_size | 0 / 50 / 100 |
| top_k | 2 / 4 / 6 |
| 프롬프트 | 기본 QA / 근거 인용 강제 / 페르소나(세무 상담사) 부여 |

각 변수는 나머지 조건을 고정한 상태에서 변경하며, 동일한 평가 데이터셋으로 성능을 비교한다.

---

## 평가 지표

| 지표 | 방법 |
|------|------|
| BERTScore (필수) | 생성 답변과 정답 간 유사도 측정 (F1 기준) |
| 정답 포함 여부 | 핵심 수치·조건(한도액, 공제율 등)이 답변에 포함되는지 확인 |
| Hallucination | 근거 문서에 없는 내용을 생성한 사례 수집 및 분석 |
| 사람(팀원) 평가 | 정확성·근거성·완결성 3항목 5점 척도 교차 평가 |

- 평가 데이터셋: 질문·정답·근거 문장 **10개 이상** 직접 구축 (`eval/questions.csv`, `eval/references.csv`)

---

## 응답 시간

- 각 실험 조건별 질문당 응답 시간(검색 시간 + 생성 시간)을 측정하여 `eval/results.csv`에 기록
- 모델 크기·top_k에 따른 **응답 품질 vs 응답 시간 트레이드오프** 분석 포함

---

## 폴더 구조

```
Project_team1/
├── README.md
├── requirements.txt
├── data/                # 국세청 연말정산 PDF
├── src/
│   ├── load_pdf.py      # 문서 로드·전처리
│   ├── build_vectorstore.py
│   ├── rag_chain.py
│   ├── evaluate.py
│   └── main.py
├── eval/
│   ├── questions.csv
│   ├── references.csv
│   └── results.csv
├── report/
└── slides/
```
