"""
[담당: 이수민] 벡터스토어 구축.

청크를 임베딩하여 Chroma 또는 FAISS 벡터스토어에 저장한다.
설정(벡터스토어 종류, 임베딩 모델, chunk_size, overlap)별로
artifacts/ 아래에 따로 저장되어 실험 간 재사용이 가능하다.

실행 예:
    python src/build_vectorstore.py                                # Chroma + OpenAI 임베딩
    python src/build_vectorstore.py --vectorstore faiss
    python src/build_vectorstore.py --embedding huggingface        # 한국어 특화 모델 비교
    python src/build_vectorstore.py --chunk-size 500 --overlap 100
"""
import argparse

from dotenv import load_dotenv

import config
from load_pdf import get_chunks

load_dotenv()


def get_embeddings(provider=config.EMBEDDING_PROVIDER):

    """임베딩 모델을 생성한다. (실험 파라미터 1: 임베딩 모델)"""
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBEDDING_MODEL)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    raise ValueError(f"지원하지 않는 임베딩: {provider}")


def build_vectorstore(vectorstore=config.VECTORSTORE,
                      embedding=config.EMBEDDING_PROVIDER,
                      chunk_size=config.CHUNK_SIZE,
                      overlap=config.CHUNK_OVERLAP):
    
    """벡터스토어를 구축하고 저장한다. (실험 파라미터 2: 벡터스토어 종류)"""
    path = config.vectorstore_path(vectorstore, embedding, chunk_size, overlap)
    embeddings = get_embeddings(embedding)

    if path.exists():
        print(f"[build_vectorstore] 이미 존재: {path.name} (재사용하려면 load_vectorstore 사용)")
        return load_vectorstore(vectorstore, embedding, chunk_size, overlap)

    chunks = get_chunks(chunk_size, overlap)

    if vectorstore == "chroma":
        from langchain_chroma import Chroma
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=str(path))
    elif vectorstore == "faiss":
        from langchain_community.vectorstores import FAISS
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(str(path))
    else:
        raise ValueError(f"지원하지 않는 벡터스토어: {vectorstore}")

    print(f"[build_vectorstore] 저장 완료 → {path}")
    return vs


def load_vectorstore(vectorstore=config.VECTORSTORE,
                     embedding=config.EMBEDDING_PROVIDER,
                     chunk_size=config.CHUNK_SIZE,
                     overlap=config.CHUNK_OVERLAP):
    
    """저장된 벡터스토어를 로드한다. 없으면 새로 구축한다."""
    path = config.vectorstore_path(vectorstore, embedding, chunk_size, overlap)
    if not path.exists():
        return build_vectorstore(vectorstore, embedding, chunk_size, overlap)

    embeddings = get_embeddings(embedding)
    if vectorstore == "chroma":
        from langchain_chroma import Chroma
        return Chroma(persist_directory=str(path), embedding_function=embeddings)
    if vectorstore == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    raise ValueError(f"지원하지 않는 벡터스토어: {vectorstore}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="벡터스토어 구축")
    parser.add_argument("--vectorstore", choices=["chroma", "faiss"], default=config.VECTORSTORE)
    parser.add_argument("--embedding", choices=["gemini", "huggingface", "openai"], default=config.EMBEDDING_PROVIDER)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    vs = build_vectorstore(args.vectorstore, args.embedding, args.chunk_size, args.overlap)

    # 검색 동작 확인
    query = "월세 세액공제 조건은?"
    for doc in vs.similarity_search(query, k=3):
        print("-" * 60)
        print(f"페이지: {doc.metadata.get('page')}, chunk_id: {doc.metadata.get('chunk_id')}")
        print(doc.page_content[:150])
