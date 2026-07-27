"""
[담당: 이수민] 벡터스토어 구축.

청크를 임베딩하여 Chroma 또는 FAISS 벡터스토어에 저장한다.
설정(벡터스토어 종류, 임베딩 모델, chunk_size, overlap)별로
artifacts/ 아래에 따로 저장되어 실험 간 재사용이 가능하다.

실행 예:
    python src/build_vectorstore.py                                # Chroma + OpenAI 임베딩
    python src/build_vectorstore.py --vectorstore faiss             # FAISS Flat
    python src/build_vectorstore.py --vectorstore faiss --faiss-index hnsw
    python src/build_vectorstore.py --embedding huggingface        # 한국어 특화 모델 비교
    python src/build_vectorstore.py --chunk-size 500 --overlap 100
"""
import argparse

from dotenv import load_dotenv

import config
from eval_retrieval import get_embedding_model_name
from load_pdf import LOADER_CHOICES, get_chunks

load_dotenv()

FAISS_INDEX_CHOICES = ("flat", "hnsw")
FAISS_HNSW_M = 32


def get_vectorstore_path(vectorstore, embedding, chunk_size, overlap,
                         embedding_model=None, faiss_index="flat", loader=None):
    """FAISS 인덱스·PDF 로더가 다른 스토어가 서로 덮어쓰이지 않게 경로를 만든다.

    ※ 버그 수정 (regression_check 실행 중 발견): embedding_model을 명시적으로
    지정하지 않으면 model_tag가 비어, config의 provider 기본 모델이 무엇이든
    경로가 항상 "chroma_huggingface_..."로 동일했다. config.HF_EMBEDDING_MODEL을
    ko-sroberta(768차원) → bge-m3(1024차원)로 바꾼 뒤 --embedding-model 없이
    evaluate.py를 실행하면, bge-m3 인코더로 예전 ko-sroberta 컬렉션을 쿼리하게
    되어 "Collection expecting embedding with dimension of 768, got 1024"
    에러가 발생했다. embedding_model이 None이면 실제 사용될 provider 기본
    모델명을 먼저 resolve해서 경로에 반영하도록 고쳤다 (기존 --embedding-model
    명시 실행에서 만들어둔 artifacts/chroma_huggingface_bge-m3_cs500_ov100은
    그대로 재사용됨).
    """
    resolved_model = embedding_model or get_embedding_model_name(embedding, config)
    path = config.vectorstore_path(
        vectorstore, embedding, chunk_size, overlap, resolved_model,
        loader or config.PDF_LOADER,
    )
    if vectorstore == "faiss" and faiss_index != "flat":
        path = path.with_name(f"{path.name}_{faiss_index}")
    return path


def create_faiss_from_embeddings(text_embeddings, embeddings, metadatas,
                                 ids=None, faiss_index="flat"):
    """동일한 사전 계산 임베딩으로 FAISS Flat 또는 HNSW 스토어를 만든다."""

    from langchain_community.vectorstores import FAISS

    pairs = list(text_embeddings)
    if faiss_index == "flat":
        return FAISS.from_embeddings(
            pairs, embeddings, metadatas=metadatas, ids=ids
        )

    if faiss_index == "hnsw":
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore

        if not pairs:
            raise ValueError("FAISS HNSW 인덱스에 추가할 임베딩이 없습니다.")

        dimension = len(pairs[0][1])
        index = faiss.IndexHNSWFlat(dimension, FAISS_HNSW_M, faiss.METRIC_L2)
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vectorstore.add_embeddings(pairs, metadatas=metadatas, ids=ids)
        return vectorstore

    raise ValueError(
        f"지원하지 않는 FAISS 인덱스: {faiss_index} "
        f"(선택: {', '.join(FAISS_INDEX_CHOICES)})"
    )


def create_faiss_from_documents(chunks, embeddings, faiss_index="flat"):
    """Document를 임베딩해 선택한 FAISS 인덱스로 만든다."""

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    return create_faiss_from_embeddings(
        zip(texts, vectors),
        embeddings,
        metadatas,
        faiss_index=faiss_index,
    )



def get_embeddings(provider=config.EMBEDDING_PROVIDER, model=None):
    """임베딩 모델을 생성한다. model이 None이면 config 기본값 사용. (실험 파라미터: 임베딩 모델)"""

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model or config.GEMINI_EMBEDDING_MODEL)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or config.OPENAI_EMBEDDING_MODEL)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model or config.HF_EMBEDDING_MODEL)

    raise ValueError(f"지원하지 않는 임베딩: {provider}")


def build_vectorstore(vectorstore=config.VECTORSTORE,
                      embedding=config.EMBEDDING_PROVIDER,
                      chunk_size=config.CHUNK_SIZE,
                      overlap=config.CHUNK_OVERLAP,
                      embedding_model=None,
                      faiss_index="flat",
                      loader=None):
    """벡터스토어를 구축하고 저장한다. (실험 파라미터 2: 벡터스토어 종류)"""

    loader = loader or config.PDF_LOADER
    path = get_vectorstore_path(
        vectorstore, embedding, chunk_size, overlap, embedding_model, faiss_index, loader
    )
    embeddings = get_embeddings(embedding, embedding_model)

    if path.exists():
        print(f"[build_vectorstore] 이미 존재: {path.name} (재사용하려면 load_vectorstore 사용)")
        return load_vectorstore(
            vectorstore, embedding, chunk_size, overlap, embedding_model, faiss_index, loader
        )

    chunks = get_chunks(chunk_size, overlap, loader=loader)

    if vectorstore == "chroma":
        from langchain_chroma import Chroma
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=str(path))
    elif vectorstore == "faiss":
        vs = create_faiss_from_documents(chunks, embeddings, faiss_index)
        vs.save_local(str(path))
    else:
        raise ValueError(f"지원하지 않는 벡터스토어: {vectorstore}")

    print(f"[build_vectorstore] 저장 완료 → {path}")
    return vs


def load_vectorstore(vectorstore=config.VECTORSTORE,
                     embedding=config.EMBEDDING_PROVIDER,
                     chunk_size=config.CHUNK_SIZE,
                     overlap=config.CHUNK_OVERLAP,
                     embedding_model=None,
                     faiss_index="flat",
                     loader=None):
    """저장된 벡터스토어를 로드한다. 없으면 새로 구축한다."""

    loader = loader or config.PDF_LOADER
    path = get_vectorstore_path(
        vectorstore, embedding, chunk_size, overlap, embedding_model, faiss_index, loader
    )
    if not path.exists():
        return build_vectorstore(
            vectorstore, embedding, chunk_size, overlap, embedding_model, faiss_index, loader
        )

    embeddings = get_embeddings(embedding, embedding_model)

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
    parser.add_argument("--faiss-index", choices=FAISS_INDEX_CHOICES, default="flat")
    parser.add_argument("--embedding", choices=["huggingface", "openai", "gemini"], default=config.EMBEDDING_PROVIDER)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--loader", choices=LOADER_CHOICES, default=config.PDF_LOADER)
    args = parser.parse_args()

    vs = build_vectorstore(
        args.vectorstore,
        args.embedding,
        args.chunk_size,
        args.overlap,
        faiss_index=args.faiss_index,
        loader=args.loader,
    )

    # 검색 동작 확인
    query = "월세 세액공제 조건은?"
    for doc in vs.similarity_search(query, k=3):
        print("-" * 60)
        print(f"페이지: {doc.metadata.get('page')}, chunk_id: {doc.metadata.get('chunk_id')}")
        print(doc.page_content[:150])
