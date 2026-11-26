"""Small local RAG demo for cyclone equipment PDFs.

Commands:
    python rag_prototype.py inspect
    python rag_prototype.py index
    python rag_prototype.py ask "Which Kice Cyclone components should be checked for wear and damage?"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
INDEX_DIR = HERE / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANSWER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHUNK_WORDS = 150
OVERLAP_WORDS = 30
TOP_K = 5
MIN_SCORE = 0.30  # Starting value; assess against evaluation questions.
FALLBACK = "I could not find a supported answer in the selected documents."


def pdf_passages(docs_dir: Path) -> tuple[list[dict], int]:
    """Extract page-bounded chunks and count pages without useful text."""
    passages = []
    skipped_pages = 0
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        for page_number, page in enumerate(reader.pages, start=1):
            words = re.sub(r"\s+", " ", page.extract_text() or "").strip().split()
            if len(words) < 20:
                skipped_pages += 1
                continue
            step = CHUNK_WORDS - OVERLAP_WORDS
            for start in range(0, len(words), step):
                group = words[start : start + CHUNK_WORDS]
                if len(group) < 20:
                    break
                passages.append({
                    "source": pdf.name,
                    "page": page_number,
                    "text": " ".join(group),
                })
                if start + CHUNK_WORDS >= len(words):
                    break
    if not passages:
        raise ValueError("No usable text was extracted from the PDFs.")
    return passages, skipped_pages


def build_index(docs_dir: Path, index_dir: Path) -> None:
    """Encode passages and save a FAISS index with matching metadata."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    passages, skipped = pdf_passages(docs_dir)
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        [item["text"] for item in passages],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "passages.faiss"))
    (index_dir / "passages.json").write_text(
        json.dumps(passages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(passages)} passages from {len(list(docs_dir.glob('*.pdf')))} PDFs.")
    print(f"Pages with little extractable text: {skipped}.")
    print(f"Index saved to {index_dir}")


def retrieve(question: str, index_dir: Path, top_k: int) -> list[dict]:
    """Combine semantic and keyword scores for this small local corpus."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    index_path = index_dir / "passages.faiss"
    metadata_path = index_dir / "passages.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Index missing. Run the index command first.")
    index = faiss.read_index(str(index_path))
    passages = json.loads(metadata_path.read_text(encoding="utf-8"))
    if index.ntotal != len(passages):
        raise ValueError("Index and metadata disagree. Rebuild the index.")

    search_text = re.sub(
        r"\b(regular|routine|periodic) inspection\b",
        "maintenance inspection", question, flags=re.IGNORECASE
    )
    model = SentenceTransformer(EMBED_MODEL)
    vector = model.encode([search_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vector)
    dense_scores, ids = index.search(vector, index.ntotal)
    dense = np.zeros(index.ntotal, dtype="float32")
    dense[ids[0]] = dense_scores[0]

    keywords = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = keywords.fit_transform(item["text"] for item in passages)
    keyword = (keywords.transform([search_text]) @ matrix.T).toarray()[0]
    combined = 0.7 * dense + 0.3 * keyword
    # Prefer a rare exact phrase such as "insufficient airflow" when present.
    file_words = set(re.findall(r"[a-z]{4,}", " ".join(
        item["source"] for item in passages
    ).lower()))
    file_words.update({"cyclone", "separator", "collector", "manual", "report"})
    words = re.findall(r"[a-z]+", question.lower())
    phrases = [
        a + " " + b for a, b in zip(words, words[1:])
        if len(a) >= 4 and len(b) >= 4
        and a not in file_words and b not in file_words
    ]
    matched = [phrase for phrase in phrases if any(
        phrase in item["text"].lower() for item in passages
    )]
    if matched:
        def rarity(phrase: str) -> float:
            return sum(keywords.idf_[keywords.vocabulary_[word]]
                       for word in phrase.split()
                       if word in keywords.vocabulary_)
        phrase = max(matched, key=rarity)
        combined += np.array([
            0.30 if phrase in item["text"].lower() else 0.0
            for item in passages
        ])
    best_ids = np.argsort(-combined)[:top_k]
    return [
        {**passages[int(i)], "score": round(float(combined[i]), 3)}
        for i in best_ids
    ]

def blocked_question(question: str) -> bool:
    """Refuse simple safety-bypass and private-information requests."""
    q = question.lower()
    safety_bypass = (
        re.search(r"\b(bypass|disable|override|defeat)\b", q)
        and re.search(r"\b(safety|interlock|alarm|guard)\b", q)
    )
    private_disclosure = (
        re.search(r"\b(show|reveal|give|list|print|share|disclose|send|provide)\b", q)
        and re.search(
            r"\b(passwords?|api keys?|secret keys?|credentials?|private (?:information|data)|personal (?:information|data)|employee (?:records?|details?))\b",
            q,
        )
    )
    return bool(safety_bypass or private_disclosure)

def source_excerpt(item: dict, question: str) -> str:
    """Show the relevant part of a source page when an exact phrase is found."""
    pdf_path = DOCS_DIR / item["source"]
    if not pdf_path.is_file():
        return item["text"]
    page_text = re.sub(
        r"\s+", " ", PdfReader(str(pdf_path)).pages[item["page"] - 1].extract_text() or ""
    ).strip()
    words = re.findall(r"[a-z]+", question.lower())
    phrases = [
        a + " " + b for a, b in zip(words, words[1:])
        if len(a) >= 4 and len(b) >= 4
    ]
    matches = [phrase for phrase in phrases if phrase in item["text"].lower()]
    if not matches:
        text = item["text"]
        return text.rsplit(". ", 1)[0] + "." if ". " in text else text
    position = page_text.lower().find(matches[-1])
    if position < 0:
        return item["text"]
    if question.lower().startswith("how "):
        before = page_text[:position].split()[-60:]
        after = page_text[position:].split()[:180]
        return " ".join(before + after)
    return " ".join(page_text[position:].split()[:220])


def generate_answer(question: str, sources: list[dict]) -> str:
    """Draft a short answer from selected passages using a local model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    sources = list(sources)
    first = sources[0]
    continuation = False
    # Manual sections often continue on the next page.
    for item in sources[1:]:
        if (
            item["source"] == first["source"]
            and item["page"] == first["page"] + 1
            and item["score"] >= first["score"] - 0.10
        ):
            sources.remove(item)
            sources.insert(0, item)
            continuation = True
            break

    selected = sources[:2] if continuation else sources[:1]
    primary_excerpt = source_excerpt(selected[0], question)
    context = "\n\n".join(
        f"[{number}] {item['source']}, page {item['page']}:\n"
        f"{primary_excerpt if number == 1 else item['text']}"
        for number, item in enumerate(selected, start=1)
    )
    tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(ANSWER_MODEL)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied manual passages. Be concise and "
                "include every relevant item. Do not add outside facts. If the "
                f"passages do not answer the question, say exactly: {FALLBACK}"
            ),
        },
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    output = model.generate(
        **encoded, max_new_tokens=120, do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(
        output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    if not answer or FALLBACK.lower() in answer.lower():
        return FALLBACK
    evidence = " ".join(item["text"] for item in selected).lower()
    evidence_words = set(re.findall(r"[a-z]{4,}", evidence))
    ignore = {"answer", "checked", "check", "during", "following", "should",
              "these", "those", "there", "their", "which", "would", "could",
              "parts", "regular", "inspection", "summary", "cyclone"}
    answer_words = [word for word in re.findall(r"[a-z]{4,}", answer.lower())
                    if word not in ignore]
    coverage = sum(word in evidence_words for word in answer_words) / max(1, len(answer_words))
    numbers_ok = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer)) <= set(
        re.findall(r"\b\d+(?:\.\d+)?\b", evidence)
    )
    if coverage < 0.80 or not numbers_ok:
        answer = ""
    pages = dict.fromkeys((item["source"], item["page"]) for item in selected)
    citations = "; ".join(f"{name}, p. {page}" for name, page in pages)
    summary = f"Model summary: {answer}\n" if answer else ""
    return (
        f"{summary}Answer from source ({selected[0]['source']}, p. {selected[0]['page']}):\n"
        f"{primary_excerpt}\nSources reviewed: {citations}"
    )

def answer_question(question: str, index_dir: Path, min_score: float) -> None:
    if blocked_question(question):
        print("I cannot help bypass safety controls or disclose private information. Consult the approved procedure.")
        return
    sources = retrieve(question, index_dir, TOP_K)
    if not sources or sources[0]["score"] < min_score:
        print(FALLBACK)
    else:
        print(generate_answer(question, sources))
    print("\nRetrieved passages:")
    for number, item in enumerate(sources, start=1):
        snippet = item["text"][:200].replace("\n", " ")
        print(
            f"[{number}] {item['source']}, p. {item['page']} "
            f"(score {item['score']:.3f}): {snippet}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check PDF extraction without models")
    inspect.add_argument("--docs", type=Path, default=DOCS_DIR)
    index = sub.add_parser("index", help="Build the local FAISS index")
    index.add_argument("--docs", type=Path, default=DOCS_DIR)
    index.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ask = sub.add_parser("ask", help="Ask a question using the saved index")
    ask.add_argument("question")
    ask.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ask.add_argument("--min-score", type=float, default=MIN_SCORE)
    args = parser.parse_args()

    if args.command == "inspect":
        passages, skipped = pdf_passages(args.docs)
        print(f"Found {len(passages)} passages; {skipped} pages had little text.")
    elif args.command == "index":
        build_index(args.docs, args.index_dir)
    else:
        answer_question(args.question, args.index_dir, args.min_score)


if __name__ == "__main__":
    main()







