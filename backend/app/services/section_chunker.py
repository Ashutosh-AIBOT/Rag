import re
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger

logger = get_logger(__name__)

HEADING_PATTERN = r'^(#{1,2})\s+(.+)$'


def _extract_section(text: str) -> str:
    match = re.search(HEADING_PATTERN, text, re.MULTILINE)
    if match:
        return match.group(2).strip()
    return "Untitled"


def _section_split(documents: list[Document]) -> list[Document]:
    chunks = []
    for doc in documents:
        content = doc.page_content
        sections = re.split(HEADING_PATTERN, content, flags=re.MULTILINE)

        current_section = "Untitled"
        current_content = []

        for part in sections:
            if re.match(HEADING_PATTERN, part, re.MULTILINE):
                if current_content:
                    section_text = "\n".join(current_content).strip()
                    if section_text:
                        chunk = Document(
                            page_content=section_text,
                            metadata={
                                **doc.metadata,
                                "strategy": "section",
                                "section": current_section,
                            }
                        )
                        chunks.append(chunk)
                    current_content = []
                current_section = part.strip()
            else:
                current_content.append(part)

        if current_content:
            section_text = "\n".join(current_content).strip()
            if section_text:
                chunk = Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "strategy": "section",
                        "section": current_section,
                    }
                )
                chunks.append(chunk)

    logger.info(f"Section chunking: {len(chunks)} chunks created")
    return chunks


section_chunking_chain = RunnableLambda(_section_split)
