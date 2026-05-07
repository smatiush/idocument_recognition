from __future__ import annotations

from .labels import PageLabel


def enforce_valid_sequence(predicted_labels: list[str]) -> list[str]:
    if not predicted_labels:
        return []

    normalized: list[str] = []
    open_document = False

    for index, label in enumerate(predicted_labels):
        is_last_page = index == len(predicted_labels) - 1

        if label == PageLabel.SINGLE_PAGE_DOC:
            normalized.append(PageLabel.SINGLE_PAGE_DOC.value)
            open_document = False
            continue

        if label == PageLabel.START_DOC:
            normalized.append(PageLabel.START_DOC.value)
            open_document = True
            continue

        if label == PageLabel.MIDDLE_DOC:
            if not open_document:
                normalized.append(
                    PageLabel.SINGLE_PAGE_DOC.value if is_last_page else PageLabel.START_DOC.value
                )
                open_document = not is_last_page
            else:
                normalized.append(PageLabel.END_DOC.value if is_last_page else PageLabel.MIDDLE_DOC.value)
            continue

        if label == PageLabel.END_DOC:
            if not open_document:
                normalized.append(PageLabel.SINGLE_PAGE_DOC.value)
            else:
                normalized.append(PageLabel.END_DOC.value)
                open_document = False
            continue

        normalized.append(PageLabel.SINGLE_PAGE_DOC.value)
        open_document = False

    if normalized and normalized[-1] == PageLabel.START_DOC.value:
        normalized[-1] = PageLabel.SINGLE_PAGE_DOC.value

    return normalized


def build_documents(predicted_labels: list[str]) -> list[tuple[int, int]]:
    documents: list[tuple[int, int]] = []
    start_page: int | None = None

    for page_number, label in enumerate(predicted_labels, start=1):
        if label == PageLabel.SINGLE_PAGE_DOC:
            documents.append((page_number, page_number))
            start_page = None
        elif label == PageLabel.START_DOC:
            start_page = page_number
        elif label == PageLabel.END_DOC:
            if start_page is None:
                documents.append((page_number, page_number))
            else:
                documents.append((start_page, page_number))
                start_page = None

    if start_page is not None:
        documents.append((start_page, len(predicted_labels)))

    return documents


def build_documents_from_same_doc_probs(
    same_document_probabilities: list[float],
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    if not same_document_probabilities:
        return [(1, 1)]

    documents: list[tuple[int, int]] = []
    start_page = 1

    for left_page, same_document_probability in enumerate(same_document_probabilities, start=1):
        if same_document_probability < threshold:
            documents.append((start_page, left_page))
            start_page = left_page + 1

    documents.append((start_page, len(same_document_probabilities) + 1))
    return documents


def ranges_to_page_labels(ranges: list[tuple[int, int]], total_pages: int) -> list[str]:
    labels = [PageLabel.MIDDLE_DOC.value] * total_pages

    for start_page, end_page in ranges:
        if start_page == end_page:
            labels[start_page - 1] = PageLabel.SINGLE_PAGE_DOC.value
            continue

        labels[start_page - 1] = PageLabel.START_DOC.value
        labels[end_page - 1] = PageLabel.END_DOC.value

        for page_number in range(start_page + 1, end_page):
            labels[page_number - 1] = PageLabel.MIDDLE_DOC.value

    return labels
