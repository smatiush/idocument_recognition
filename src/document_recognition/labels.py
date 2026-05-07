from enum import Enum


class PageLabel(str, Enum):
    START_DOC = "START_DOC"
    MIDDLE_DOC = "MIDDLE_DOC"
    END_DOC = "END_DOC"
    SINGLE_PAGE_DOC = "SINGLE_PAGE_DOC"


LABELS = [label.value for label in PageLabel]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class PairLabel(str, Enum):
    SAME_DOCUMENT = "SAME_DOCUMENT"
    NEW_DOCUMENT = "NEW_DOCUMENT"


PAIR_LABELS = [label.value for label in PairLabel]
PAIR_LABEL_TO_ID = {label: idx for idx, label in enumerate(PAIR_LABELS)}
PAIR_ID_TO_LABEL = {idx: label for label, idx in PAIR_LABEL_TO_ID.items()}
