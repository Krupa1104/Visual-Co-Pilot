"""
Shared answer-label mapping for Radar VQA.

27 valid closed-set classes + 1 reserved output.
Callsign recognition is handled separately by OCR.
"""

NUM_OUTPUTS = 28
NUM_VALID_CLASSES = 27
RESERVED_CLASS_ID = 27

ANSWER_TO_ID = {
    # Altitude: 4 classes
    "10k-20k": 0,
    "20k-30k": 1,
    "30k-40k": 2,
    "40k+": 3,

    # Heading: 8 classes
    "N": 4,
    "NE": 5,
    "E": 6,
    "SE": 7,
    "S": 8,
    "SW": 9,
    "W": 10,
    "NW": 11,

    # Aircraft count: 9 classes
    "0": 12,
    "1": 13,
    "2": 14,
    "3": 15,
    "4": 16,
    "5": 17,
    "6": 18,
    "7": 19,
    "8+": 20,

    # Yes / No: 2 classes
    "yes": 21,
    "no": 22,

    # Quadrant: 4 classes
    "north-east": 23,
    "north-west": 24,
    "south-east": 25,
    "south-west": 26,
}

ID_TO_ANSWER = {
    class_id: answer
    for answer, class_id in ANSWER_TO_ID.items()
}

QUESTION_TYPE_TO_ANSWERS = {
    "altitude": {"10k-20k", "20k-30k", "30k-40k", "40k+"},
    "heading": {"N", "NE", "E", "SE", "S", "SW", "W", "NW"},
    "aircraft_count": {"0", "1", "2", "3", "4", "5", "6", "7", "8+"},
    "yes_no": {"yes", "no"},
    "quadrant": {
        "north-east", "north-west", "south-east", "south-west"
    },
}

QUESTION_TYPE_TO_TASK = {
    "altitude": "classification",
    "heading": "classification",
    "aircraft_count": "classification",
    "yes_no": "classification",
    "quadrant": "classification",
    "callsign": "ocr",
}


def encode_answer(question_type, answer):
    """Encode a valid closed-set answer for its question type."""

    if question_type == "callsign":
        raise ValueError(
            "Callsign must be encoded by the OCR character encoder."
        )

    if question_type not in QUESTION_TYPE_TO_ANSWERS:
        raise ValueError(f"Unknown question type: {question_type}")

    if answer not in QUESTION_TYPE_TO_ANSWERS[question_type]:
        raise ValueError(
            f"Invalid answer '{answer}' for question type '{question_type}'."
        )

    return ANSWER_TO_ID[answer]


def decode_answer(class_id):
    """Decode a valid classifier class ID."""

    if class_id == RESERVED_CLASS_ID:
        raise ValueError("Reserved class 27 is not a valid answer.")

    if class_id not in ID_TO_ANSWER:
        raise ValueError(f"Invalid classifier class ID: {class_id}")

    return ID_TO_ANSWER[class_id]


def get_valid_logits(logits):
    """
    Return only the 27 valid classifier outputs.

    This is safe with label-smoothed cross-entropy.
    Do not replace the reserved logit with -inf before
    applying label smoothing.
    """
    if logits.shape[-1] != NUM_OUTPUTS:
        raise ValueError(
            f"Expected {NUM_OUTPUTS} logits, got {logits.shape[-1]}."
        )

    return logits[..., :NUM_VALID_CLASSES]


def validate_mapping():
    ids = list(ANSWER_TO_ID.values())

    assert len(ANSWER_TO_ID) == NUM_VALID_CLASSES
    assert len(set(ids)) == NUM_VALID_CLASSES
    assert set(ids) == set(range(NUM_VALID_CLASSES))
    assert RESERVED_CLASS_ID not in ids
    assert NUM_VALID_CLASSES + 1 == NUM_OUTPUTS

    # Every classification question type must have valid answers.
    for question_type, answers in QUESTION_TYPE_TO_ANSWERS.items():
        assert answers
        assert answers.issubset(ANSWER_TO_ID)

    # Every closed-set answer must belong to exactly one question type.
    all_typed_answers = [
        answer
        for answers in QUESTION_TYPE_TO_ANSWERS.values()
        for answer in answers
    ]

    assert len(all_typed_answers) == len(set(all_typed_answers))
    assert set(all_typed_answers) == set(ANSWER_TO_ID)

    return True


if __name__ == "__main__":
    validate_mapping()

    print("ANSWER MAPPING VALIDATED")
    print("Classifier outputs:", NUM_OUTPUTS)
    print("Valid classes:", NUM_VALID_CLASSES)
    print("Reserved output:", RESERVED_CLASS_ID)
    print("Question-type answer validation: PASS")
