"""
File: validEmailAddress.py
Name: 曲華德
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO: 0.6153846153846154
"""

WEIGHT = [  # The weight vector selected by Jerry
    [0.4],  # (see assignment handout for more details)
    [0.4],
    [0.2],
    [0.2],
    [0.9],
    [-0.65],
    [0.1],
    [0.1],
    [0.1],
    [-0.7],
]

DATA_FILE = "is_valid_email.txt"  # This is the file name to be processed


def main():
    maybe_email_list = read_in_data()
    correct_classifications = 0
    total_emails = len(maybe_email_list)
    index = 0
    for maybe_email in maybe_email_list:
        index += 1
        feature_vector = feature_extractor(maybe_email)
        score = sum(
            f * w[0] for f, w in zip(feature_vector, WEIGHT)
        )  # Calculate dot product

        if score > 0 and index > 12:
            correct_classifications += 1
        if score < 0 and index <= 12:
            correct_classifications += 1

    accuracy = correct_classifications / total_emails
    print(f"Accuracy of this model: {accuracy:.16f}")


def feature_extractor(maybe_email):
    """
    :param maybe_email: str, the string to be processed
    :return: list, feature vector with 10 values of 0's or 1's
    """
    feature_vector = [
        0
    ] * 10  # Assuming there are 10 features based on the weight vector.

    feature_vector[0] = 1 if "@" in maybe_email else 0

    # For features dependent on '@' presence
    if feature_vector[0]:
        at_splits = maybe_email.split("@")
        feature_vector[1] = 1 if "." not in at_splits[0] else 0
        feature_vector[2] = 1 if at_splits[0] else 0
        feature_vector[3] = 1 if len(at_splits) > 1 and at_splits[-1] else 0
        feature_vector[4] = 1 if len(at_splits) > 1 and "." in at_splits[-1] else 0
    feature_vector[5] = 1 if not any(c.isspace() for c in maybe_email) else 0
    feature_vector[6] = 1 if maybe_email.endswith(".com") else 0
    feature_vector[7] = 1 if maybe_email.endswith(".edu") else 0
    feature_vector[8] = 1 if maybe_email.endswith(".tw") else 0
    feature_vector[9] = 1 if len(maybe_email) > 10 else 0

    return feature_vector


def read_in_data():
    """
    :return: list, containing strings that might be valid email addresses
    """
    with open(DATA_FILE, "r") as f:
        return [line.strip() for line in f]


if __name__ == "__main__":
    main()
