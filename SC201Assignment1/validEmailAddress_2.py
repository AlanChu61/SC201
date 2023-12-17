import numpy as np

"""
File: validEmailAddress_2.py
Name: 曲華德
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO: 是否包含@符號。 
feature2:  TODO: @符號前的部分（local）是否沒有連續的點號。
feature3:  TODO: @後面至少有一個.。
feature4:  TODO: @符號後的部分（domain）中點號之間是否有字符（不是連續的點號）。
feature5:  TODO: 本地部分是否只包含有效字符（字母、數字、下划線、加號、減號和點號）。
feature6:  TODO: 域名部分是否只包含有效字符（字母、數字和減號）。
feature7:  TODO: 域名部分是否不以減號或點號開始或結束。
feature8:  TODO: 本地部分是否不以特殊字符（例如引號、空格、反斜槓等）開始或結束。
feature9:  TODO: 域名部分是否包含至少一個頂級域名（TLD），如.com, .org, .net, .edu, .gov, .tw等。
feature10: TODO: @後面的部分domian是否不包含特殊字符。

Accuracy of your model: TODO:0.7307692307692307
"""

WEIGHT = np.array(
    [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ]
)


DATA_FILE = "is_valid_email.txt"  # This is the file name to be processed


def main():
    maybe_email_list = read_in_data()
    total_emails = len(maybe_email_list)
    correct_classifications = 0

    index = 0
    for maybe_email in maybe_email_list:
        index += 1
        feature_vector = feature_extractor(maybe_email)
        score = WEIGHT.T.dot(feature_vector)
        if score == len(WEIGHT) and index > 12:
            correct_classifications += 1
        if score != len(WEIGHT) and index <= 12:
            correct_classifications += 1
    accuracy = correct_classifications / total_emails
    print(f"Accuracy of this model: {accuracy:.16f}")


def feature_extractor(maybe_email):
    feature_vector = np.zeros((10, 1))

    # Feature 1: Whether it contains the '@' symbol.
    feature_vector[0] = 1 if "@" in maybe_email else 0

    # Split the email into local and domain parts
    parts = maybe_email.split("@")
    local_part = parts[0] if len(parts) > 1 else ""
    domain_part = parts[1] if len(parts) == 2 else ""

    # Feature 2: The local part does not contain consecutive dots.
    feature_vector[1] = 1 if ".." not in local_part else 0

    # Feature 3: There is at least one '.' in the domain part.
    feature_vector[2] = 1 if "." in domain_part else 0

    # Feature 4: Domain part contains characters between dots.
    feature_vector[3] = (
        1 if all(len(part) > 0 for part in domain_part.split(".")) else 0
    )

    # Feature 5: Local part contains only valid characters.
    valid_chars_local = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+"
    )
    feature_vector[4] = 1 if set(local_part).issubset(valid_chars_local) else 0

    # Feature 6: Domain part contains only valid characters.
    valid_chars_domain = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
    )
    feature_vector[5] = (
        1
        if set(domain_part).issubset(valid_chars_domain) and ".." not in domain_part
        else 0
    )

    # Feature 7: Domain part does not start or end with a hyphen or dot.
    feature_vector[6] = (
        1
        if not (
            domain_part.startswith("-")
            or domain_part.endswith("-")
            or domain_part.startswith(".")
            or domain_part.endswith(".")
        )
        else 0
    )

    # Feature 8: Local part does not start or end with special characters.
    special_chars = set('()"\\,;:<>[]')
    feature_vector[7] = (
        1
        if not (
            local_part.startswith(tuple(special_chars))
            or local_part.endswith(tuple(special_chars))
        )
        else 0
    )

    # Feature 9: Domain part contains at least one top-level domain.
    tlds = {".com", ".org", ".net", ".edu", ".gov", ".tw"}
    feature_vector[8] = 1 if any(domain_part.endswith(tld) for tld in tlds) else 0

    # Feature 10: Domain part does not contain special characters.
    feature_vector[9] = (
        1 if not any(char in special_chars for char in domain_part) else 0
    )

    return feature_vector


def read_in_data():
    """
    :return: list, containing strings that might be valid email addresses
    """
    with open(DATA_FILE, "r") as f:
        return [line.strip() for line in f]


if __name__ == "__main__":
    main()
