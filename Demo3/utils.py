import json
def load_data(args, filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        for line in f:
            print(json.loads(line)['labels'])
            exit()
        examples = [json.loads(line) for line in f]


    print(examples)
    exit()
    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
                ex['question_char'] = [w.lower() for w in ex['question_char']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]
                ex['document_char'] = [w.lower() for w in ex['document_char']]

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]
    return examples