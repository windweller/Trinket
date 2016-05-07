def decode_sentences(sentences, idx_to_word):
    """
    Decode the sentence

    Parameters
    ----------
    sentences: can be a numpy array of indicies
    idx_to_word: the entire dictionary

    Returns
    -------
    a sentence or a list of sentences
    """
    singleton = False
    if sentences.ndim == 1:
        singleton = True
        sentences = sentences[None]  # this is actually lifting this to a 2-dim array
    decoded = []
    N, T = sentences.shape
    for i in xrange(N):
        words = []
        for t in xrange(T):
            word = idx_to_word[sentences[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def decode_sentences_list(sentence, idx_to_word):
    """
    This assumes sentence is a list []
    :param sentence:
    :param idx_to_word:
    :return:
    """
    N = len(sentence)
    decoded = []
    for i in xrange(N):
        word = idx_to_word[sentence[i]]
        if word != '<NULL>':
            decoded.append(word)
        if word == '<END>':
            break
    return decoded