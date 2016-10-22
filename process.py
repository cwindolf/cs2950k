import re
import os
from collections import Counter


def basic_tokenizer(sentence, word_split=re.compile('([.,!?"\'\-*:;)(])')):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]


num_lines = 0
with open('mobydick.txt', 'r') as moby, open('tmp.dat', 'w') as tmp:
    for line in moby:
        print(' '.join(basic_tokenizer(line)), file=tmp)
        num_lines += 1


with open('tmp.dat', 'r') as tmp, \
     open('mobyUNK_train.txt', 'w') as train:
    lines_so_far = 0
    out = train
    for line in tmp:
        if lines_so_far > num_lines * 9 / 10:
            break
        out.write(line)
        out.write('\n')
        lines_so_far += 1

cnt = Counter()
with open('mobyUNK_train.txt', 'r') as train:
    for line in train:
        for word in line.split():
            cnt[word] += 1
common = set(word for word, _ in cnt.most_common(8000))


with open('tmp.dat', 'r') as tmp, \
     open('mobyUNK_train.txt', 'w') as train, \
     open('mobyUNK_test.txt', 'w') as test:
    lines_so_far = 0
    out = train
    for line in tmp:
        if lines_so_far > num_lines * 9 / 10:
            out = test
        for word in line.split():
            if word in common:
                out.write('%s ' % word)
            else:
                out.write('UNK ')
        out.write('\n')
        lines_so_far += 1

os.remove('tmp.dat')
