import re

import unidecode

def format_text(text, convert_to_lower, convert_to_unidecode):

    if convert_to_unidecode:
        if re.search('([一-龯])', text):
            """ kanji """
            raise Exception('kanji in katakana')
        if re.search('[\u3040-\u309F\u30A0-\u30FF]', text):
            """ katakana """
            text = re.sub('[・·=゠＝\u3000 \-、，,]', '・', text)
        else:
            text = unidecode.unidecode(text)
            text = text.replace('`', "'")
    if convert_to_lower:
        text = text.lower()
    return text
