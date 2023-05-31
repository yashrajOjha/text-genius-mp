from spellchecker import SpellChecker

def spellchecker(text):
    spell = SpellChecker()
    
    words = text.split()
    misspelled = spell.unknown(words)

    for word in misspelled:
        corrected_word = spell.correction(word)
        text = text.replace(word, corrected_word)
    return text
