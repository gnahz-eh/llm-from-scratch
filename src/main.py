import re

# 1. prepare the tokens from the traning data
with open("./resources/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# 8. test the new tokenizer
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))