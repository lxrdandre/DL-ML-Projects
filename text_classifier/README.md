A complete text classification pipeline designed to categorize news articles into World, Sports, Business, and Sci/Tec topics using the AG_NEWS dataset.
It was built with PyTorch and torchtext, utilizing data processing utilities like get_tokenizer and build_vocab_from_iterator to handle a vocabulary of over 95,000 tokens.
I implements a lightweight DL model comprising an efficient EmbeddingBag layer feeding directly into a fully connected linear layer for fast training. I later added a Transformer to find out if the model performs better with that.
Training Loop: Features a custom training process over 40 epochs using CrossEntropyLoss and an SGD optimizer with a StepLR scheduler to adjust learning rates dynamically.
