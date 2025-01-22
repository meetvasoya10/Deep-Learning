# Shakespeare Text Generation with LSTM

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

A character-level text generation model using LSTM neural networks, trained on Shakespeare's works. Generates creative text in Shakespearean style.

## Features
- Trains on Shakespearean text corpus
- Generates text with temperature-controlled randomness
- Saves generated samples to `outputs/` folder

## Getting Started

### Prerequisites
- Python 3.10+
- TensorFlow 2.12+

### Installation
1. Clone the repository:
```bash
git clone https://github.com/meetvasoya10/RNN-Project.git
cd RNN-Project
```

2. Install dependencies:
```bash
pip install tensorflow numpy
```

3. Download dataset:
```bash
mkdir -p data
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

### Train the Model
```bash
python src/train.py
```

### Generate Text
```bash
python src/generate.py --seed "ROMEO:" --length 500 --temperature 0.5
```

**Arguments**:
- `--seed`: Starting text (e.g. "ROMEO:")
- `--length`: Number of characters to generate (default: 300)
- `--temperature`: Controls randomness (0.1-1.5, default: 0.5)

## Project Structure
```
.
├── data/               # Training datasets
├── models/             # Saved model weights
├── outputs/            # Generated text samples
└── src/
    ├── model.py        # LSTM model architecture
    ├── train.py        # Training script
    └── generate.py     # Text generation script
```

## Model Architecture
```python
Sequential(
    Embedding(vocab_size, 256, input_length=100),
    LSTM(512, return_sequences=False),
    Dense(vocab_size, activation='softmax')
)
```

## Dataset
The model is trained on [Shakespeare's complete works](https://www.gutenberg.org/ebooks/100). Sample input:
```
ROMEO:
but soft, what light through yonder window breaks?
it is the east, and juliet is the sun.
```

## Example Output
```
ROMEO:
best thou, peace as i could speak well, what laves us a
read for lighting and hearted the dool or tisit
call am here to the prid please thou slave ever feer.
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

*This project is intended for educational purposes. Generated text may contain inaccuracies or unconventional phrasing.*