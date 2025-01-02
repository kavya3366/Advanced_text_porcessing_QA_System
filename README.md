# Advanced Text Processing and QA System

This project is an advanced text processing and question-answering system that combines three powerful components: a sophisticated text preprocessing pipeline, a self-attention based classification model, and an integrated document QA system. It leverages PyTorch for deep learning, OpenAI's language models for enhanced comprehension, and includes a robust toxicity filtering mechanism to ensure safe content processing. The system is particularly useful for content moderation, document analysis, automated question answering, and text classification tasks, making it valuable for applications in customer support, educational tools, and research assistance.

## 🎯 Project Overview

This project implements an advanced text processing and question-answering system that combines self-attention mechanisms, toxicity filtering, and OpenAI's language models. It features three main components:

1. Text Preprocessing Pipeline
2. Self-Attention Based Classification
3. Document QA System with Toxicity Filtering

## 🚀 Key Features

- **Advanced Text Preprocessing**
  - Custom text cleaning and normalization
  - Vocabulary building with configurable size
  - Sequence padding and truncation
  - Efficient batch processing with PyTorch DataLoader

- **Self-Attention Model**
  - Multi-head attention mechanism
  - Positional encoding
  - Transformer encoder layers
  - Classification capabilities
  
- **Integrated Document QA**
  - PDF document processing
  - Toxicity filtering
  - OpenAI integration
  - Vector-based retrieval
  - Source attribution

## 🛠️ Technical Architecture

### Text Preprocessing (`TextPreprocessor`)
- Handles text cleaning and normalization
- Builds vocabulary from training data
- Converts text to numerical sequences
- Manages padding and truncation

### Self-Attention Model
- `MultiHeadAttention`: Implements multi-head attention mechanism
- `PositionalEncoding`: Adds positional information to embeddings
- `TransformerEncoderLayer`: Processes text through attention and feed-forward networks
- `SelfAttentionClassifier`: Complete classification model

### Document QA System
- `IntegratedToxicityFilter`: Filters harmful content
- `OpenAIDocumentQA`: Manages document processing and question answering

## 📋 Requirements

```bash
pip install langchain langchain-openai openai pypdf faiss-cpu torch pandas numpy
```

### Hardware Requirements
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Python 3.8+

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-text-qa.git
cd advanced-text-qa

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
export OPENAI_API_KEY='your-api-key'
```

## 🔍 Usage Examples

### Text Preprocessing

```python
# Initialize preprocessor
preprocessor = TextPreprocessor(max_vocab_size=10000, max_seq_length=512)

# Prepare data
dataloader, vocab_size, df = prepare_data(
    'your_data.csv',
    preprocessor,
    text_column='text',
    label_column='label'
)
```

### Self-Attention Model

```python
# Initialize model
model = SelfAttentionClassifier(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    num_classes=2
)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
train_epoch(model, dataloader, criterion, optimizer, device)
```

### Document QA System

```python
# Initialize QA system
doc_qa = OpenAIDocumentQA(
    api_key='your-openai-key',
    toxic_threshold=0.7
)

# Load and process document
doc_qa.load_document('your_document.pdf')

# Ask questions
result = doc_qa.answer_question("Your question here?")
print(result['answer'])
```

## 🎯 Applications

1. **Content Moderation**
   - Automated toxicity detection
   - Content filtering systems
   - Safe content delivery

2. **Document Analysis**
   - Intelligent document processing
   - Information extraction
   - Automated summarization

3. **Question Answering**
   - Customer support automation
   - Educational tools
   - Research assistance

4. **Text Classification**
   - Sentiment analysis
   - Topic categorization
   - Intent detection

## 📈 Performance Considerations

- The self-attention model performs best with GPU acceleration
- Toxicity filtering can be adjusted via threshold parameters
- Document processing speed depends on chunk size and overlap settings
- Vector store performance scales with document size

## 🐛 Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce batch size
   - Decrease model size
   - Process shorter sequences

2. **Slow Document Processing**
   - Adjust chunk size
   - Reduce chunk overlap
   - Use GPU acceleration

3. **Toxicity Filter Too Strict**
   - Adjust threshold value
   - Modify sensitivity parameters
   - Fine-tune the model

4. **OpenAI API Issues**
   - Verify API key
   - Check rate limits
   - Monitor usage quotas
