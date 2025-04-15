# 🧠 Polish Spoken Word Recognition System 🇵🇱🔊

This project was created for the **Artificial Intelligence** course.  
It implements a neural network classifier trained with backpropagation to recognize spoken Polish words using the **PSD dataset**.

---

## 🎯 Project Goal

The aim was to build a complete speech recognition pipeline for isolated Polish words:
- Preprocess audio data from real-world Polish recordings  
- Convert sentences into word-level `.wav` files  
- Train a **Convolutional Neural Network (CNN)** on spectrograms  
- Achieve high accuracy in classifying 50+ distinct words

---

## 📦 Dataset – PSD (Polish Speech Database)

- Contains recordings of **spoken Polish sentences**
- Each sentence includes `.wav`, `.txt`, and `.TextGrid` files
- Words are segmented using timestamp alignment from `.TextGrid`
- Over **18,000+ audio samples** used for model training

---

## 🛠️ Technologies & Libraries

- `TensorFlow`, `Keras` – for model creation and training  
- `sklearn` – for label encoding and data splitting  
- `pydub` – for audio slicing  
- `tf.signal.stft` – to compute spectrograms  
- `tf.data.Dataset` – for optimized input pipelines  

---

## 🧱 Project Structure

| File | Purpose |
|------|---------|
| `dividing sentences into words.py` | Extracts and saves individual word-level `.wav` files using `.TextGrid` timestamps |
| `model.py` | Full training pipeline including preprocessing, CNN model, evaluation, and looped training |
| `words_useful2.txt` | List of words used for training (at least 100 instances each) |
| `Raport AI project.pdf` | Full documentation: dataset, architecture, metrics, and conclusions |

---

## 🎙️ Preprocessing

1. **Split Sentences to Words**  
   Uses timestamps from `.TextGrid` to extract word-level `.wav` files  
2. **Audio Normalization**  
   Padding or trimming waveforms to uniform length (16,000 samples)  
3. **Feature Extraction**  
   Generates **spectrograms** from audio for CNN input  
4. **Label Encoding**  
   Words are encoded using `LabelEncoder` and one-hot encoded

---

## 🧠 Neural Network Architecture

- **Input**: Normalized spectrograms  
- 3 × `Conv2D` layers (with `ReLU`, `L2` regularization)  
- 3 × `MaxPooling2D`  
- `Flatten → Dense(256) → Dropout(0.5) → Dense(softmax)`  
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Metric**: Accuracy  
- **Callbacks**: EarlyStopping, ModelCheckpoint  
- Trained across **10 training loops**, best model per loop saved

---

## 📊 Results

- Trained on **50 Polish words** with ≥100 instances each  
- ~80% **average accuracy**  
- Correctly identifies **8 out of 10 words** on average  
- Model shows strong generalization despite limited vocabulary  
- Can be scaled to more words and more speakers

---

## ▶️ How to Run

1. Prepare dataset in required PSD format  
2. Run `dividing sentences into words.py` to extract word `.wav` files  
3. Update paths in `model.py` for:
   - `words_useful2.txt`  
   - `PSD/words/` folder  
4. Run `model.py` to start training

---

## 👥 Authors

- **Olga Paszkiewicz**  
- **Krzysztof Ostrzycki**

