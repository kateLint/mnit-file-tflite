# 🧠 MNIST TensorFlow Lite Model Generator

## 📄 Overview

This project trains a **Convolutional Neural Network (CNN)** on the **MNIST dataset** (handwritten digits 0–9) using **TensorFlow**, and then converts the trained model to a lightweight **TensorFlow Lite (`.tflite`) format** for mobile and edge deployment.

The script (`mnit_file.py`) includes:

* Full preprocessing and training pipeline.
* Learning rate decay scheduling.
* Batch normalization and dropout for regularization.
* Model conversion to `.tflite` with error handling.
* Automatic saving of the final model file.

---

## ⚙️ Features

* **Dataset:** MNIST (60,000 training and 10,000 testing images)
* **Model Input:** 28×28 grayscale images
* **Output:** Probability distribution across 10 digit classes (0–9)
* **Architecture:**

  * 3 convolutional layers
  * Batch normalization and dropout for stability
  * 1 dense hidden layer (200 neurons)
  * Softmax output layer
* **Optimization:** Adam optimizer with sparse categorical cross-entropy loss
* **Learning Rate Decay:** Exponential decay using a custom scheduler

---

## 🧩 Model Architecture Summary

| Layer Type | Details                          | Activation | Dropout |
| ---------- | -------------------------------- | ---------- | ------- |
| Conv2D     | 24 filters, 6×6 kernel           | ReLU       | 0.25    |
| Conv2D     | 48 filters, 5×5 kernel, stride 2 | ReLU       | 0.25    |
| Conv2D     | 64 filters, 4×4 kernel, stride 2 | ReLU       | 0.25    |
| Dense      | 200 units                        | ReLU       | 0.25    |
| Output     | 10 units                         | Softmax    | —       |

---

## 🧪 Training Configuration

* **Batch Size:** 128
* **Epochs:** 20
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metrics:** Accuracy

A custom **learning rate scheduler** gradually reduces the learning rate to improve convergence:

```python
lr_decay = lambda epoch: 0.0001 + 0.02 * math.pow(1.0 / math.e, epoch / 3.0)
```

---

## 🧰 Conversion to TensorFlow Lite

After training:

* BatchNorm layers are frozen for inference stability.
* The model is converted to `.tflite` format using `TFLiteConverter`.
* Fallback logic ensures conversion even when Select TensorFlow Ops are required.

The generated file:

```
mnist.tflite
```

is saved in the same directory as the script.

---

## 💻 Usage

### 1️⃣ Install dependencies

```bash
pip install tensorflow tensorflow-datasets
```

### 2️⃣ Run the training and conversion

```bash
python mnit_file.py
```

### 3️⃣ Output

A file named **`mnist.tflite`** will be created in the same folder.

---

## 📊 Example Inference with `mnist.tflite`

You can use the following Python snippet to test the generated model:

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="mnist.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input (dummy data)
input_data = np.random.rand(1, 28, 28, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_digit = np.argmax(output_data)
print("Predicted digit:", predicted_digit)
```

---

## 🛡️ Error Handling

If the standard TFLite conversion fails (for example, due to unsupported ops), the script automatically retries using:

```python
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
```

ensuring maximum compatibility across hardware.

---

## 📦 Output Files

| File           | Description                         |
| -------------- | ----------------------------------- |
| `mnit_file.py` | Training and conversion script      |
| `mnist.tflite` | TensorFlow Lite model for inference |

---

