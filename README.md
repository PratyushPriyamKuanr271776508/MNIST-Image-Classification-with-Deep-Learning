# **MNIST Digit Classification with CNN**

This repository contains a complete implementation of a **Convolutional Neural Network (CNN)** for classifying handwritten digits from the MNIST dataset using TensorFlow/Keras. The project includes:

- Data loading and preprocessing
- Exploratory data analysis (EDA) with visualizations
- CNN model building and training
- Model evaluation and prediction analysis
- Detailed pixel-level image visualization

## **Features**

1. **Data Processing**
   - Normalizes pixel values (0-255 → 0-1)
   - Reshapes images for CNN input (28×28×1)
   - One-hot encodes labels

2. **Model Architecture**
   - 2 convolutional layers with ReLU activation
   - Max pooling for dimensionality reduction
   - Dropout for regularization
   - Achieves **~99% test accuracy**

3. **Visualizations**
   - Sample digit displays
   - Training history plots
   - Confusion matrix
   - Pixel-value annotation utility

4. **Utilities**
   - `visualize_input()` function for inspecting individual digits with pixel values
   - Detailed model prediction analysis

## **Requirements**

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install requirements:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## **Files**

- `mnist_cnn.ipynb`: Jupyter notebook with complete implementation
- `README.md`: This file

## **Usage**

1. Run the notebook cells sequentially to:
   - Load and preprocess data
   - Train the CNN model
   - Evaluate performance
   - Visualize results

2. Key functions:
   ```python
   # Visualize a single digit with pixel values
   visualize_input(image, axes)
   
   # Train the model
   model.fit(X_train, y_train, epochs=10)
   
   # Evaluate
   model.evaluate(X_test, y_test)
   ```

## **Results**

- Test accuracy: **~99%**
- Training time: ~2 minutes on CPU
- Confusion matrix shows most errors occur between similar digits (4/9, 7/9)

## **Example Output**

![Sample Visualization](https://miro.medium.com/max/700/1*5T7kK7X4h5l5X5Q5Q5Q5Qw.png)

## **Contributing**

Contributions welcome! Please open an issue or PR for:
- Model improvements
- Additional visualizations
- Documentation enhancements

## **License**

MIT License
