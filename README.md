This project is an AI-based energy forecasting system that predicts electricity consumption using Machine Learning and Deep Learning (LSTM) techniques. It analyzes historical time-series data to forecast future energy usage and provides visual insights for better decision-making.
Tech Stack
Programming Language: Python
Libraries:
NumPy
Pandas
Matplotlib
Scikit-learn
TensorFlow / Keras
Dataset Source: Kaggle (Hourly Energy Consumption Dataset)
📂 Project Structure
📁 energy-forecasting-project
│── 📄 main.py                # Main project code
│── 📄 README.md             # Project documentation
│── 📄 requirements.txt      # Required libraries
│── 📁 models                # Saved models
│── 📁 data                  # Dataset (optional)
⚙️ Installation & Setup
Step 1: Clone the repository
git clone https://github.com/your-username/energy-forecasting-project.git
cd energy-forecasting-project
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the project
python main.py
📊 How It Works
Load dataset from Kaggle using KaggleHub
Perform data preprocessing and normalization
Convert data into time-series sequences
Train LSTM deep learning model
Evaluate model performance (RMSE, R²)
Visualize predictions using graphs
Forecast future energy consumption
📈 Output
Graph showing energy consumption trends
Actual vs Predicted comparison
Scatter plot for prediction accuracy
Future 24-hour energy forecast
