# Physiologically-Informed Predictability of a Teammate's Future Actions Forecasts Team Performance

This repository contains code for analyzing **synchrony, predictability, and performance** in multi-human teams. The analysis spans multiple modalities, including **EEG, pupil size, speech events, and remote controller actions**, alongside transformer-based modeling for remote controller action prediction.

## 📌 Features

- **ISC Performance Analysis:** Computes **Inter-Subject Correlation (ISC)** for different data modalities, such as EEG, pupil size, and speech events, and examines their correlation with team performance.
- **Predicting Action Performance:** Analyzes the correlation between predicted remote controller actions and participants' actual actions.
- **Preprocessing Pipelines:** Provides tools to clean, normalize, and prepare data for analysis.
- **Transformer-Based Modeling:** Implements a **transformer-based predictive model** to forecast future remote controller actions of individual participants based on their team members' past physiological and behavioral data.
- **Plotting Utilities:** Includes visualization scripts for performance and synchrony analysis.

## 🛠 Installation

Ensure you have Python 3.x installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/YinuoQ/predictability_performance_and_ISC.git
cd synchrony_and_performance
pip install -r requirements.txt 
```

## 🚀 Usage

### 1️⃣ Running ISC Analysis
To compute performance correlation with ISCs using EEG, pupil size, and speech event data:

```bash
cd ISC_performance
python eeg_ISC_performance.py
python pupil_ISC_performance.py
python speech_event_ISC_performance.py
python action_ISC_performance.py
```

To plot ISC correlation with team performance:

```bash
python synchrony_plot.py
```

### 2️⃣ Predicting Actions Using the Transformer Model
To generate training and testing data:

```bash
cd transformer/data
python generate_transformer_data.py
```

To train the transformer model on preprocessed data:

```bash
bash ./train.sh
```

To test the trained model:

```bash
bash ./test.sh
```

### 3️⃣ Computing Predictability and Its Correlation with Team Performance
To compute action predictability and its correlation with team performance:

```bash
cd predicted_action_performance
python action_performance.py
```

To plot predictability results:

```bash
python plot_predictability.py
```

To compute the correlation between performance, ISC, and predictability using mixed-effects models:

```bash
python correlations.py
```

## 📊 Data

The dataset is available at:  
[Dropbox Link](https://www.dropbox.com/scl/fo/7t7cfaad9z6867yy40da6/ACPsNFnyYmzhNECURkwuStI?rlkey=2qjwzagkm14z973eq6hpa0qou&st=mo6211uy&dl=0)

## 🔧 Configuration

The transformer model has configurations stored in `configs/`, categorized into:
  - `pitch/`
  - `thrust/`
  - `yaw/`

Modify these configurations based on your experiment requirements.

## 🏗 Contributing

If you'd like to contribute, feel free to submit a **pull request**. Please ensure your code adheres to best practices and is well-documented.

## 📜 License

This repository follows the **MIT License**. See [LICENSE](LICENSE) for details.