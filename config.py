# config.py — настройки, пути, целевые переменные

# 🎯 Целевые переменные
EGE_TARGET = (
    "Средний балл ЕГЭ студентов, принятых по результатам ЕГЭ на обучение по очной форме "
    "по программам бакалавриата и специалитета за счет средств соответствующих бюджетов "
    "бюджетной системы РФ"
)

NIOKR_TARGET = (
    "Общий объем научно-исследовательских и опытно-конструкторских работ (далее – НИОКР)"
)

# 🧠 Пути к моделям
MODEL_PATHS = {
    "ege": "models/model_ege.pkl",
    "niokr": "models/model_niokr.pkl",
    "ege_linear": "models/model_ege_linear.pkl",
    "niokr_linear": "models/model_niokr_linear.pkl",
    "kmeans": "models/model_kmeans.pkl"
}

# 📂 Пути к данным
DATA_PATHS = {
    "processed": "data/processed_data.csv",
    "filtered": "data/filtered_data.csv",
    "export": "data/filtered_data_export.csv"
}

# 🖼️ Пути к отчётам и графикам
REPORT_PATH = "edu_monitor_report.docx"
PLOTS_DIR = "plots"

# ⚙️ Гиперпараметры модели Random Forest
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

# 🔧 Общие настройки
RANDOM_STATE = 42
N_FEATURES_SELECT = 20
DEFAULT_TEST_SIZE = 0.2