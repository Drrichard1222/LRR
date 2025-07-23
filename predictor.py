# 导入 Streamlit 库，用于构建Web应用
import streamlit as st

# 导入joblib库，用于加载和保存机器学习模型
import joblib

# 导入NumPy库，用于数值计算
import numpy as np

# 导入Pandas库，用于数据处理和操作
import pandas as pd

# 导入SHAP库，用于解释机器学习模型的预测
import shap

# 导入Matplotlib库，用于数据可视化
import matplotlib.pyplot as plt  # 修正了这里的空格问题

# 从LIME库中导入LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer

# 加载训练好的模型（LRR.pkl）
model = joblib.load('LRR.pkl')

# 从X_test.csv文件加载测试数据，以便用于LIME解释器
X_test = pd.read_csv('X_test.csv')

# 定义特征名称，对应数据集中的列名
feature_names = [
    "Sodium_max",  # Sodium_max
    "Vasopressor",  # Vasopressor
    "APSIII",  # APSIII
    "Age",  # Age
    "Calcium_max",  # Calcium_max
    "Respiratory_rate_min",  # Respiratory_rate_min
    "Ventilation",  # Ventilation
    "Dementia",  # Dementia
    "Metastatic_solid_tumor",  # Metastatic_solid_tumor
]

# Streamlit页面
st.title("In-hospital mortality risk of TBI patients")  # 设置网页标题

# Sodium_max：数值输入框
Sodium_max = st.number_input("Sodium_max:", min_value=0, max_value=500, value=12)

# Vasopressor：选择框（修正了显示标签）
Vasopressor = st.selectbox("Vasopressor:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# APSIII：数值输入框
APSIII = st.number_input("APSIII:", min_value=0, max_value=100, value=10)

# Age：数值输入框
Age = st.number_input("Age:", min_value=0, max_value=100, value=65)

# Calcium_max：数值输入框
Calcium_max = st.number_input("Calcium_max:", min_value=0, max_value=20, value=20)

# Respiratory_rate_min：数值输入框
Respiratory_rate_min = st.number_input("Respiratory_rate_min:", min_value=0, max_value=30, value=20)

# Ventilation：选择框（修正了显示标签）
Ventilation = st.selectbox("Ventilation:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Dementia：选择框（修正了显示标签）
Dementia = st.selectbox("Dementia:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Metastatic_solid_tumor：选择框（修正了显示标签）
Metastatic_solid_tumor = st.selectbox("Metastatic_solid_tumor:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 处理输入数据并进行预测
feature_values = [Sodium_max, Vasopressor, APSIII, Age, Calcium_max, Respiratory_rate_min, Ventilation, Dementia, Metastatic_solid_tumor]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为NumPy数组，适用于模型输入

# 当用户点击"Predict"按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：低风险，1：高风险）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]
    
    # 显示预测结果（修正了显示格式）
    risk_status = "high risk" if predicted_class == 1 else "low risk"
    st.write(f"**Predicted Risk Status:** {risk_status} of in-hospital mortality")
    st.write(f"**Probability of Low Risk:** {predicted_proba[0]:.4f}")
    st.write(f"**Probability of High Risk:** {predicted_proba[1]:.4f}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, this patient has a high risk of in-hospital mortality. "
            f"The model predicts a {probability:.1f}% probability of high risk. "
            "Immediate clinical attention and intervention are recommended."
        )
    else:
        advice = (
            f"According to our model, this patient has a low risk of in-hospital mortality. "
            f"The model predicts a {probability:.1f}% probability of low risk. "
            "Standard monitoring and care are advised."
        )
    st.write(advice)

    # SHAP解释部分（完全重写）
    st.subheader("SHAP Explanation")
    
    # 创建SHAP解释器 - 使用KernelExplainer
    explainer = shap.KernelExplainer(model.predict_proba, X_test.sample(50, random_state=42))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(features)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 根据预测类别选择显示哪个类别的解释
    class_idx = 1 if predicted_class == 1 else 0
    
    # 生成瀑布图
    shap.waterfall_plot(explainer.expected_value[class_idx], 
                       shap_values[class_idx][0], 
                       feature_names=feature_names,
                       show=False)
    
    plt.title(f"SHAP Values for {'High Risk' if predicted_class == 1 else 'Low Risk'} Prediction")
    plt.tight_layout()
    
    # 在Streamlit中显示图表
    st.pyplot(fig)
    plt.clf()  # 清除图形避免后续绘图重叠

    # LIME解释部分
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Low risk', 'High risk'],
        mode='classification'
    )

    # 解释当前实例
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # 显示LIME解释（不显示特征值表格）
    lime_html = lime_exp.as_html(show_table=False)  # 禁用特征值表格
    st.components.v1.html(lime_html, height=800, scrolling=True)