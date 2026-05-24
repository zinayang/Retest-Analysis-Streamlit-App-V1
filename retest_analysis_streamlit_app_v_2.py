"""
复测数据自动分析系统（兼容无Streamlit环境版）
-------------------------------------------------
修复内容：
1. 修复 ModuleNotFoundError: streamlit
2. 增加 streamlit 缺失时的降级模式
3. 增强字段兼容识别
4. 增强 safe_float 对异常值处理
5. 修复 uploaded_file 重复读取导致的问题
6. 增加测试数据与测试函数
7. 修复空DataFrame统计问题
8. 修复中文字段兼容性
"""

import pandas as pd
import numpy as np
from io import BytesIO


# ======================================
# Streamlit兼容处理
# ======================================
STREAMLIT_AVAILABLE = True

try:
    import streamlit as st
    import plotly.express as px
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

    class MockStreamlit:
        def error(self, x):
            print(f"ERROR: {x}")

        def success(self, x):
            print(f"SUCCESS: {x}")

        def info(self, x):
            print(f"INFO: {x}")

        def warning(self, x):
            print(f"WARNING: {x}")

        def write(self, x):
            print(x)

    st = MockStreamlit()


# ======================================
# 字段自动识别
# ======================================
COLUMN_CANDIDATES = {
    "id": ["样本ID", "Sample_ID", "ID", "样本号"],
    "item": ["项目", "检测项目", "Item"],
    "result": ["原始结果", "结果", "Result"],
    "flag": ["结果标识", "标识", "Flag"],
    "time": ["测试时间", "时间", "检测时间", "Time"]
}


def find_column(df, candidates):
    """
    自动查找字段
    """

    columns = [str(c).strip() for c in df.columns]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


# ======================================
# 数值安全转换
# ======================================
def safe_float(value):
    """
    安全转换浮点数
    """

    if value is None:
        return None, "结果为空"

    if pd.isna(value):
        return None, "结果为空"

    value = str(value).strip()

    if value == "":
        return None, "结果为空"

    # 去除特殊符号
    value = value.replace(",", "")

    # 不允许的文本
    invalid_keywords = [
        "++++",
        "阴性",
        "阳性",
        "溶血",
        "NULL"
    ]

    for key in invalid_keywords:
        if key in value:
            return None, f"结果格式错误: {value}"

    try:
        return float(value), None
    except:
        return None, f"结果格式错误: {value}"


# ======================================
# 主分析函数
# ======================================
def analyze_data(df, data_type="样本"):

    result_rows = []
    invalid_rows = []

    # ==========================
    # 自动识别字段
    # ==========================
    id_col = find_column(df, COLUMN_CANDIDATES["id"])
    item_col = find_column(df, COLUMN_CANDIDATES["item"])
    result_col = find_column(df, COLUMN_CANDIDATES["result"])
    flag_col = find_column(df, COLUMN_CANDIDATES["flag"])
    time_col = find_column(df, COLUMN_CANDIDATES["time"])

    required_mapping = {
        "样本ID": id_col,
        "项目": item_col,
        "原始结果": result_col,
        "结果标识": flag_col,
        "测试时间": time_col
    }

    missing_cols = [
        k for k, v in required_mapping.items()
        if v is None
    ]

    if missing_cols:
        st.error(f"缺少必要字段: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ==========================
    # 时间处理
    # ==========================
    df = df.copy()

    df[time_col] = pd.to_datetime(
        df[time_col],
        errors='coerce'
    )

    # ==========================
    # 无效时间记录
    # ==========================
    invalid_time_df = df[df[time_col].isna()]

    for _, row in invalid_time_df.iterrows():
        invalid_rows.append({
            "数据类型": data_type,
            "样本ID": row.get(id_col, ""),
            "项目": row.get(item_col, ""),
            "原因": "测试时间缺失或格式错误"
        })

    df = df[df[time_col].notna()].copy()

    # 无有效数据
    if len(df) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(invalid_rows)
        )

    # ==========================
    # 排序
    # ==========================
    df = df.sort_values(
        by=[id_col, item_col, time_col]
    )

    # ==========================
    # 分组分析
    # ==========================
    grouped = df.groupby([id_col, item_col])

    for (sample_id, item), group in grouped:

        group = group.sort_values(time_col).reset_index(drop=True)

        # ==========================
        # 找到所有R记录
        # ==========================
        r_group = group[
            group[flag_col].astype(str).str.strip() == 'R'
        ]

        if len(r_group) == 0:
            continue

        # ==========================
        # 首次原始结果作为基准值
        # ==========================
        first_result_raw = group.iloc[0][result_col]

        first_result, err = safe_float(first_result_raw)

        if err:
            invalid_rows.append({
                "数据类型": data_type,
                "样本ID": sample_id,
                "项目": item,
                "原因": f"首次原始结果异常: {err}"
            })
            continue

        if first_result == 0:
            invalid_rows.append({
                "数据类型": data_type,
                "样本ID": sample_id,
                "项目": item,
                "原因": "首次原始结果为0，无法计算偏差"
            })
            continue

        # ==========================
        # 遍历R记录
        # ==========================
        for _, row in r_group.iterrows():

            retest_raw = row[result_col]

            retest_result, err = safe_float(retest_raw)

            if err:
                invalid_rows.append({
                    "数据类型": data_type,
                    "样本ID": sample_id,
                    "项目": item,
                    "原因": f"复测结果异常: {err}"
                })
                continue

            deviation = abs(retest_result - first_result) / abs(first_result) * 100

            result_rows.append({
                "数据类型": data_type,
                "样本ID": sample_id,
                "项目": item,
                "原结果": first_result,
                "复测结果": retest_result,
                "偏差(%)": round(deviation, 2),
                "是否超20%": "是" if deviation > 20 else "否",
                "测试时间": row[time_col]
            })

    result_df = pd.DataFrame(result_rows)

    if len(result_df) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(invalid_rows)
        )

    over20_df = result_df[
        result_df["偏差(%)"] > 20
    ].copy()

    invalid_df = pd.DataFrame(invalid_rows)

    return result_df, over20_df, invalid_df


# ======================================
# 统计函数
# ======================================
def calculate_statistics(result_df, over20_df):

    # ==========================
    # 空结果保护
    # ==========================
    if result_df is None:
        result_df = pd.DataFrame()

    if over20_df is None:
        over20_df = pd.DataFrame()

    # ==========================
    # 总复测记录数
    # ==========================
    total_records = len(result_df)

    # ==========================
    # 异常记录数
    # ==========================
    abnormal_records = len(over20_df)

    # ==========================
    # 异常记录占比
    # ==========================
    if total_records == 0:
        record_ratio = 0
    else:
        record_ratio = round(
            abnormal_records / total_records * 100,
            2
        )

    # ==========================
    # 总复测样本数
    # ==========================
    if (
        len(result_df) == 0 or
        "样本ID" not in result_df.columns
    ):
        total_samples = 0

    else:
        total_samples = result_df["样本ID"].nunique()

    # ==========================
    # 异常样本数
    # ==========================
    if (
        len(over20_df) == 0 or
        "样本ID" not in over20_df.columns
    ):
        abnormal_samples = 0

    else:
        abnormal_samples = over20_df["样本ID"].nunique()

    # ==========================
    # 异常样本占比
    # ==========================
    if total_samples == 0:
        sample_ratio = 0

    else:
        sample_ratio = round(
            abnormal_samples / total_samples * 100,
            2
        )

    return {
        "总复测记录数": total_records,
        "异常记录数": abnormal_records,
        "异常记录占比": record_ratio,
        "总复测样本数": total_samples,
        "异常样本数": abnormal_samples,
        "异常样本占比": sample_ratio
    }

# ======================================
# 测试函数
# ======================================
def run_tests():

    test_df = pd.DataFrame({
        "样本ID": [
            "S001",
            "S001",
            "S001",
            "S002",
            "S002"
        ],
        "项目": [
            "ALT",
            "ALT",
            "ALT",
            "AST",
            "AST"
        ],
        "原始结果": [
            20,
            25,
            24,
            10,
            "++++"
        ],
        "结果标识": [
            "",
            "R",
            "R",
            "",
            "R"
        ],
        "测试时间": [
            "2026-05-20 09:00",
            "2026-05-20 10:00",
            "2026-05-20 11:00",
            "2026-05-20 09:00",
            "2026-05-20 10:00"
        ]
    })

    result_df, over20_df, invalid_df = analyze_data(test_df)

    print("\n=== 测试结果 ===")
    print(result_df)

    print("\n=== 超20% ===")
    print(over20_df)

    print("\n=== 无法分析 ===")
    print(invalid_df)

    stats = calculate_statistics(result_df, over20_df)

    print("\n=== 统计 ===")
    print(stats)


# ======================================
# Streamlit 网页入口
# ======================================
def run_streamlit_app():

    st.set_page_config(
        page_title="复测数据自动分析系统",
        page_icon="🧪",
        layout="wide"
    )

    st.title("🧪 复测数据自动分析系统")

    st.markdown(
        "上传Excel后自动分析复测数据、偏差率及异常占比"
    )

    uploaded_file = st.file_uploader(
        "上传Excel文件",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.info("请上传Excel文件")
        return

    excel_file = pd.ExcelFile(uploaded_file)

    st.success(f"成功读取 {len(excel_file.sheet_names)} 个Sheet")

    all_results = []
    all_over20 = []
    all_invalid = []

    # ==========================
    # 分类结果容器
    # ==========================
    sample_results = []
    sample_over20 = []

    qc_l_results = []
    qc_l_over20 = []

    qc_h_results = []
    qc_h_over20 = []

    # ==========================
    # 遍历sheet
    # ==========================
    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name
        )

        # ==========================
        # 判断数据类型
        # ==========================
        if "质控L" in sheet_name or "QC-L" in sheet_name:
            data_type = "QC-L"

        elif "质控H" in sheet_name or "QC-H" in sheet_name:
            data_type = "QC-H"

        else:
            data_type = "样本"

        result_df, over20_df, invalid_df = analyze_data(
            df,
            data_type=data_type
        )

        # ==========================
        # 分类存储
        # ==========================
        if data_type == "样本":

            if len(result_df) > 0:
                sample_results.append(result_df)

            if len(over20_df) > 0:
                sample_over20.append(over20_df)

        elif data_type == "QC-L":

            if len(result_df) > 0:
                qc_l_results.append(result_df)

            if len(over20_df) > 0:
                qc_l_over20.append(over20_df)

        elif data_type == "QC-H":

            if len(result_df) > 0:
                qc_h_results.append(result_df)

            if len(over20_df) > 0:
                qc_h_over20.append(over20_df)

        # 无法分析数据
        if len(invalid_df) > 0:
            all_invalid.append(invalid_df)

    # ==========================
    # 合并结果
    # ==========================
    sample_result_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()

    sample_over20_df = pd.concat(sample_over20, ignore_index=True) if sample_over20 else pd.DataFrame()

    qc_l_result_df = pd.concat(qc_l_results, ignore_index=True) if qc_l_results else pd.DataFrame()

    qc_l_over20_df = pd.concat(qc_l_over20, ignore_index=True) if qc_l_over20 else pd.DataFrame()

    qc_h_result_df = pd.concat(qc_h_results, ignore_index=True) if qc_h_results else pd.DataFrame()

    qc_h_over20_df = pd.concat(qc_h_over20, ignore_index=True) if qc_h_over20 else pd.DataFrame()

    result_df = pd.concat([
        sample_result_df,
        qc_l_result_df,
        qc_h_result_df
    ], ignore_index=True)

    over20_df = pd.concat([
        sample_over20_df,
        qc_l_over20_df,
        qc_h_over20_df
    ], ignore_index=True) if (
        len(sample_over20_df) > 0 or
        len(qc_l_over20_df) > 0 or
        len(qc_h_over20_df) > 0
    ) else pd.DataFrame()
    # ==========================
    # 无法分析数据汇总
    # ==========================
    invalid_df = pd.concat(all_invalid, ignore_index=True) if all_invalid else pd.DataFrame()

    # ==========================
    # 分别统计（样本/QC-L/QC-H）
    # ==========================
    sample_stats = calculate_statistics(sample_result_df, sample_over20_df)
    qc_l_stats = calculate_statistics(qc_l_result_df, qc_l_over20_df)
    qc_h_stats = calculate_statistics(qc_h_result_df, qc_h_over20_df)

    # ==========================
    # 页面统计展示
    # ==========================
    st.divider()
    st.header("📊 统计结果")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "样本异常占比",
        f"{sample_stats['异常样本占比']}%"
    )

    col2.metric(
        "QC-L异常占比",
        f"{qc_l_stats['异常样本占比']}%"
    )

    col3.metric(
        "QC-H异常占比",
        f"{qc_h_stats['异常样本占比']}%"
    )

    # ==========================
    # 样本异常结果
    # ==========================
    st.divider()
    st.header("🚨 样本偏差超过20%")

    if len(sample_over20_df) > 0:

        st.dataframe(
            sample_over20_df,
            use_container_width=True
        )

    else:
        st.success("未发现样本偏差超过20%")

    # ==========================
    # QC-L异常结果
    # ==========================
    st.divider()
    st.header("🧪 QC-L偏差超过20%")

    if len(qc_l_over20_df) > 0:

        st.dataframe(
            qc_l_over20_df,
            use_container_width=True
        )

    else:
        st.success("未发现QC-L偏差超过20%")

    # ==========================
    # QC-H异常结果
    # ==========================
    st.divider()
    st.header("🧪 QC-H偏差超过20%")

    if len(qc_h_over20_df) > 0:

        st.dataframe(
            qc_h_over20_df,
            use_container_width=True
        )

    else:
        st.success("未发现QC-H偏差超过20%")

    # ==========================
    # 全部异常结果
    # ==========================
    over20_df = pd.concat([
        sample_over20_df,
        qc_l_over20_df,
        qc_h_over20_df
    ], ignore_index=True) if (
        len(sample_over20_df) > 0 or
        len(qc_l_over20_df) > 0 or
        len(qc_h_over20_df) > 0
    ) else pd.DataFrame()

    st.divider()
    st.header("📋 全部异常汇总")
    # ==========================
    st.divider()
    st.header("🚨 偏差超过20%的数据")

    if len(over20_df) > 0:

        st.dataframe(
            over20_df,
            use_container_width=True
        )

    else:
        st.success("未发现偏差超过20%的数据")

    # ==========================
    # 无法分析数据
    # ==========================
    st.divider()
    st.header("⚠️ 无法分析的数据")

    if len(invalid_df) > 0:

        st.dataframe(
            invalid_df,
            use_container_width=True
        )

    else:
        st.success("无无法分析数据")

    # ==========================
    # 导出Excel
    # ==========================
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        # ==========================
        # 样本结果
        # ==========================
        sample_result_df.to_excel(
            writer,
            sheet_name='样本全部分析',
            index=False
        )

        sample_over20_df.to_excel(
            writer,
            sheet_name='样本超20%',
            index=False
        )

        # ==========================
        # QC-L结果
        # ==========================
        qc_l_result_df.to_excel(
            writer,
            sheet_name='QC-L全部分析',
            index=False
        )

        qc_l_over20_df.to_excel(
            writer,
            sheet_name='QC-L超20%',
            index=False
        )

        # ==========================
        # QC-H结果
        # ==========================
        qc_h_result_df.to_excel(
            writer,
            sheet_name='QC-H全部分析',
            index=False
        )

        qc_h_over20_df.to_excel(
            writer,
            sheet_name='QC-H超20%',
            index=False
        )

        # ==========================
        # 无法分析数据
        # ==========================
        invalid_df.to_excel(
            writer,
            sheet_name='无法分析数据',
            index=False
        )

        # ==========================
        # Excel格式修复
        # ==========================
        for current_sheet in writer.sheets:

            worksheet = writer.sheets[current_sheet]

            for col in worksheet.iter_cols():

                header = col[0].value

                if header in ['样本ID', '项目ID', 'ID']:

                    for cell in col:
                        cell.number_format = '@'

    output.seek(0)

    st.download_button(
        label="📥 下载分析结果Excel",
        data=output,
        file_name="复测分析结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ======================================
# 启动入口
# ======================================
if __name__ == "__main__":

    if STREAMLIT_AVAILABLE:
        run_streamlit_app()

    else:
        print("当前环境未安装Streamlit")
        print("请部署到Streamlit Cloud网页环境")
        print("或安装依赖后运行：")
        print("pip install streamlit pandas openpyxl plotly")
        print("streamlit run app.py")


# ======================================
# requirements.txt
# ======================================
"""
streamlit
pandas
numpy
openpyxl
plotly
"""



