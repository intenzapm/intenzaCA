import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
import requests
import base64
from io import BytesIO
import numpy as np
import openai
import time
import plotly.io as pio
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
import html

openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


# 去白底函數
def remove_white_background(img):
    img = img.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_threshold = 240
    mask = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
    data[mask] = [255, 255, 255, 0]
    return Image.fromarray(data)

# 統一尺寸函數
def resize_with_padding(img, target_size=(500, 500)):
    img = img.convert("RGBA")
    old_size = img.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    new_img = Image.new("RGBA", target_size, (255, 255, 255, 0))
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_img.paste(img, paste_position)
    return new_img

# ChatGPT 函數
def ask_chatgpt(prompt):
    client = openai.Client(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 頁面設定
st.set_page_config(page_title="INTENZA 競品分析工具", layout="wide")
st.title("💡 INTENZA 競品分析數位化轉型工具")


with st.sidebar.expander("📂 請上傳 CSV 檔案", expanded=False):
    uploaded_file = st.file_uploader("上傳", type=["csv"], label_visibility="collapsed")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.expander("📄 原始資料表格（點擊展開/收合）", expanded=False):
        st.dataframe(df, use_container_width=True)

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    品牌_fix_map = {"LifeFitness": "Life Fitness", "TRUE": "True Fitness", "VISION": "Vision Fitness"}
    df["品牌"] = df["品牌"].replace(品牌_fix_map)

    品牌_logos = {
        "Life Fitness": "logos/LF.jpg",
        "Matrix": "logos/matrix.jpg",
        "Precor": "logos/PRECOR.jpg",
        "Technogym": "logos/TG.jpg",
        "True Fitness": "logos/true.jpg",
        "Vision Fitness": "logos/VISON.jpg"
    }

    df["label"] = df["產品型號"]
    model_options = (df["品牌"] + " - " + df["產品型號"]).drop_duplicates().tolist()

    st.sidebar.header("⚙️ 比較設定")
    selected_models = st.sidebar.multiselect("🏷️ 選擇最多 5 個品牌與機型", model_options, max_selections=5)
    selected_numeric_cols = st.sidebar.multiselect("📈 選擇要比較的數值欄位（可多選）", numeric_columns, default=numeric_columns[:1])

    chart_width, chart_height, bottom_margin = 900, 500, 170
    logo_sizey, logo_y_offset, product_img_sizey, product_img_y_offset = 0.14, -0.10, 0.30, -0.33

    chart_type_map = {}
    if selected_numeric_cols:
        with st.sidebar.expander("📐 圖表類型設定", expanded=True):
            for col in selected_numeric_cols:
                chart_type_map[col] = st.selectbox(f"圖表類型 - {col}", ["長條圖（Bar）", "折線圖（Line）", "散點圖（Scatter）"], key=f"{col}_chart")

    # 非數值欄位全選邏輯
    st.sidebar.write("📋 其他類別欄位選擇")
    all_text_cols = [col for col in non_numeric_columns if col not in ['品牌', '產品型號', 'label', '圖片網址']]
    select_all = st.sidebar.checkbox("全選")
    if select_all:
        selected_text_cols = st.sidebar.multiselect("選擇欄位", all_text_cols, default=all_text_cols)
    else:
        selected_text_cols = st.sidebar.multiselect("選擇欄位", all_text_cols)

    chart_width, chart_height, bottom_margin = 900, 500, 170
    logo_sizey, logo_y_offset, product_img_sizey, product_img_y_offset = 0.14, -0.10, 0.30, -0.33

    if selected_models:
        selected_品牌s = [x.split(" - ")[0] for x in selected_models]
        selected_products = [x.split(" - ")[1] for x in selected_models]
        filtered_df = df[df["品牌"].isin(selected_品牌s) & df["產品型號"].isin(selected_products)].drop_duplicates(subset=["品牌", "產品型號"])

        for col in selected_numeric_cols:
            chart_data = filtered_df[["label", "品牌", "產品型號", col, "圖片網址"]].copy()
            chart_data[col] = chart_data[col].fillna(0)
            chart_data = chart_data.sort_values(by=col)  # 排序
        
            st.subheader(f"📊 【{col}】比較圖（共 {len(chart_data)} 筆）")
        
            fig = go.Figure()
            chart_type = chart_type_map.get(col, "長條圖（Bar）")
        
            for 品牌 in chart_data["品牌"].unique():
                subset = chart_data[chart_data["品牌"] == 品牌]
        
                if chart_type == "長條圖（Bar）":
                    fig.add_trace(go.Bar(
                        x=subset["label"],
                        y=subset[col],
                        name=品牌,
                        text=subset[col],
                        textposition='outside',
                        textfont=dict(size=11)
                    ))
                elif chart_type == "折線圖（Line）":
                    fig.add_trace(go.Scatter(
                        x=subset["label"],
                        y=subset[col],
                        mode='lines+markers+text',
                        name=品牌,
                        text=subset[col],
                        textposition='top center',
                        textfont=dict(size=11)
                    ))
                elif chart_type == "散點圖（Scatter）":
                    fig.add_trace(go.Scatter(
                        x=subset["label"],
                        y=subset[col],
                        mode='markers+text',
                        name=品牌,
                        text=subset[col],
                        textposition='top center',
                        textfont=dict(size=11)
                    ))
        
            fig.update_layout(
                width=chart_width,
                height=chart_height,
                margin=dict(l=50, r=50, t=50, b=bottom_margin),
                xaxis=dict(categoryorder="array", categoryarray=chart_data["label"].tolist())
            )

        
            for _, row in chart_data.iterrows():
                label, 品牌 = row["label"], row["品牌"]
                logo_path = 品牌_logos.get(品牌)
                if logo_path and os.path.exists(logo_path):
                    img = Image.open(logo_path)
                    fig.add_layout_image(dict(
                        source=img, x=label, y=logo_y_offset,
                        xref="x", yref="paper",
                        sizex=1, sizey=logo_sizey,
                        xanchor="center", yanchor="top",
                        layer="above"
                    ))
        
            for _, row in chart_data.iterrows():
                label, img_url = row["label"], row["圖片網址"]
                if isinstance(img_url, str) and img_url.startswith("http"):
                    try:
                        img = Image.open(BytesIO(requests.get(img_url).content))
                        img = remove_white_background(img)
                        img = resize_with_padding(img)
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        img_base64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                        fig.add_layout_image(dict(
                            source=img_base64, x=label, y=product_img_y_offset,
                            xref="x", yref="paper",
                            sizex=1, sizey=product_img_sizey,
                            xanchor="center", yanchor="top",
                            layer="above"
                        ))
                    except:
                        pass
        
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        st.write("")
        st.write("")
        st.subheader("📋 非數值規格比較表格")
        
        # 重組資料
        filtered_df = filtered_df.reset_index(drop=True)
        transposed_data = []
        for spec in selected_text_cols:
            row = [spec]
            for _, product_row in filtered_df.iterrows():
                row.append(product_row[spec])
            transposed_data.append(row)
        
        # 表頭兩層
        brand_row = [""] + [row["品牌"] for _, row in filtered_df.iterrows()]
        model_row = ["規格名稱"] + [row["產品型號"] for _, row in filtered_df.iterrows()]
        
        st.markdown(f"""
        <style>
        .custom-table-container {{
            width: {chart_width}px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        @media print {{
        
            .custom-table-container {{
                width: 100% !important;
            }}
        
            .plotly-graph-div {{
                width: 100% !important;
                max-width: 100% !important;
            }}
        
            .stApp {{
                width: 100% !important;
                max-width: 100% !important;
                overflow: visible !important;
            }}
        
            img {{
                max-width: 100% !important;
                height: auto !important;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)


        
        # 組出 HTML 表格
        html_table = f"""
        <style>
        .custom-table-container {{
            width: {chart_width}px;
            margin-left: auto;
            margin-right: auto;
        }}
        .custom-table {{
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
        }}
        .custom-table th, .custom-table td {{
            border: none;
            padding: 8px;
            text-align: center;
            word-wrap: break-word;
            font-size: 14px;
        }}
        .custom-table th:first-child, .custom-table td:first-child {{
            width: 100px;
            color: rgb(245,245,245);
            font-weight: bold;
        }}
        .custom-table th {{
            color: rgb(188,188,188);
            font-weight: bold;
            background-color: transparent;
        }}
        .custom-table td {{
            color: rgb(188,188,188);
            background-color: transparent;
        }}
        </style>
        <div class="custom-table-container">
        <table class="custom-table">
        <thead>
        <tr>
        """
        
        # 第一層：品牌
        for col in brand_row:
            html_table += f"<th>{col}</th>"
        html_table += "</tr><tr>"
        
        # 第二層：型號
        for col in model_row:
            html_table += f"<th>{col}</th>"
        html_table += "</tr></thead><tbody>"
        
        # 資料內容
        for row in transposed_data:
            html_table += "<tr>"
            for cell in row:
                display = "-" if pd.isna(cell) or cell == "" else cell
                html_table += f"<td>{display}</td>"
            html_table += "</tr>"
        
        html_table += "</tbody></table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)




if uploaded_file is not None and selected_models and (selected_numeric_cols or selected_text_cols):
    
    st.write("---")
    st.subheader("🤖 ChatGPT 自動分析")

    compare_cols = ["品牌", "產品型號"] + selected_numeric_cols + selected_text_cols
    compare_df = filtered_df[compare_cols]

    # 初始化 session_state
    if "gpt_response" not in st.session_state:
        st.session_state["gpt_response"] = ""

    if st.button("請 ChatGPT 總結這次的比較結果"):
        prompt = f"""
以下為健身器材競品詳細比較資料，請依據資料客觀整理內容，並模擬專業報告或高端品牌型錄的視覺層次與排版節奏，具體規則如下：

請僅輸出乾淨、結構清晰的 HTML 片段，嚴格遵循以下規範：

【數值型規格分析】
請使用 <table> 表格結構，欄位依序為：
- 規格名稱
- 數值範圍（最小值 ~ 最大值）
- 具有最大值之品牌與型號
- 具有最小值之品牌與型號

【文字描述類規格分析】
每個規格：
- 使用 <h2> 作為規格標題
- 差異描述部分，請使用條列 <ul><li> 或簡短段落 <p> 呈現

【SWOT 分析】
每個產品請依以下格式呈現：
<h3>[產品名稱] 產品 SWOT 分析</h3>
<table>
<tr><th>面向</th><th>說明</th></tr>
<tr><td>優勢</td><td>…</td></tr>
<tr><td>劣勢</td><td>…</td></tr>
<tr><td>機會</td><td>…</td></tr>
<tr><td>威脅</td><td>…</td></tr>
</table>

【競爭力規格設計建議】
請條列具體建議，重要詞彙使用 <b> 加粗，並且若是涉及有參數的規格，給出相關參數的建議。

補充規定：
- 僅使用 <h1>、<h2>、<h3>、<table>、<tr>、<th>、<td>、<p>、<ul>、<li>、<b>
- 嚴禁輸出 <html>、<head>、<body>、<!DOCTYPE html> 等外層結構
- 嚴禁符號轉義，保留標籤原樣
- 全程繁體中文，內容簡潔利於閱讀

以下為本次重點規格：
{', '.join(selected_numeric_cols + selected_text_cols)}

資料如下：
{compare_df.to_string(index=False)}
"""

        with st.spinner("分析中..."):
            gpt_response = ask_chatgpt(prompt)
        
        st.session_state["gpt_response"] = gpt_response

    # 自定義CSS強化視覺效果
    custom_css = """
    <style>
    h1 { font-size: 34px; font-weight: bold; }
    h2 { font-size: 26px; font-weight: bold; }
    h3 { font-size: 22px; font-weight: bold; }
    p  { font-size: 16px; line-height: 1.6; }
    b  { font-weight: bold; color: #d9534f; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    if st.session_state["gpt_response"]:
        clean_text = st.session_state["gpt_response"]
        if clean_text.startswith("html\n"):
            clean_text = clean_text[len("html\n"):]
        st.markdown(clean_text, unsafe_allow_html=True)
    
    st.write("---")
    st.subheader("💬 ChatGPT 自由提問")
    user_question = st.text_input("請輸入您的問題")
    if st.button("送出問題"):
        if user_question.strip():
            with st.spinner("回覆中..."):
                st.write(ask_chatgpt(user_question))


elif uploaded_file is None:
    st.info("請上傳 CSV 檔案以開始。") 