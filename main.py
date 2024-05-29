import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 设置页面标题
st.title("WISERS大众点评餐饮数据分析系统")

# 上传文件
uploaded_file = st.file_uploader("上传Excel或CSV文件", type=["xlsx", "csv"], accept_multiple_files=False)


def avg_price(df):
    df.replace('空', np.nan, inplace=True)
    df['Price_per_person'] = pd.to_numeric(df['Price_per_person'], errors='coerce')
    avg_price_per_merchant = df.groupby('Merchant')['Price_per_person'].mean().reset_index()
    avg_price_per_merchant.columns = ['店铺', '人均价格']
    # 按平均人均价格从高到低排序
    avg_price_per_merchant = avg_price_per_merchant.sort_values(by='人均价格', ascending=False)
    st.write("店铺平均人均价格统计：")
    st.dataframe(avg_price_per_merchant, width=800, height=300)
    # 可视化平均人均价格
    st.subheader("可视化：")
    fig = px.bar(avg_price_per_merchant, x='店铺', y='人均价格', height=200)
    fig.update_layout(
        height=800,  # 设置图表高度
        xaxis=dict(),  # 旋转x轴标签，自动调整边距
        yaxis=dict(title='人均价格 (元)'),  # 设置y轴标题
        font=dict(size=14)  # 设置默认字体大小
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)


def price_value(df):
    df.replace('空', np.nan, inplace=True)
    df['Price_per_person'] = pd.to_numeric(df['Price_per_person'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Cost_performance_index'] = df['Rating'] / df['Price_per_person'] * 20

    avg_cpi_per_merchant = df.groupby('Merchant')['Cost_performance_index'].mean().reset_index()
    avg_cpi_per_merchant.columns = ['Merchant', 'Avg_Cost_performance_index']

    # 按性价比指数从高到低排序
    avg_cpi_per_merchant = avg_cpi_per_merchant.sort_values(by='Avg_Cost_performance_index', ascending=False)

    # 可视化性价比指数
    st.write("可视化性价比指数：")
    fig = px.bar(avg_cpi_per_merchant, x='Merchant', y='Avg_Cost_performance_index', title='每个店铺的性价比指数')
    fig.update_layout(
        height=800,  # 设置图表高度
        xaxis=dict(),  # 旋转x轴标签，自动调整边距
        yaxis=dict(title='人均价格 (元)'),  # 设置y轴标题
        font=dict(size=14)  # 设置默认字体大小
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)


def rate(data, y: str):
    avg_scores_per_merchant = data
    fig = px.box(avg_scores_per_merchant, x='Merchant', y=y, title='店铺口味评分离散图')
    fig.add_shape(type='line', x0=-0.5, x1=len(avg_scores_per_merchant) - 0.5,
                  y0=avg_scores_per_merchant[y].mean(),
                  y1=avg_scores_per_merchant[y].mean(),
                  line=dict(color='red', width=2, dash='dash'))
    fig.add_annotation(x=len(avg_scores_per_merchant) - 1, y=avg_scores_per_merchant[y].mean(),
                       text=f"Overall Avg: {avg_scores_per_merchant[y].mean():.2f}",
                       showarrow=True, arrowhead=2)
    fig.update_layout(
        height=600,  # 设置图表高度
        xaxis=dict(title='店铺名称', automargin=True),  # 旋转x轴标签，自动调整边距
        yaxis=dict(title='平均评分'),  # 设置y轴标题
        font=dict(size=14)  # 设置默认字体大小
    )
    st.plotly_chart(fig)

def selectmerchant(key):
    merchants_price = st.multiselect("选择店铺", ["全选"] + merchants,
                                     default="全选", key=key)
    if "全选" in merchants_price:
        df_price =df
    else:
        df_price = df[df['Merchant'].isin(merchants_price)]
    return df_price


if uploaded_file is not None:
    # 检测文件类型并读取数据
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding='gbk', encoding_errors='ignore')
    # 数据预处理
    df = df.dropna()
    st.write("删除缺失值所在行后数据：")
    st.dataframe(df, width=800, height=300)
    st.divider()

    merchants =list(df['Merchant'].unique())
    # 统计每个店铺的平均人均价格
    st.header("店铺人均价格：")
    df_price= selectmerchant("price")
    if st.button("显示结果", key="pricebut"):
        avg_price(df_price)
        st.divider()

    #性价比指数
    st.header("店铺性价比指数：")
    df_value = selectmerchant("value")
    if st.button("显示结果", key="valuebut"):
        price_value(df_value)

    st.divider()

    # 计算每个店铺的口味、服务和环境均分
    st.header("店铺的口味、服务、环境评分：")
    df=selectmerchant("rate")
    if st.button("显示结果",key="ratebut"):
        st.write('平均分：')
        df['Score_taste'] = pd.to_numeric(df['Score_taste'], errors='coerce')
        df['Score_service'] = pd.to_numeric(df['Score_service'], errors='coerce')
        df['Score_environment'] = pd.to_numeric(df['Score_environment'], errors='coerce')
        avg_scores_per_merchant = df.groupby('Merchant').agg({
            'Score_taste': 'mean',
            'Score_service': 'mean',
            'Score_environment': 'mean'
        }).reset_index()
        st.dataframe(avg_scores_per_merchant, width=800, height=300)
        avg_scores_per_merchant.columns = ['Merchant', 'Avg_Score_taste', 'Avg_Score_service', 'Avg_Score_environment']

        # 生成店铺口味评分离散图

        rate(avg_scores_per_merchant, 'Avg_Score_taste')
        #生成店铺服务评分离散图

        rate(avg_scores_per_merchant, 'Avg_Score_service')
        # 生成店铺环境评分离散图

        rate(avg_scores_per_merchant, 'Avg_Score_environment')

        # 生成雷达图，展示每个店铺口味、服务和环境的均分
        st.subheader("店铺口味、服务和环境均分雷达图：")

        selected_merchant = st.selectbox("选择一个店铺查看雷达图", avg_scores_per_merchant['Merchant'].tolist())

        radar_fig = go.Figure()

        for i, row in avg_scores_per_merchant.iterrows():
            if row['Merchant'] == selected_merchant:
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row['Avg_Score_taste'], row['Avg_Score_service'], row['Avg_Score_environment']],
                    theta=['口味', '服务', '环境'],
                    fill='toself',
                    name=row['Merchant']
                ))

        radar_fig.update_layout(
            height=600,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            showlegend=True,
            title="每个店铺的口味、服务和环境均分雷达图"
        )

        st.plotly_chart(radar_fig)
    st.divider()
