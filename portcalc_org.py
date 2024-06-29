import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_portfolio_value(data, weights):
    return (data * weights).sum()

def calculate_yearly_cagr(values):
    yearly_returns = values.resample('Y').last().pct_change(fill_method=None)
    return yearly_returns.mean()

def calculate_mdd(values):
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return drawdown.min()

def rebalance(data, weights, frequency):
    if frequency == 'monthly':
        rebalance_dates = data.resample('M').last().index
    elif frequency == 'quarterly':
        rebalance_dates = data.resample('Q').last().index
    elif frequency == 'yearly':
        rebalance_dates = data.resample('Y').last().index
    else:
        raise ValueError("Invalid frequency. Choose 'monthly', 'quarterly', or 'yearly'.")
    
    portfolio_values = []
    
    for i in range(len(rebalance_dates)):
        if i == 0:
            start_date = data.index[0]
        else:
            start_date = rebalance_dates[i-1]
        
        end_date = rebalance_dates[i]
        period_data = data.loc[start_date:end_date]
        
        if i == 0:
            initial_portfolio_value = calculate_portfolio_value(period_data.iloc[0], weights)
            period_returns = (period_data.pct_change().fillna(0) * weights).sum(axis=1)
            period_portfolio_value = initial_portfolio_value * (1 + period_returns).cumprod()
        else:
            previous_value = portfolio_values[-1].iloc[-1]
            period_returns = (period_data.pct_change().fillna(0) * weights).sum(axis=1)
            period_portfolio_value = previous_value * (1 + period_returns).cumprod()
        
        portfolio_values.append(period_portfolio_value)
    
    return pd.concat(portfolio_values)

def calculate_monthly_returns(portfolio_value):
    monthly_values = portfolio_value.resample('M').last()
    monthly_returns = monthly_values.pct_change(fill_method=None)
    return monthly_returns

def calculate_yearly_returns(portfolio_value):
    yearly_values = portfolio_value.resample('Y').last()
    yearly_returns = yearly_values.pct_change(fill_method=None)
    return yearly_returns

def create_monthly_table(returns):
    monthly_table = returns.groupby([returns.index.year, returns.index.month]).first().unstack()
    monthly_table.index.name = 'Year'
    monthly_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return monthly_table

def create_yearly_table(returns):
    def yearly_return(x):
        return (1 + x).prod() - 1
    
    def yearly_mdd(x):
        cumulative_returns = (1 + x).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    yearly_returns = returns.groupby(returns.index.year).apply(yearly_return)
    yearly_mdd = returns.groupby(returns.index.year).apply(yearly_mdd)
    
    yearly_table = pd.DataFrame({
        'Return': yearly_returns,
        'MDD': yearly_mdd
    })
    
    return yearly_table

def format_percentage(value):
    return f"{value:+.2%}"

def plot_heatmap_horizontal(data, title, figsize=None, text_size=8):
    if figsize is None:
        figsize = (16, len(data) * 0.5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmin = data.min().min()
    vmax = data.max().max()
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data, cmap='RdYlGn', norm=norm)
    
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=text_size, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=text_size)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            value = data.iloc[i, j]
            text_color = "black" if abs(value) < (vmax - vmin) / 2 else "white"
            ax.text(j, i, f"{value:+.2%}",
                    ha="center", va="center", color=text_color, fontsize=text_size)

    ax.set_title(title, fontsize=text_size+2)
    fig.colorbar(im, ax=ax, label='Returns')
    fig.tight_layout()
    return fig

def plot_heatmap_vertical(data, title, figsize=None, text_size=8):
    if figsize is None:
        figsize = (8, len(data) * 0.25)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmin = data.min().min()
    vmax = data.max().max()
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data.T, cmap='RdYlGn', norm=norm, aspect='auto')
    
    ax.set_yticks(np.arange(len(data.columns)))
    ax.set_yticklabels(data.columns, fontsize=text_size)
    ax.set_xticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.index, fontsize=text_size, rotation=90)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            value = data.iloc[i, j]
            text_color = "black" if abs(value) < (vmax - vmin) / 2 else "white"
            ax.text(i, j, f"{value:+.2%}",
                    ha="center", va="center", color=text_color, fontsize=text_size,
                    rotation=90)

    ax.set_title(title, fontsize=text_size+2)
    fig.colorbar(im, ax=ax, label='Values')
    fig.tight_layout()
    return fig

# 자산군 후보:  SPY: S&P 500  QQQ: NASDAQ 100
# TLT: 미국장기국채  IEF: 미국중기국채  SHY: 미국단기국채
# GLD: 금 ETF  EFA: 선진국주식  EEM: 개발도상국주식 

tickers = ['SPY', 'QQQ', 'TLT', 'SHY', 'GLD']
weights = [ 0.35,  0.35,  0.1,  0.1,  0.1]  # 각 자산의 비중을 설정하세요. 합이 1이 되어야 합니다.

# 날짜 범위 설정
end_date = datetime.now()
start_date = datetime(2005, 1, 1)

# 데이터 가져오기
data = fetch_data(tickers, start_date, end_date)

# 리밸런싱 주기 선택
rebalance_frequency = 'quarterly'  # 'monthly', 'quarterly', 'yearly' 중 선택

# Calculate portfolio values
portfolio_value = rebalance(data, weights, rebalance_frequency)

# Calculate returns
monthly_returns = calculate_monthly_returns(portfolio_value)
monthly_returns = monthly_returns.dropna()  # Remove NaN values

yearly_returns = calculate_yearly_returns(portfolio_value)
yearly_returns = yearly_returns.dropna()  # Remove NaN values

# Create tables
monthly_table = create_monthly_table(monthly_returns)
yearly_table = create_yearly_table(monthly_returns)  # Use monthly returns to calculate yearly table

# Calculate yearly CAGR and MDD
total_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
yearly_cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / total_years) - 1
all_time_mdd = calculate_mdd(portfolio_value)

# Print results
print(f"Rebalancing Frequency: {rebalance_frequency}")
print(f"Yearly CAGR: {format_percentage(yearly_cagr)}")
print(f"All-time MDD: {format_percentage(all_time_mdd)}")

# Display tables
print("\nMonthly Returns:")
print(monthly_table.map(format_percentage).to_string())

print("\nYearly Returns and MDD:")
print(yearly_table.map(format_percentage).to_string())

# Monthly Returns Heatmap
plot_heatmap_horizontal(monthly_table, 'Monthly Returns Heatmap', text_size=6)

# Yearly Returns and MDD Heatmap
yearly_data = yearly_table.copy()
yearly_data.columns = ['Returns', 'MDD']
plot_heatmap_vertical(yearly_data, 'Yearly Returns and MDD Heatmap', figsize=(8, len(yearly_data) * 0.25), text_size=8)

# Plot portfolio value over time with vertical lines at year changes
plt.figure(figsize=(16, 8))
plt.plot(portfolio_value.index, portfolio_value.values)
plt.title(f'Portfolio Value Over Time ({rebalance_frequency.capitalize()} Rebalancing)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')

# Add vertical lines at year changes
years = portfolio_value.index.year.unique()
for year in years[1:]:
    plt.axvline(x=pd.Timestamp(year=year, month=1, day=1), color='gray', linestyle='--', alpha=0.5)

plt.show()

# New inputs for future investing expectations
initial_value = 150000
annual_savings = 25000  # Annual savings amount
savings_increment = 0.04  # 4% annual increase in savings
inflation_rate = 0.03  # 3% annual inflation rate
projection_years = 20

print("\nProjected 20 Years of Investment Returns (with additional savings and inflation adjustment):")
print(f"Initial Investment: ${initial_value:,.0f}")
print(f"Initial Annual Savings: ${annual_savings:,.0f}")
print(f"Annual Savings Increment: {savings_increment:.1%}")
print(f"Inflation Rate: {inflation_rate:.1%}")
print(f"Projected Annual Return: {yearly_cagr:.2%}")

projected_values = [initial_value]
current_savings = annual_savings

print("\nYear | Future Value | Present Value (Inflation Adjusted)")
print("-" * 60)

for year in range(1, projection_years + 1):
    # Calculate this year's portfolio value
    previous_value = projected_values[-1]
    investment_return = previous_value * (1 + yearly_cagr)
    total_value = investment_return + current_savings
    
    # Calculate present value adjusted for inflation
    present_value = total_value / (1 + inflation_rate) ** year
    
    projected_values.append(total_value)
    
    print(f"{end_date.year + year} | ${total_value:,.0f} | ${present_value:,.0f}")
    
    # Increase savings for next year
    current_savings *= (1 + savings_increment)

# Plot projected values
plt.figure(figsize=(16, 8))
years = range(end_date.year, end_date.year + projection_years + 1)
plt.plot(years, projected_values, marker='o', label='Future Value')

# Calculate and plot present values
present_values = [projected_values[i] / (1 + inflation_rate) ** i for i in range(len(projected_values))]
plt.plot(years, present_values, marker='s', label='Present Value (Inflation Adjusted)')

plt.title('Projected Portfolio Value for Next 20 Years')
plt.xlabel('Year')
plt.ylabel('Projected Portfolio Value')
plt.grid(True)
plt.xticks(years[::2])  # Show every other year on x-axis for clarity
plt.ylim(bottom=0)  # Start y-axis from 0
plt.legend()

# 숫자 레이블 추가
for i, (future_value, present_value) in enumerate(zip(projected_values, present_values)):
    if i % 5 == 0:  # 5년마다 레이블 표시
        plt.annotate(f'${future_value:,.0f}', (years[i], future_value), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'${present_value:,.0f}', (years[i], present_value), textcoords="offset points", xytext=(0,-15), ha='center')

plt.tight_layout()
plt.show()

# Streamlit 페이지 설정
st.set_page_config(page_title="포트폴리오 분석 앱", layout="wide", initial_sidebar_state="expanded")

# 개선된 CSS 스타일
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
    }
    .Widget>label {
        color: #31333F;
        font-weight: 600;
    }
    .stSelectbox>div>div>select {
        background-color: #f0f2f6;
    }
    .stNumberInput>div>div>input {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        width: 400px !important;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = None

def main():
    initialize_session_state()

    if not st.session_state.show_results:
        st.sidebar.title('포트폴리오 분석 앱')

        # 사이드바에 입력 파라미터 추가
        st.sidebar.header('포트폴리오 설정')
        
        st.sidebar.markdown("""
        선택가능한 자산 : S&P500(SPY), NASDAQ100(QQQ), 미국장기채(TLT), 미국중기채(IEF), 
        미국단기채(SHY), 금(GLD), 선진국주식(EFA), 개발도상국주식(EEM)
        """)

        assets = ['SPY', 'QQQ', 'TLT', 'IEF', 'SHY', 'GLD', 'EFA', 'EEM']
        default_assets = ['SPY', 'QQQ', 'TLT', 'SHY', 'GLD']
        
        selected_assets = [st.sidebar.selectbox(f'자산 {i+1}', assets, index=assets.index(default_assets[i])) for i in range(5)]

        weights = []
        st.sidebar.markdown("### 자산 비중 설정 (1-100%)")
        for i, asset in enumerate(selected_assets):
            default_weight = 35 if i < 2 else 10
            weight = st.sidebar.number_input(f'{asset} 비중 (%)', min_value=1, max_value=100, value=default_weight, step=1)
            weights.append(weight / 100)

        weights = [w / sum(weights) for w in weights]  # Normalize weights
        
        min_date = datetime(2005, 1, 1)
        start_date = st.sidebar.date_input('시작 날짜', min_date, min_value=min_date)
        end_date = st.sidebar.date_input('종료 날짜', datetime.now())
        rebalance_frequency = st.sidebar.selectbox('리밸런싱 주기', ['monthly', 'quarterly', 'yearly'])

        # 미래 투자 전망 입력
        st.sidebar.header('미래 투자 전망 설정')
        initial_value = st.sidebar.number_input('초기 투자금 ($)', value=150000)
        annual_savings = st.sidebar.number_input('연간 저축액 ($)', value=25000)
        savings_increment = st.sidebar.number_input('저축 증가율 (%)', value=4.0, format="%.1f") / 100
        inflation_rate = st.sidebar.number_input('인플레이션율 (%)', value=3.0, format="%.1f") / 100
        expected_return = st.sidebar.number_input('예상 연평균 수익률 (%)', value=10.0, format="%.1f") / 100
        projection_years = st.sidebar.number_input('전망 기간 (년)', value=20, min_value=1, max_value=50, step=1)

        if st.sidebar.button('분석 실행'):
            st.session_state.show_results = True
            st.session_state.analysis_params = {
                'tickers': selected_assets, 'weights': weights, 'start_date': start_date, 'end_date': end_date,
                'rebalance_frequency': rebalance_frequency, 'initial_value': initial_value,
                'annual_savings': annual_savings, 'savings_increment': savings_increment,
                'inflation_rate': inflation_rate, 'expected_return': expected_return,
                'projection_years': projection_years
            }
            st.experimental_rerun()

    else:
        params = st.session_state.analysis_params
        
        # 데이터 가져오기
        data = fetch_data(params['tickers'], params['start_date'], params['end_date'])

        # 포트폴리오 가치 계산
        portfolio_value = rebalance(data, params['weights'], params['rebalance_frequency'])

        # 수익률 계산
        monthly_returns = calculate_monthly_returns(portfolio_value)
        yearly_returns = calculate_yearly_returns(portfolio_value)

        # 테이블 생성
        monthly_table = create_monthly_table(monthly_returns)
        yearly_table = create_yearly_table(monthly_returns)

        # CAGR 및 MDD 계산
        total_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
        yearly_cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / total_years) - 1
        all_time_mdd = calculate_mdd(portfolio_value)

        # 결과 표시
        st.title('포트폴리오 분석 결과')
        col1, col2, col3 = st.columns(3)
        col1.metric("리밸런싱 주기", params['rebalance_frequency'])
        col2.metric("연간 CAGR", f"{yearly_cagr:.2%}")
        col3.metric("전체 기간 MDD", f"{all_time_mdd:.2%}")

        # 월별 수익률 히트맵
        st.subheader('월별 수익률 히트맵')
        fig = plot_heatmap_horizontal(monthly_table, 'Monthly Returns Heatmap')
        st.pyplot(fig)

        # 연간 수익률 및 MDD 히트맵
        st.subheader('연간 수익률 및 MDD 히트맵')
        yearly_data = yearly_table.copy()
        yearly_data.columns = ['Returns', 'MDD']
        fig = plot_heatmap_vertical(yearly_data, 'Yearly Returns and MDD Heatmap')
        st.pyplot(fig)

        # 포트폴리오 가치 그래프
        st.subheader('포트폴리오 가치 변화')
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(portfolio_value.index, portfolio_value.values)
        ax.set_title(f'Portfolio Value Over Time ({params["rebalance_frequency"].capitalize()} Rebalancing)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        years = portfolio_value.index.year.unique()
        for year in years[1:]:
            ax.axvline(x=pd.Timestamp(year=year, month=1, day=1), color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # 미래 투자 전망 계산
        st.header('미래 투자 전망')
        projected_values = [params['initial_value']]
        current_savings = params['annual_savings']

        data = []
        for year in range(1, params['projection_years'] + 1):
            previous_value = projected_values[-1]
            investment_return = previous_value * (1 + params['expected_return'])
            total_value = investment_return + current_savings
            present_value = total_value / (1 + params['inflation_rate']) ** year
            projected_values.append(total_value)
            data.append([params['end_date'].year + year, total_value, present_value])
            current_savings *= (1 + params['savings_increment'])

        df = pd.DataFrame(data, columns=['Year', 'Future Value', 'Present Value'])
        
        st.subheader("미래 투자 전망 결과:")
        col1, col2, col3 = st.columns(3)
        col1.metric("초기 투자금", f"${params['initial_value']:,.0f}")
        col2.metric("초기 연간 저축액", f"${params['annual_savings']:,.0f}")
        col3.metric("예상 연간 수익률", f"{params['expected_return']:.1%}")

        col1, col2 = st.columns(2)
        col1.metric("연간 저축액 증가율", f"{params['savings_increment']:.1%}")
        col2.metric("인플레이션율", f"{params['inflation_rate']:.1%}")
        
        st.subheader("미래 가치 전망:")
        st.dataframe(df.style.format({
            'Year': '{:.0f}',
            'Future Value': '${:,.0f}',
            'Present Value': '${:,.0f}'
        }))

        # 미래 전망 그래프
        st.subheader("미래 가치 전망 그래프")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df['Year'], df['Future Value'], marker='o', label='Future Value')
        ax.plot(df['Year'], df['Present Value'], marker='s', label='Present Value (Inflation Adjusted)')
        ax.set_title(f'Projected Portfolio Value for Next {params["projection_years"]} Years')
        ax.set_xlabel('Year')
        ax.set_ylabel('Projected Portfolio Value')
        ax.grid(True)
        ax.set_ylim(bottom=0)
        ax.legend()

        # 숫자 레이블 추가
        for i in range(0, len(df), 5):
            ax.annotate(f'${df["Future Value"].iloc[i]:,.0f}', (df['Year'].iloc[i], df['Future Value'].iloc[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(f'${df["Present Value"].iloc[i]:,.0f}', (df['Year'].iloc[i], df['Present Value'].iloc[i]), 
                        textcoords="offset points", xytext=(0,-15), ha='center')

        plt.tight_layout()
        st.pyplot(fig)

        if st.button('메인으로 돌아가기'):
            st.session_state.show_results = False
            st.session_state.analysis_params = None
            st.experimental_rerun()

if __name__ == '__main__':
    main()