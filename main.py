import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import pandas as pd

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Sales Forcasting Predection")

categoty = ("AAPL", "GOOG")


# selected_category = st.selectbox("select category for predecttion", categoty)
#
# n_years = st.slider("Yeaars of predection :", 1, 4)
# period = n_years * 365


@st.cache
def load_data():
    data = pd.read_csv("E:\Book7.csv")
    data.reset_index(inplace=True)
    print(data)
    return data


data_load_state = st.text("Load data...")
data = load_data()
data_load_state.text("Loading data...data!")

st.subheader("Row data")
st.write(data.tail())


def plot_row_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["DocDate"], y=data["SaleValue"], name='stock_open'))
    fig.add_trace(go.Scatter(x=data["DocDate"], y=data["RTNValue"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_row_data()

# forecasting
df_train = data[['DocDate', 'SaleValue']]
df_train = df_train.rename(columns={"DocDate": "ds", "SaleValue": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=10)
forecast = m.predict(future)

st.subheader("Row Data")
st.write(forecast.tail())

st.write("Sales Forecasting Chart")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Sales Forecasting Component")
fig2 = m.plot_components(forecast)
st.write(fig2)

# return forecasting
df_train2 = data[['DocDate', 'RTNValue']]
df_train2 = df_train.rename(columns={"DocDate": "ds", "RTNValue": "y"})

m1 = Prophet()
m1.fit(df_train2)
future1 = m1.make_future_dataframe(periods=10)
forecast1 = m1.predict(future1)

st.subheader("Row Data")
st.write(forecast1.tail())

st.write("Return Forecasting Chart")
fig3 = plot_plotly(m1, forecast1)
st.plotly_chart(fig3)

st.write("Return Forecasting Component")
fig4 = m1.plot_components(forecast1)
st.write(fig4)
