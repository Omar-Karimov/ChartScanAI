import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = 'weights/custom_yolov8.pt'

# Logo URL
logo_url = "images/chartscan.png"

# Setting page layout
st.set_page_config(
    page_title="ChartScanAI",  # Setting page title
    page_icon="📊",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Function to download and plot chart
def download_and_plot_chart(ticker, interval="1d", chunk_size=180, figsize=(18, 6.5), dpi=100):
    if interval == "1h":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        period = None
    else:
        start_date = None
        end_date = None
        period = "max"
    
    # Download data for the ticker
    data = yf.download(ticker, interval=interval, start=start_date, end=end_date, period=period)
    
    # Ensure the index is a DatetimeIndex and check if data is not empty
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        # Select only the latest 180 candles
        data = data.iloc[-chunk_size:]

        # Save plot with desired size
        save_path = f"{ticker}_latest_{chunk_size}_candles.png"
        save = dict(fname=save_path, dpi=dpi)
        mpf.plot(data, type="candle", style="yahoo",
                 title=f"{ticker} Latest {chunk_size} Candles",
                 axisoff=True,
                 ylabel="",
                 ylabel_lower="",
                 volume=False,
                 figsize=figsize,
                 savefig=save)
        return save_path
    else:
        st.error("No data found for the specified ticker and interval.")
        return None

# Creating sidebar
with st.sidebar:
    # Add a logo to the top of the sidebar
    st.image(logo_url, use_column_width="auto")
    st.write("")
    st.header("Configurations")     # Adding header to sidebar
    # Section to download and plot chart
    st.subheader("Download and Plot Chart")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"])
    if st.button("Download and Plot Chart"):
        if ticker:
            chart_path = download_and_plot_chart(ticker, interval=interval)
            if chart_path:
                st.success(f"Chart downloaded. Right-click to save: [Download {ticker} Chart]({chart_path})")
                st.image(chart_path, caption=f"{ticker} Chart", use_column_width=True)
        else:
            st.error("Please enter a valid ticker symbol.")
    st.write("")
    st.subheader("Upload Image for Detection")
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 30)) / 100

# Creating main page heading
st.title("ChartScanAI")
st.caption('📈 To use the app, you have two options:')
st.caption('1. **Upload Your Own Image:** Upload a candlestick chart image from the web using the sidebar.')
st.caption('2. **Download and Plot Chart:** Use the "Download and Plot Chart" section by providing the ticker symbol and interval to generate and download a candlestick chart. Then, upload the downloaded image using the sidebar.')
st.caption('After uploading the image, click the :blue[Detect Objects] button to analyze the chart.')


# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

# Load the model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Perform object detection if the button is clicked
if st.sidebar.button('Detect Objects'):
    if source_img:
        # Re-open the image to reset the file pointer
        source_img.seek(0)
        uploaded_image = Image.open(source_img)
        
        # Perform object detection
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                     caption='Detected Image',
                     use_column_width=True
                     )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("Error displaying detection results.")
    else:
        st.error("Please upload an image first.")
