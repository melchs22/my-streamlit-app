import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pydeck as pdk
import requests
import os
import re
from dotenv import load_dotenv
import io
from fpdf import FPDF
import base64

# Load environment variables
load_dotenv()
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')

# Configure page
st.set_page_config(
    page_title="Union App Metrics Dashboard",
    page_icon=r"C:\Users\TUTU\PyCharmMiscProject\your_image.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label, .stMetric div {
        color: black !important;
    }
    .stPlotlyChart, .stPydeckChart {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Define the data file path (backend only)
DATA_FILE_PATH = r"./DIBYA.xlsx"

# Define backend configuration for user base metrics
TOTAL_RIDERS = 2514  # Set your total rider pool size here
TOTAL_PASSENGERS = 154  # Set your total passenger base here


@st.cache_data
def load_data():
    try:
        # Load data from the predefined backend file path
        df = pd.read_excel(DATA_FILE_PATH)

        # Data cleaning
        df['Trip Date'] = pd.to_datetime(df['Trip Date'], errors='coerce')
        df['Trip Hour'] = df['Trip Date'].dt.hour
        df['Day of Week'] = df['Trip Date'].dt.day_name()
        df['Month'] = df['Trip Date'].dt.month_name()

        # Process Trip Pay Amount - extract and sum all UGX values
        if 'Trip Pay Amount' in df.columns:
            def extract_and_sum_amounts(value):
                try:
                    if pd.isna(value):
                        return 0.0
                    amounts = re.findall(r'UGX(\d+)', str(value))
                    return sum(float(amount) for amount in amounts) if amounts else 0.0
                except:
                    return 0.0

            df['Trip Pay Amount Cleaned'] = df['Trip Pay Amount'].apply(extract_and_sum_amounts)
        else:
            st.warning("No 'Trip Pay Amount' column found - creating placeholder")
            df['Trip Pay Amount Cleaned'] = 0.0

        # Process Distance
        df['Distance'] = pd.to_numeric(df['Trip Distance (KM/Mi)'], errors='coerce').fillna(0)

        # Process Company Commission
        if 'Company Amt (UGX)' in df.columns:
            df['Company Commission Cleaned'] = pd.to_numeric(df['Company Amt (UGX)'], errors='coerce').fillna(0)
        else:
            st.warning("No 'Company Amt (UGX)' column found - creating placeholder")
            df['Company Commission Cleaned'] = 0.0

        # Process Pay Mode
        if 'Pay Mode' not in df.columns:
            st.warning("No 'Pay Mode' column found - adding placeholder")
            df['Pay Mode'] = 'Unknown'

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def calculate_cancellation_rate(df):
    if 'Trip Status' not in df.columns:
        return None

    # Define cancelled statuses
    cancelled_statuses = [
        'Cancelled by Rider',
        'Cancelled by Driver at Pickup Location',
        'Cancelled by Driver'
    ]

    # Count cancelled trips
    cancelled_trips = df[df['Trip Status'].isin(cancelled_statuses)].shape[0]

    # Count all other statuses (excluding cancelled)
    other_statuses = df[~df['Trip Status'].isin(cancelled_statuses)].shape[0]

    # Calculate cancellation rate
    if (cancelled_trips + other_statuses) > 0:
        cancellation_rate = (cancelled_trips / (cancelled_trips + other_statuses)) * 100
        return cancellation_rate
    else:
        return 0


def calculate_passenger_search_timeout(df):
    if 'Trip Status' not in df.columns:
        return None

    # Count expired trips
    expired_trips = df[df['Trip Status'] == 'Expired'].shape[0]

    # Count all other statuses (excluding expired)
    other_statuses = df[df['Trip Status'] != 'Expired'].shape[0]

    # Calculate timeout rate
    if (expired_trips + other_statuses) > 0:
        timeout_rate = (expired_trips / (expired_trips + other_statuses)) * 100
        return timeout_rate
    else:
        return 0


def completed_vs_cancelled_daily(df):
    if 'Trip Status' not in df.columns or 'Trip Date' not in df.columns:
        return None

    # Define completed and cancelled statuses
    completed_status = ['Job Completed', 'Partner Assigned']
    cancelled_statuses = [
        'Cancelled by Rider',
        'Cancelled by Driver at Pickup Location',
        'Cancelled by Driver'
    ]
    expired_status = ['Expired']

    # Filter and group data
    df['Date'] = df['Trip Date'].dt.date
    daily_data = df.groupby(['Date', 'Trip Status']).size().unstack(fill_value=0)

    # Sum completed, cancelled, and expired trips
    completed_cols = [col for col in daily_data.columns if any(status in str(col) for status in completed_status)]
    cancelled_cols = [col for col in daily_data.columns if any(status in str(col) for status in cancelled_statuses)]
    expired_cols = [col for col in daily_data.columns if any(status in str(col) for status in expired_status)]

    if len(completed_cols) > 0:
        daily_data['Completed'] = daily_data[completed_cols].sum(axis=1)
    else:
        daily_data['Completed'] = 0

    if len(cancelled_cols) > 0:
        daily_data['Cancelled'] = daily_data[cancelled_cols].sum(axis=1)
    else:
        daily_data['Cancelled'] = 0

    if len(expired_cols) > 0:
        daily_data['Expired'] = daily_data[expired_cols].sum(axis=1)
    else:
        daily_data['Expired'] = 0

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily_data.index,
        y=daily_data['Completed'],
        name='Completed Trips',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=daily_data.index,
        y=daily_data['Cancelled'],
        name='Cancelled Trips',
        marker_color='red'
    ))

    fig.add_trace(go.Bar(
        x=daily_data.index,
        y=daily_data['Expired'],
        name='Expired Trips',
        marker_color='orange'
    ))

    fig.update_layout(
        title='Daily Trip Status Breakdown',
        xaxis_title='Date',
        yaxis_title='Number of Trips',
        barmode='group'
    )

    return fig


def calculate_driver_retention_rate(total_riders, total_passengers, unique_drivers):
    """
    Calculate driver retention rate based on backend inputs
    Formula: (Unique Drivers / Total Riders) * 100
    Also calculates Driver-to-Passenger ratio
    """
    if total_riders > 0:
        retention_rate = (unique_drivers / total_riders) * 100
    else:
        retention_rate = 0

    if unique_drivers > 0:
        passenger_ratio = total_passengers / unique_drivers
    else:
        passenger_ratio = 0

    return retention_rate, passenger_ratio


def total_trips_by_status(df):
    if 'Trip Status' in df.columns:
        status_counts = df['Trip Status'].value_counts().reset_index()
        status_counts.columns = ['Trip Status', 'Count']
        fig = px.bar(
            status_counts,
            x='Trip Status',
            y='Count',
            title='Total Trips by Status',
            labels={'Count': 'Number of Trips'}
        )
        st.plotly_chart(fig, use_container_width=True)


def total_distance_covered(df):
    total_distance = df['Distance'].sum()
    st.metric("Total Distance Covered", f"{total_distance:,.1f} km")


def revenue_by_day(df):
    if 'Trip Date' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        df['Day'] = df['Trip Date'].dt.date
        daily_revenue = df.groupby('Day')['Trip Pay Amount Cleaned'].sum().reset_index()
        fig = px.line(
            daily_revenue,
            x='Day',
            y='Trip Pay Amount Cleaned',
            title='Revenue by Day'
        )
        st.plotly_chart(fig, use_container_width=True)


def avg_revenue_per_trip(df):
    avg_revenue = df['Trip Pay Amount Cleaned'].mean()
    st.metric("Avg. Revenue per Trip", f"{avg_revenue:,.0f} UGX")


def top_pickup_locations(df):
    if 'From Location' in df.columns:
        top_pickups = df['From Location'].value_counts().nlargest(10).reset_index()
        top_pickups.columns = ['Location', 'Trips']
        st.dataframe(top_pickups)
    else:
        st.warning("No pickup location data available")


def total_trips_by_type(df):
    if 'Trip Type' in df.columns:
        trip_type_counts = df['Trip Type'].value_counts().reset_index()
        trip_type_counts.columns = ['Trip Type', 'Count']
        fig = px.pie(
            trip_type_counts,
            names='Trip Type',
            values='Count',
            title='Total Trips by Type'
        )
        st.plotly_chart(fig, use_container_width=True)


def top_drivers_by_revenue(df):
    if 'Driver' in df.columns:
        driver_performance = df.groupby('Driver').agg({
            'Trip Pay Amount Cleaned': 'sum',
            'Distance': 'sum'
        }).nlargest(10, 'Trip Pay Amount Cleaned')
        st.subheader("Top 10 Drivers by Revenue")
        fig = px.bar(
            driver_performance,
            x=driver_performance.index,
            y='Trip Pay Amount Cleaned',
            labels={'Trip Pay Amount Cleaned': 'Revenue (UGX)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No driver data available")


def passenger_insights(df):
    if 'Passenger' in df.columns:
        st.subheader("Top Passengers by Number of Trips")
        # Filter completed trips for passengers (Job Completed status)
        completed_trips = df[df['Trip Status'] == 'Job Completed']
        top_passengers = completed_trips['Passenger'].value_counts().nlargest(10).reset_index()
        top_passengers.columns = ['Passenger', 'Trips']
        st.dataframe(top_passengers)
    else:
        st.warning("No passenger data available")


def peak_hours(df):
    if 'Trip Hour' in df.columns:
        hourly_trips = df['Trip Hour'].value_counts().sort_index()
        fig = px.bar(
            hourly_trips,
            title='Requets by Hour of Day',
            labels={'value': 'Number of Trips', 'index': 'Hour'}
        )
        st.plotly_chart(fig, use_container_width=True)


def trip_status_trends(df):
    if 'Trip Status' in df.columns and 'Trip Date' in df.columns:
        status_trends = df.groupby([df['Trip Date'].dt.date, 'Trip Status']).size().reset_index(name='Count')
        fig = px.line(
            status_trends,
            x='Trip Date',
            y='Count',
            color='Trip Status',
            title='Trip Status Trends Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)


def total_commission(df):
    total_commission = df['Company Commission Cleaned'].sum()
    st.metric("Total Commission", f"{total_commission:,.0f} UGX")


def unique_driver_count(df):
    unique_drivers = df['Driver'].nunique()
    st.metric("Number of Unique Drivers", unique_drivers)


def top_10_drivers_by_earnings(df):
    if 'Driver' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        top_drivers = df.groupby('Driver')['Trip Pay Amount Cleaned'].sum().nlargest(10)
        st.subheader("Top 10 Drivers by Earnings (Total Trip Pay)")
        st.dataframe(top_drivers)
    else:
        st.warning("No driver or trip pay data available")


def most_frequent_locations(df):
    if 'From Location' in df.columns and 'To Location' in df.columns:
        from_location_counts = df['From Location'].value_counts().nlargest(10).reset_index()
        to_location_counts = df['To Location'].value_counts().nlargest(10).reset_index()
        from_location_counts.columns = ['Location', 'Trips']
        to_location_counts.columns = ['Location', 'Trips']

        st.subheader("Most Frequent Pickup Locations")
        st.dataframe(from_location_counts)
        st.subheader("Most Frequent Drop-off Locations")
        st.dataframe(to_location_counts)
    else:
        st.warning("No location data available")


def revenue_share(df):
    total_revenue = df['Trip Pay Amount Cleaned'].sum()
    total_commission = df['Company Commission Cleaned'].sum()
    if total_revenue > 0:
        revenue_share = total_commission / total_revenue
        st.metric("Revenue Share (Company vs Driver)", f"{revenue_share * 100:.2f}%")
    else:
        st.metric("Revenue Share (Company vs Driver)", "N/A")


def total_payout_to_drivers(df):
    total_payout = df['Company Commission Cleaned'].sum()
    st.metric("Total Commission Paid by Drivers", f"{total_payout:,.0f} UGX")


def fare_per_km(df):
    total_fare = df['Trip Pay Amount Cleaned'].sum()
    total_distance = df['Distance'].sum()
    if total_distance > 0:
        fare_per_km = total_fare / total_distance
        st.metric("Fare per Kilometer/Mile", f"{fare_per_km:,.2f} UGX")
    else:
        st.metric("Fare per Kilometer/Mile", "N/A")


def customer_payment_methods(df):
    if 'Pay Mode' in df.columns:
        pay_mode_counts = df['Pay Mode'].value_counts().reset_index()
        pay_mode_counts.columns = ['Payment Method', 'Count']
        fig = px.pie(
            pay_mode_counts,
            names='Payment Method',
            values='Count',
            title='Customer Payment Methods Breakdown'
        )
        st.plotly_chart(fig, use_container_width=True)


def gross_profit(df):
    total_revenue = df['Trip Pay Amount Cleaned'].sum()
    total_commission = df['Company Commission Cleaned'].sum()
    gross_profit = total_revenue - total_commission
    st.metric("Rider Revenue", f"{gross_profit:,.0f} UGX")


def avg_commission_per_trip(df):
    total_commission = df['Company Commission Cleaned'].sum()
    num_trips = len(df)
    if num_trips > 0:
        avg_commission = total_commission / num_trips
        st.metric("Avg. Commission per Trip", f"{avg_commission:,.0f} UGX")
    else:
        st.metric("Avg. Commission per Trip", "N/A")


def revenue_per_driver(df):
    total_revenue = df['Trip Pay Amount Cleaned'].sum()
    unique_drivers = df['Driver'].nunique()
    if unique_drivers > 0:
        revenue_per_driver = total_revenue / unique_drivers
        st.metric("Revenue per Driver", f"{revenue_per_driver:,.0f} UGX")
    else:
        st.metric("Revenue per Driver", "N/A")


def driver_earnings_per_trip(df):
    total_revenue = df['Trip Pay Amount Cleaned'].sum()
    total_commission = df['Company Commission Cleaned'].sum()
    num_trips = len(df)
    if num_trips > 0:
        earnings_per_trip = (total_revenue - total_commission) / num_trips
        st.metric("Driver Earnings per Trip", f"{earnings_per_trip:,.0f} UGX")
    else:
        st.metric("Driver Earnings per Trip", "N/A")


def trips_per_driver(df):
    if 'Driver' in df.columns:
        num_trips = len(df)
        unique_drivers = df['Driver'].nunique()
        if unique_drivers > 0:
            trips_per_driver = num_trips / unique_drivers
            st.metric("Avg. Trips per Driver", f"{trips_per_driver:.1f}")
        else:
            st.metric("Avg. Trips per Driver", "N/A")


def payment_method_revenue(df):
    if 'Pay Mode' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        payment_revenue = df.groupby('Pay Mode')['Trip Pay Amount Cleaned'].sum().reset_index()
        fig = px.pie(
            payment_revenue,
            names='Pay Mode',
            values='Trip Pay Amount Cleaned',
            title='Revenue by Payment Method'
        )
        st.plotly_chart(fig, use_container_width=True)


def distance_vs_revenue_scatter(df):
    if 'Distance' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        fig = px.scatter(
            df,
            x='Distance',
            y='Trip Pay Amount Cleaned',
            title='Distance vs. Revenue',
            trendline='ols'
        )
        st.plotly_chart(fig, use_container_width=True)


def weekday_vs_weekend_analysis(df):
    if 'Trip Date' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        df['Is Weekend'] = df['Trip Date'].dt.dayofweek >= 5  # 5 and 6 are Saturday and Sunday
        weekend_data = df.groupby('Is Weekend').agg({
            'Trip Pay Amount Cleaned': 'sum',
            'Distance': 'sum',
            'Trip Status': 'count'
        }).reset_index()
        weekend_data['Is Weekend'] = weekend_data['Is Weekend'].map({True: 'Weekend', False: 'Weekday'})

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                weekend_data,
                x='Is Weekend',
                y='Trip Pay Amount Cleaned',
                title='Revenue: Weekend vs Weekday'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                weekend_data,
                x='Is Weekend',
                y='Trip Status',
                title='Trips: Weekend vs Weekday'
            )
            st.plotly_chart(fig, use_container_width=True)


def driver_performance_comparison(df):
    if 'Driver' in df.columns and 'Trip Pay Amount Cleaned' in df.columns and 'Distance' in df.columns:
        driver_stats = df.groupby('Driver').agg({
            'Trip Pay Amount Cleaned': ['mean', 'sum'],
            'Distance': ['mean', 'sum'],
            'Trip Status': 'count'
        }).reset_index()
        driver_stats.columns = ['Driver', 'Avg Revenue per Trip', 'Total Revenue',
                                'Avg Distance per Trip', 'Total Distance', 'Number of Trips']

        st.subheader("Driver Performance Comparison")
        st.dataframe(driver_stats.sort_values('Total Revenue', ascending=False))


def passenger_value_segmentation(df):
    if 'Passenger' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        passenger_stats = df.groupby('Passenger').agg({
            'Trip Pay Amount Cleaned': 'sum',
            'Trip Status': 'count'
        }).reset_index()
        passenger_stats.columns = ['Passenger', 'Total Spend', 'Number of Trips']

        # Create segments
        passenger_stats['Segment'] = pd.qcut(passenger_stats['Total Spend'],
                                             q=3,
                                             labels=['Low', 'Medium', 'High'])

        st.subheader("Passenger Value Segmentation")
        fig = px.scatter(
            passenger_stats,
            x='Number of Trips',
            y='Total Spend',
            color='Segment',
            title='Passenger Value Segmentation'
        )
        st.plotly_chart(fig, use_container_width=True)


def get_download_data(df):
    """Prepare the data for download with all cleaned columns"""
    # Create a copy of the dataframe to avoid modifying the original
    download_df = df.copy()

    # Add any additional processing if needed
    return download_df


def create_metrics_pdf(df, date_range, retention_rate, passenger_ratio):
    """Create a PDF report of all dashboard metrics"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Union App Metrics Dashboard Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date Range: {date_range[0]} to {date_range[1]}", ln=1, align='C')
    pdf.ln(10)

    # Overview Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. Overview Metrics", ln=1)
    pdf.set_font("Arial", size=12)

    # Calculate metrics
    total_trips = len(df)
    completed_trips = len(df[df['Trip Status'] == 'Job Completed'])
    avg_distance = df['Distance'].mean() if 'Distance' in df.columns else 0
    cancellation_rate = calculate_cancellation_rate(df) or 0
    timeout_rate = calculate_passenger_search_timeout(df) or 0
    unique_drivers = df['Driver'].nunique() if 'Driver' in df.columns else 0
    trips_per_driver = total_trips / unique_drivers if unique_drivers > 0 else 0

    # Add metrics to PDF
    pdf.cell(200, 10, txt=f"Total Requests: {total_trips}", ln=1)
    pdf.cell(200, 10, txt=f"Completed Trips: {completed_trips}", ln=1)
    pdf.cell(200, 10, txt=f"Average Distance: {avg_distance:.1f} km", ln=1)
    pdf.cell(200, 10, txt=f"Driver Cancellation Rate: {cancellation_rate:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"Passenger Search Timeout: {timeout_rate:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"Driver Retention Rate: {retention_rate:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"Driver-to-Passenger Ratio: {passenger_ratio:.1f}", ln=1)
    pdf.cell(200, 10, txt=f"Average Trips per Driver: {trips_per_driver:.1f}", ln=1)

    # Financial Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Financial Metrics", ln=1)
    pdf.set_font("Arial", size=12)

    total_revenue = df['Trip Pay Amount Cleaned'].sum()
    total_commission = df['Company Commission Cleaned'].sum()
    gross_profit = total_revenue - total_commission
    avg_revenue = df['Trip Pay Amount Cleaned'].mean()
    avg_commission = total_commission / total_trips if total_trips > 0 else 0
    revenue_per_driver = total_revenue / unique_drivers if unique_drivers > 0 else 0
    earnings_per_trip = gross_profit / total_trips if total_trips > 0 else 0
    fare_per_km = total_revenue / df['Distance'].sum() if df['Distance'].sum() > 0 else 0
    revenue_share = (total_commission / total_revenue * 100) if total_revenue > 0 else 0

    pdf.cell(200, 10, txt=f"Total Value of Rides: {total_revenue:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Total Commission: {total_commission:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Rider Revenue: {gross_profit:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Average Revenue per Trip: {avg_revenue:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Average Commission per Trip: {avg_commission:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Revenue per Driver: {revenue_per_driver:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Driver Earnings per Trip: {earnings_per_trip:,.0f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Fare per Kilometer/Mile: {fare_per_km:,.2f} UGX", ln=1)
    pdf.cell(200, 10, txt=f"Revenue Share (Company vs Driver): {revenue_share:.2f}%", ln=1)

    # User Analysis Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="3. User Analysis Metrics", ln=1)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Number of Unique Drivers: {unique_drivers}", ln=1)
    pdf.cell(200, 10, txt=f"Total Rider: {TOTAL_RIDERS}", ln=1)
    pdf.cell(200, 10, txt=f"Total Passengers: {TOTAL_PASSENGERS}", ln=1)

    # Top Drivers
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Top 10 Drivers by Earnings:", ln=1)
    pdf.set_font("Arial", size=10)

    if 'Driver' in df.columns and 'Trip Pay Amount Cleaned' in df.columns:
        top_drivers = df.groupby('Driver')['Trip Pay Amount Cleaned'].sum().nlargest(10)
        for i, (driver, amount) in enumerate(top_drivers.items(), 1):
            pdf.cell(200, 7, txt=f"{i}. {driver}: {amount:,.0f} UGX", ln=1)

    return pdf


def main():
    st.title("Union App Metrics Dashboard")

    # Check if data file exists
    try:
        df = load_data()

        if df.empty:
            st.error("No data loaded - please check the backend data file")
            return

        # Date filter
        if 'Trip Date' not in df.columns:
            st.error("No 'Trip Date' column found in the data")
            return

        min_date = df['Trip Date'].min().date()
        max_date = df['Trip Date'].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            df = df[(df['Trip Date'].dt.date >= date_range[0]) &
                    (df['Trip Date'].dt.date <= date_range[1])]

        # Calculate unique drivers from data
        unique_drivers = df['Driver'].nunique() if 'Driver' in df.columns else 0

        # Calculate driver retention metrics using backend-defined values
        retention_rate, passenger_ratio = calculate_driver_retention_rate(
            TOTAL_RIDERS, TOTAL_PASSENGERS, unique_drivers
        )

        # Add download buttons to sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Data")

        # Create a buffer to hold the Excel file
        output = io.BytesIO()

        # Use ExcelWriter to write the DataFrame to the buffer
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            get_download_data(df).to_excel(writer, sheet_name='Dashboard Data', index=False)

        # Get the Excel data from the buffer
        excel_data = output.getvalue()

        # Create Excel download button
        st.sidebar.download_button(
            label="📊 Download Full Data (Excel)",
            data=excel_data,
            file_name=f"union_app_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Create PDF report
        pdf = create_metrics_pdf(df, date_range, retention_rate, passenger_ratio)
        pdf_output = pdf.output(dest='S').encode('latin1')

        # Create PDF download button
        st.sidebar.download_button(
            label="📄 Download Metrics Report (PDF)",
            data=pdf_output,
            file_name=f"union_app_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Financial", "User Analysis", "Geographic"])

        with tab1:  # Overview Tab
            st.header("Trips Overview")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Requests", len(df))
            with col2:
                completed_trips = len(df[df['Trip Status'] == 'Job Completed'])
                st.metric("Completed Trips", completed_trips)
            with col3:
                st.metric("Avg. Distance", f"{df['Distance'].mean():.1f} km" if 'Distance' in df.columns else "N/A")
            with col4:
                # Calculate and display cancellation rate
                cancellation_rate = calculate_cancellation_rate(df)
                if cancellation_rate is not None:
                    st.metric("Driver Cancellation Rate", f"{cancellation_rate:.1f}%")
                else:
                    st.metric("Driver Cancellation Rate", "N/A")
            with col5:
                # Calculate and display passenger search timeout rate
                timeout_rate = calculate_passenger_search_timeout(df)
                if timeout_rate is not None:
                    st.metric("Passenger Search Timeout", f"{timeout_rate:.1f}%")
                else:
                    st.metric("Passenger Search Timeout", "N/A")

            # Display the completed vs cancelled vs expired trips chart
            status_breakdown_fig = completed_vs_cancelled_daily(df)
            if status_breakdown_fig:
                st.plotly_chart(status_breakdown_fig, use_container_width=True)
            else:
                st.warning("Could not generate trip status breakdown chart - missing required data")

            # New metrics in Overview tab
            col6, col7, col8 = st.columns(3)
            with col6:
                trips_per_driver(df)
            with col7:
                st.metric("Driver Retention Rate", f"{retention_rate:.1f}%",
                          help="Percentage of total riders who are active drivers")
            with col8:
                st.metric("Driver-to-Passenger Ratio", f"{passenger_ratio:.1f}",
                          help="Number of passengers per active driver")

            total_trips_by_status(df)
            total_distance_covered(df)
            revenue_by_day(df)
            avg_revenue_per_trip(df)
            total_commission(df)

        with tab2:  # Financial Tab
            st.header("Financial Performance")

            # Financial metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                total_revenue = df['Trip Pay Amount Cleaned'].sum()
                st.metric("Total Value Of Rides", f"{total_revenue:,.0f} UGX")
            with col2:
                total_commission(df)
            with col3:
                gross_profit(df)

            col4, col5, col6 = st.columns(3)
            with col4:
                avg_commission_per_trip(df)
            with col5:
                revenue_per_driver(df)
            with col6:
                driver_earnings_per_trip(df)

            col7, col8, col9 = st.columns(3)
            with col7:
                fare_per_km(df)
            with col8:
                revenue_share(df)
            with col9:
                st.metric("Total Riders (Pool Size)", TOTAL_RIDERS)

            # Financial visualizations
            total_trips_by_type(df)
            payment_method_revenue(df)
            distance_vs_revenue_scatter(df)
            weekday_vs_weekend_analysis(df)

        with tab3:  # User Analysis Tab
            st.header("User Performance")

            # User metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                unique_driver_count(df)
            with col2:
                st.metric("Total Riders (Pool Size)", TOTAL_RIDERS)
            with col3:
                st.metric("Total Passengers (Base)", TOTAL_PASSENGERS)

            # Display retention metrics
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Driver Retention Rate", f"{retention_rate:.1f}%",
                          help="Percentage of total riders who are active drivers")
            with col5:
                st.metric("Passenger-to-Driver Ratio", f"{passenger_ratio:.1f}",
                          help="Number of passengers per active driver")

            # User visualizations
            top_drivers_by_revenue(df)
            driver_performance_comparison(df)
            passenger_insights(df)
            passenger_value_segmentation(df)
            top_10_drivers_by_earnings(df)

        with tab4:  # Geographic Tab
            st.header("Geographic Analysis")
            most_frequent_locations(df)
            peak_hours(df)
            trip_status_trends(df)
            customer_payment_methods(df)

    except FileNotFoundError:
        st.error("Data file not found. Please ensure the Excel file is placed in the data/ directory.")
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Check if data file exists
    if not os.path.exists(DATA_FILE_PATH):
        st.warning(f"Please place your Excel data file at: {DATA_FILE_PATH}")
    else:
        main()
