import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all routes.
# This is crucial for allowing your frontend (running on a different origin/port)
# to make requests to this Flask backend.
CORS(app)

# Configure Flask-Caching for API endpoint responses.
# 'CACHE_TYPE': 'simple' uses a basic in-memory cache.
# Caching improves performance by storing the results of expensive database queries
# and serving them directly from memory for subsequent identical requests,
# reducing the load on the database.
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load database configuration from environment variables.
# This makes the application more flexible and secure, as sensitive
# information like database credentials are not hardcoded directly in the script.
# Default values are provided for local development convenience.
app.config.update({
    'DATABASE_HOST': os.getenv('DATABASE_HOST', 'localhost'),
    'DATABASE_USER': os.getenv('DATABASE_USER', 'root'),
    'DATABASE_PASSWORD': os.getenv('DATABASE_PASSWORD', ''),
    'DATABASE_NAME': os.getenv('DATABASE_NAME', 'mattel'),
    'CACHE_TIMEOUT': int(os.getenv('CACHE_TIMEOUT', '300'))  # Cache timeout in seconds (e.g., 300s = 5 minutes)
})

# Store database configuration in a dictionary for easy access
DATABASE_CONFIG = {
    'host': app.config['DATABASE_HOST'],
    'user': app.config['DATABASE_USER'],
    'password': app.config['DATABASE_PASSWORD'],
    'database': app.config['DATABASE_NAME']
}

# Retrieve cache timeout from app config
cache_timeout = app.config['CACHE_TIMEOUT']


# --- Database Helper Functions ---

def get_db_connection():
    """
    Establishes a database connection to the MySQL server using credentials
    from the DATABASE_CONFIG.
    Handles potential connection errors gracefully.
    """
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error connecting to database: {e}")
        return None


def query_to_dataframe(query, params=None):
    """
    Executes a SQL query and returns the results as a Pandas DataFrame.
    This function is a wrapper to simplify database interactions.
    It ensures the database connection is properly closed after the query.

    Args:
        query (str): The SQL query string to execute.
        params (tuple, optional): Parameters to pass to the query for sanitization.
                                  Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the query results, or an empty
                          DataFrame if an error occurs or no data is found.
    """
    conn = get_db_connection()
    if conn:
        try:
            # pd.read_sql is a convenient way to execute SQL and get a DataFrame.
            # Using the 'params' argument helps prevent SQL injection vulnerabilities.
            df = pd.read_sql(query, conn, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            return pd.DataFrame()  # Return empty DataFrame on query execution error
        finally:
            if conn.is_connected():
                conn.close()  # Always close the connection
    return pd.DataFrame()  # Return empty DataFrame if connection fails


# --- Plotly Helper Functions ---

def create_figure_json(fig):
    """
    Converts a Plotly figure object into a JSON string.
    This JSON string can then be sent to the frontend and parsed by Plotly.js
    to render the chart.
    """
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_time_series_chart(df, x_col, y_cols, title, labels=None):
    """
    A generic helper function to create time series charts using Plotly.
    It supports plotting multiple Y-axis columns and an optional secondary Y-axis
    if one of the Y-columns contains 'amount' in its name (for common usage/revenue metrics).

    Args:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        x_col (str): The name of the column to use for the X-axis (time).
        y_cols (list of str): A list of column names to plot on the Y-axis.
        title (str): The title of the chart.
        labels (dict, optional): A dictionary mapping column names to display labels.
                                 Defaults to None.

    Returns:
        str: A JSON string representation of the Plotly figure.
    """
    # Create a subplot figure, potentially with a secondary Y-axis
    fig = make_subplots(specs=[[{"secondary_y": any('amount' in col.lower() for col in y_cols)}]])

    for i, y_col in enumerate(y_cols):
        # Add each Y-column as a scatter trace (line chart)
        fig.add_trace(
            go.Scatter(x=df[x_col], y=df[y_col], name=labels[y_col] if labels and y_col in labels else y_col),
            # Assign to secondary Y-axis if 'amount' is in the column name and it's not the first trace
            secondary_y=('amount' in y_col.lower()) if i > 0 else False,
        )

    # Update layout with title and X-axis label
    fig.update_layout(title_text=title, xaxis_title=labels.get(x_col, x_col) if labels else x_col)
    # Update primary Y-axis label
    fig.update_yaxes(title_text=labels.get(y_cols[0], y_cols[0]) if labels else y_cols[0], secondary_y=False)

    # If there's more than one Y-column, update the secondary Y-axis label if applicable
    if len(y_cols) > 1:
        if 'amount' in y_cols[1].lower():
            fig.update_yaxes(title_text=labels.get(y_cols[1], y_cols[1]) if labels else y_cols[1], secondary_y=True)

    return create_figure_json(fig)


# --- Flask Routes (API Endpoints) ---

@app.route('/')
def index():
    """
    Renders the main dashboard HTML page (`index.html`).
    This is the entry point for the web application.
    """
    return render_template('index.html',
                           page_title="Mattel Analytics Dashboard",
                           current_year=datetime.now().year)


@app.route('/api/health')
def health_check():
    """
    API endpoint to check the health of the Flask application and its
    connection to the MySQL database.
    Returns a JSON response indicating 'healthy' or 'unhealthy' status.
    """
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return jsonify({'status': 'healthy', 'database': 'connected'})
        return jsonify({'status': 'unhealthy', 'database': 'disconnected'}), 500
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/offers_summary')
@cache.cached(timeout=cache_timeout)
def offers_summary():
    """
    API endpoint to provide a summary of offers and their usage.
    Calculates subscriber count, event count, average, and total deduction amounts per offer.
    Generates a bar chart showing the number of subscribers by offer.
    """
    query = """
    SELECT 
        o.offer_id, 
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as subscriber_count,
        COUNT(ce.cdr_event_id) as event_count,
        IFNULL(AVG(ce.deduct_amount), 0) as avg_deduct_amount,
        IFNULL(SUM(ce.deduct_amount), 0) as total_deduct_amount
    FROM offers o
    LEFT JOIN cdr_events ce ON o.offer_id = ce.offer_id
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    # If no data or no subscribers, return an empty figure with a message
    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No offer summary data available.")
    else:
        # Create a bar chart using Plotly Express
        fig = px.bar(
            df,
            x='offer_name',
            y='subscriber_count',
            title='Number of Subscribers by Offer',
            labels={'offer_name': 'Offer Name', 'subscriber_count': 'Number of Subscribers'},
            hover_data=['event_count', 'avg_deduct_amount', 'total_deduct_amount']
        )

    # Return data and chart JSON
    return jsonify({
        'data': df.to_dict('records'),  # Raw data (can be used for tables on a separate page)
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_type_distribution')
@cache.cached(timeout=cache_timeout)
def customer_type_distribution():
    """
    API endpoint for customer type distribution statistics.
    Calculates the count of customers and associated objects per customer type.
    Generates a pie chart to visualize the distribution.
    """
    query = """
    SELECT 
        cust_type,
        COUNT(*) as customer_count,
        COUNT(DISTINCT o.object_id_bigint) as object_count
    FROM customers c
    LEFT JOIN objects o ON c.cust_num_bigint = o.cust_num_bigint
    GROUP BY cust_type
    """
    df = query_to_dataframe(query)

    # If no data or no customers, return an empty figure with a message
    if df.empty or df['customer_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No customer type distribution data available.")
    else:
        # Create a pie chart
        fig = px.pie(
            df,
            values='customer_count',
            names='cust_type',
            title='Customer Type Distribution',
            hover_data=['object_count']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/operations_breakdown')
@cache.cached(timeout=cache_timeout)
def operations_breakdown():
    """
    API endpoint for operations type breakdown.
    Calculates event count, customer count, average, and total deduction amounts
    for each operation type.
    Generates a bar chart showing events by operation type.
    """
    query = """
    SELECT 
        ot.operation_type_id,
        ot.description,
        COUNT(ce.cdr_event_id) as event_count,
        COUNT(DISTINCT ce.cust_num) as customer_count,
        IFNULL(AVG(ce.deduct_amount), 0) as avg_deduct_amount,
        IFNULL(SUM(ce.deduct_amount), 0) as total_deduct_amount
    FROM operation_types ot
    LEFT JOIN cdr_events ce ON ot.operation_type_id = ce.operation_type_id
    GROUP BY ot.operation_type_id, ot.description
    ORDER BY event_count DESC
    """
    df = query_to_dataframe(query)

    # If no data or no events, return an empty figure with a message
    if df.empty or df['event_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No operations breakdown data available.")
    else:
        # Create a bar chart
        fig = px.bar(
            df,
            x='description',
            y='event_count',
            title='Events by Operation Type',
            labels={'description': 'Operation Type', 'event_count': 'Number of Events'},
            hover_data=['customer_count', 'avg_deduct_amount', 'total_deduct_amount']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/object_types_analysis')
@cache.cached(timeout=cache_timeout)
def object_types_analysis():
    """
    API endpoint for analysis of object types.
    Calculates the count of objects and customers for each object type.
    Generates a pie chart to visualize the distribution.
    """
    query = """
    SELECT 
        ot.object_type_code,
        ot.description,
        COUNT(DISTINCT o.object_id_bigint) as object_count,
        COUNT(DISTINCT o.cust_num_bigint) as customer_count
    FROM object_types ot
    LEFT JOIN objects o ON ot.object_type_id = o.object_type_id
    GROUP BY ot.object_type_code, ot.description
    """
    df = query_to_dataframe(query)

    # If no data or no objects, return an empty figure with a message
    if df.empty or df['object_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No object type analysis data available.")
    else:
        # Create a pie chart
        fig = px.pie(
            df,
            values='object_count',
            names='description',
            title='Object Type Distribution',
            hover_data=['customer_count']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/time_series_analysis')
@cache.cached(timeout=cache_timeout)
def time_series_analysis():
    """
    API endpoint for time series analysis of events.
    Aggregates daily event counts, total deduction amounts, and active customers
    over the last 5 years.
    Generates a time series chart with two Y-axes for event count and active customers.
    """
    query = """
    SELECT 
        DATE(creation_time) as event_date,
        COUNT(*) as event_count,
        IFNULL(SUM(deduct_amount), 0) as total_deduct_amount,
        COUNT(DISTINCT cust_num) as active_customers
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 5 YEAR) 
    GROUP BY DATE(creation_time)
    ORDER BY event_date
    """
    df = query_to_dataframe(query)

    # If no data, return an empty figure with a message
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No daily events and active customers data available.")
    else:
        # Create a subplot with two Y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add trace for Event Count
        fig.add_trace(
            go.Scatter(x=df['event_date'], y=df['event_count'], name="Event Count"),
            secondary_y=False,  # Primary Y-axis
        )
        # Add trace for Active Customers
        fig.add_trace(
            go.Scatter(x=df['event_date'], y=df['active_customers'], name="Active Customers"),
            secondary_y=True,  # Secondary Y-axis
        )
        # Update layout with titles and axis labels
        fig.update_layout(
            title_text="Daily Events and Active Customers Over Time",
            xaxis_title="Date",
        )
        fig.update_yaxes(title_text="Event Count", secondary_y=False)
        fig.update_yaxes(title_text="Active Customers", secondary_y=True)

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/offer_performance')
@cache.cached(timeout=cache_timeout)
def offer_performance():
    """
    API endpoint for analyzing offer performance.
    Calculates subscriber count, event count, total usage (deduct_amount),
    and average offer lifetime in days for each offer.
    Generates a scatter plot to visualize offer performance, with marker size
    representing event count.
    """
    query = """
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as subscriber_count,
        COUNT(ce.cdr_event_id) as event_count,
        IFNULL(SUM(ce.deduct_amount), 0) as total_usage,
        IFNULL(AVG(DATEDIFF(ce.creation_time, 
            (SELECT MIN(creation_time) FROM cdr_events 
             WHERE cust_num = ce.cust_num AND offer_id = ce.offer_id))), 0) as avg_offer_lifetime_days
    FROM offers o
    LEFT JOIN cdr_events ce ON o.offer_id = ce.offer_id 
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    # If no data or all key metrics are zero, return an empty figure with a message
    if df.empty or (
            df['subscriber_count'].sum() == 0 and df['total_usage'].sum() == 0 and df['event_count'].sum() == 0):
        fig = go.Figure()
        fig.update_layout(title_text="No offer performance data available.")
    else:
        # Special handling for cases where all usage/subscriber data is zero,
        # which would make the scatter plot uninformative.
        if df['subscriber_count'].sum() == 0 and df['total_usage'].sum() == 0:
            fig = go.Figure()
            fig.update_layout(title_text="Offer Performance: No meaningful usage or subscriber data.")
        else:
            # Create a scatter plot
            fig = px.scatter(
                df,
                x='subscriber_count',
                y='total_usage',
                # Ensure marker size is at least 1 for visibility, even if event_count is 0
                size=df['event_count'].apply(lambda x: max(x, 1)),
                hover_name='offer_name',
                title='Offer Performance Analysis',
                labels={
                    'subscriber_count': 'Number of Subscribers',
                    'total_usage': 'Total Usage (Amount)',
                    'event_count': 'Number of Events'
                }
            )
            # Add text labels for offers if there are not too many to avoid clutter
            if len(df) < 20:
                fig.update_traces(text=df['offer_name'], textposition='top center')

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/loan_analysis')
@cache.cached(timeout=cache_timeout)
def loan_analysis():
    """
    API endpoint specifically for analyzing loan-related offers.
    Filters offers by name (LIKE 'Loan%') and calculates subscriber count,
    total/average loan amounts, and average loans per customer.
    Generates a horizontal bar chart for loan offer popularity.
    """
    query = """
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as subscriber_count,
        IFNULL(SUM(ce.deduct_amount), 0) as total_loan_amount,
        IFNULL(AVG(ce.deduct_amount), 0) as avg_loan_amount,
        IFNULL(COUNT(ce.cdr_event_id) / NULLIF(COUNT(DISTINCT ce.cust_num), 0), 0) as avg_loans_per_customer
    FROM offers o
    LEFT JOIN cdr_events ce ON o.offer_id = ce.offer_id
    WHERE o.offer_name LIKE 'Loan%'
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    # If no data or no subscribers, return an empty figure with a message
    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No loan offer popularity data available.")
    else:
        # Create a horizontal bar chart
        fig = px.bar(
            df,
            y='offer_name',
            x='subscriber_count',
            color='avg_loan_amount',  # Color bars by average loan amount
            orientation='h',
            title='Loan Offer Popularity',
            labels={
                'offer_name': 'Loan Offer',
                'subscriber_count': 'Number of Subscribers',
                'avg_loan_amount': 'Average Loan Amount'
            },
            hover_data=['total_loan_amount', 'avg_loans_per_customer']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/churn_analysis')
@cache.cached(timeout=cache_timeout)
def churn_analysis():
    """
    API endpoint for churn analysis.
    Identifies churned customers (inactive for > 90 days) and aggregates
    churn count by quarter. Also provides a breakdown of churned customers by offer.
    Generates a bar chart for customer churn by quarter.
    """
    # Query to get churn count by quarter
    query = """
    WITH customer_activity AS (
        SELECT 
            c.cust_num_bigint as cust_num,
            MAX(ce.creation_time) as last_activity_date,
            DATEDIFF(NOW(), MAX(ce.creation_time)) as days_since_last_activity
        FROM customers c
        LEFT JOIN cdr_events ce ON c.cust_num_bigint = ce.cust_num
        GROUP BY c.cust_num_bigint
    ),
    churned AS (
        SELECT 
            ca.cust_num,
            ca.last_activity_date,
            QUARTER(ca.last_activity_date) as churn_quarter,
            YEAR(ca.last_activity_date) as churn_year
        FROM customer_activity ca
        WHERE ca.days_since_last_activity > 90 -- Define churn as inactive for more than 90 days
    )
    SELECT 
        CONCAT(churn_year, '-Q', churn_quarter) as period,
        COUNT(*) as churn_count
    FROM churned
    GROUP BY churn_year, churn_quarter
    ORDER BY churn_year, churn_quarter
    """
    df = query_to_dataframe(query)

    # If no data or no churned customers, return an empty figure with a message
    if df.empty or df['churn_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No customer churn data available.")
    else:
        # Create a bar chart for churn by quarter
        fig = px.bar(
            df,
            x='period',
            y='churn_count',
            title='Customer Churn by Quarter',
            labels={'period': 'Time Period', 'churn_count': 'Number of Churned Customers'}
        )

    # Query to get offers associated with churned customers
    offers_churn_query = """
    WITH customer_last_offer AS (
        SELECT 
            ce.cust_num,
            ce.offer_id,
            MAX(ce.creation_time) as last_activity_date,
            DATEDIFF(NOW(), MAX(ce.creation_time)) as days_since_last_activity
        FROM cdr_events ce
        GROUP BY ce.cust_num, ce.offer_id
    ),
    churned AS (
        SELECT 
            clo.cust_num,
            clo.offer_id
        FROM customer_last_offer clo
        WHERE clo.days_since_last_activity > 90
    )
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ch.cust_num) as churned_customers
    FROM churned ch
    JOIN offers o ON ch.offer_id = o.offer_id
    GROUP BY o.offer_id, o.offer_name
    ORDER BY churned_customers DESC
    """
    offers_churn_df = query_to_dataframe(offers_churn_query)

    return jsonify({
        'data': df.to_dict('records'),  # Churn by quarter data
        'chart': create_figure_json(fig),
        'offers_churn': offers_churn_df.to_dict('records')  # Offers associated with churn
    })


@app.route('/api/revenue_analysis')
@cache.cached(timeout=cache_timeout)
def revenue_analysis():
    """
    API endpoint for revenue analysis.
    Calculates monthly revenue broken down by offer, focusing on usage operation types.
    Generates a stacked bar chart for monthly revenue by offer.
    """
    query = """
    SELECT 
        DATE_FORMAT(ce.creation_time, '%Y-%m') as month,
        o.offer_name,
        IFNULL(SUM(ce.deduct_amount), 0) as revenue
    FROM cdr_events ce
    LEFT JOIN offers o ON ce.offer_id = o.offer_id 
    WHERE ce.operation_type_id IN (1, 2, 3)  -- Assuming these are the operation types that generate revenue
    GROUP BY month, o.offer_name
    ORDER BY month, o.offer_name
    """
    df = query_to_dataframe(query)

    # If no data or no revenue, return an empty figure with a message
    if df.empty or df['revenue'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No revenue analysis data available.")
    else:
        # Create a stacked bar chart
        fig = px.bar(
            df,
            x='month',
            y='revenue',
            color='offer_name',  # Stack bars by offer name
            title='Monthly Revenue by Offer',
            labels={'month': 'Month', 'revenue': 'Revenue', 'offer_name': 'Offer Name'}
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/data_bundle_analysis')
@cache.cached(timeout=cache_timeout)
def data_bundle_analysis():
    """
    API endpoint specific to data bundle analysis.
    Filters offers by names containing 'Data', 'Mo', or 'Internet'.
    Calculates subscriber count, average usage (deduct_amount), and average duration in hours.
    Generates a bar chart for data bundle popularity and usage.
    """
    query = """
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as subscriber_count,
        IFNULL(AVG(ce.deduct_amount), 0) as avg_usage,
        IFNULL(AVG(TIMESTAMPDIFF(HOUR, ce.creation_time, ce.charge_expire_time)), 0) as avg_duration_hours
    FROM offers o
    LEFT JOIN cdr_events ce ON o.offer_id = ce.offer_id
    WHERE (o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Mo%' OR o.offer_name LIKE '%Internet%')
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    # If no data or no subscribers, return an empty figure with a message
    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No data bundle analysis data available.")
    else:
        # Create a bar chart
        fig = px.bar(
            df,
            x='offer_name',
            y='subscriber_count',
            color='avg_usage',  # Color bars by average usage
            title='Data Bundle Popularity and Usage',
            labels={
                'offer_name': 'Data Bundle',
                'subscriber_count': 'Number of Subscribers',
                'avg_usage': 'Average Usage'
            },
            hover_data=['avg_duration_hours']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/expiry_forecast')
@cache.cached(timeout=cache_timeout)
def expiry_forecast():
    """
    API endpoint for forecasting resource expirations.
    Aggregates expiring objects and affected customers by expiry date and object type
    for objects expiring within the next year.
    Generates an area chart to visualize the forecast.
    """
    query = """
    SELECT 
        DATE(o.expiry_date) as expiry_date,
        ot.description as object_type,
        COUNT(o.object_id_bigint) as expiring_objects,
        COUNT(DISTINCT o.cust_num_bigint) as affected_customers
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    WHERE o.expiry_date IS NOT NULL AND o.expiry_date <= DATE_ADD(NOW(), INTERVAL 1 YEAR) 
    GROUP BY expiry_date, object_type
    ORDER BY expiry_date, object_type
    """
    df = query_to_dataframe(query)

    # If no data or no expiring objects, return an empty figure with a message
    if df.empty or df['expiring_objects'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No upcoming object expirations data available.")
    else:
        # Create an area chart
        fig = px.area(
            df,
            x='expiry_date',
            y='expiring_objects',
            color='object_type',  # Stack areas by object type
            title='Upcoming Object Expirations Forecast (Next 1 Year)',
            labels={
                'expiry_date': 'Expiration Date',
                'expiring_objects': 'Number of Expiring Objects',
                'object_type': 'Object Type'
            },
            hover_data=['affected_customers']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_details/<int:cust_num>')
@cache.cached(timeout=cache_timeout)
def customer_details(cust_num):
    """
    API endpoint for detailed information about a specific customer.
    Fetches customer's basic info, associated objects/balances, and recent CDR events.
    This is a lookup endpoint, returning raw data, not a chart.
    """
    # Query for customer basic info
    customer_query = """
    SELECT cust_num_bigint, cust_type FROM customers WHERE cust_num_bigint = %s
    """
    customer_df = query_to_dataframe(customer_query, params=(cust_num,))
    if customer_df.empty:
        return jsonify({'error': 'Customer not found'}), 404

    # Query for customer's objects/balances
    objects_query = """
    SELECT 
        o.object_id_bigint, 
        ot.description as object_type_description, 
        o.expiry_date
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    WHERE o.cust_num_bigint = %s
    ORDER BY o.expiry_date DESC
    """
    objects_df = query_to_dataframe(objects_query, params=(cust_num,))

    # Query for customer's recent CDR events (limited to 100 for performance)
    cdr_events_query = """
    SELECT 
        ce.cdr_event_id, 
        ce.creation_time, 
        op.description as operation_type, 
        of.offer_name,
        ce.deduct_amount, 
        ce.charge_amount,
        ce.current_amount_pre, 
        ce.current_amount_post
    FROM cdr_events ce
    JOIN operation_types op ON ce.operation_type_id = op.operation_type_id
    LEFT JOIN offers of ON ce.offer_id = of.offer_id
    WHERE ce.cust_num = %s
    ORDER BY ce.creation_time DESC
    LIMIT 100
    """
    cdr_events_df = query_to_dataframe(cdr_events_query, params=(cust_num,))

    return jsonify({
        'customer_info': customer_df.iloc[0].to_dict(),
        'objects': objects_df.to_dict('records'),
        'cdr_events': cdr_events_df.to_dict('records')
    })


@app.route('/api/object_details/<int:object_id>')
@cache.cached(timeout=cache_timeout)
def object_details(object_id):
    """
    API endpoint for detailed information about a specific object.
    Fetches object's basic info and related CDR events.
    This is a lookup endpoint, returning raw data, not a chart.
    """
    # Query for object basic info
    object_query = """
    SELECT 
        o.object_id_bigint, 
        o.cust_num_bigint, 
        ot.description as object_type_description, 
        o.expiry_date
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    WHERE o.object_id_bigint = %s
    """
    object_df = query_to_dataframe(object_query, params=(object_id,))
    if object_df.empty:
        return jsonify({'error': 'Object not found'}), 404

    # Query for CDR events related to this object (limited to 100 for performance)
    cdr_events_query = """
    SELECT 
        ce.cdr_event_id, 
        ce.creation_time, 
        op.description as operation_type, 
        of.offer_name,
        ce.deduct_amount, 
        ce.charge_amount,
        ce.current_amount_pre, 
        ce.current_amount_post
    FROM cdr_events ce
    JOIN operation_types op ON ce.operation_type_id = op.operation_type_id
    LEFT JOIN offers of ON ce.offer_id = of.offer_id
    WHERE ce.object_id = %s
    ORDER BY ce.creation_time DESC
    LIMIT 100
    """
    cdr_events_df = query_to_dataframe(cdr_events_query, params=(object_id,))

    return jsonify({
        'object_info': object_df.iloc[0].to_dict(),
        'cdr_events': cdr_events_df.to_dict('records')
    })


@app.route('/api/overall_stats')
@cache.cached(timeout=cache_timeout)
def overall_stats():
    """
    API endpoint for overall key statistics and conclusions about the entire system.
    This endpoint gathers various high-level metrics and generates textual insights.
    """
    stats = {}

    # Total Customers: Count of all unique subscribers
    total_customers_query = "SELECT COUNT(*) as total_customers FROM customers"
    df_total_customers = query_to_dataframe(total_customers_query)
    stats['total_customers'] = int(df_total_customers['total_customers'].iloc[0]) if not df_total_customers.empty else 0

    # Total CDR Events: Total number of records in the CDR events table
    total_cdr_events_query = "SELECT COUNT(*) as total_events FROM cdr_events"
    df_total_events = query_to_dataframe(total_cdr_events_query)
    stats['total_cdr_events'] = int(df_total_events['total_events'].iloc[0]) if not df_total_events.empty else 0

    # Total Revenue: Sum of 'deduct_amount' for operations classified as usage (1, 2, 3)
    total_revenue_query = """
    SELECT IFNULL(SUM(deduct_amount), 0) as total_revenue
    FROM cdr_events
    WHERE operation_type_id IN (1, 2, 3)
    """
    df_total_revenue = query_to_dataframe(total_revenue_query)
    stats['total_revenue'] = float(df_total_revenue['total_revenue'].iloc[0]) if not df_total_revenue.empty else 0.0

    # Active Customers (last year): Count of unique customers with activity in the past 12 months
    active_customers_year_query = """
    SELECT COUNT(DISTINCT cust_num) as active_customers_year
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
    """
    df_active_customers_year = query_to_dataframe(active_customers_year_query)
    stats['active_customers_year'] = int(
        df_active_customers_year['active_customers_year'].iloc[0]) if not df_active_customers_year.empty else 0

    # Total Number of Offers: Count of all distinct offers defined in the 'offers' table
    total_offers_query = "SELECT COUNT(*) as total_offers FROM offers"
    df_total_offers = query_to_dataframe(total_offers_query)
    stats['total_offers'] = int(df_total_offers['total_offers'].iloc[0]) if not df_total_offers.empty else 0

    # Average Revenue Per Customer: Total revenue divided by total customers
    stats['avg_revenue_per_customer'] = stats['total_revenue'] / stats['total_customers'] if stats[
                                                                                                 'total_customers'] > 0 else 0.0

    # Average Events Per Customer: Total CDR events divided by total customers
    stats['avg_events_per_customer'] = stats['total_cdr_events'] / stats['total_customers'] if stats[
                                                                                                   'total_customers'] > 0 else 0.0

    # Unique Offers with Activity: Count of distinct offers that have appeared in CDR events
    unique_active_offers_query = "SELECT COUNT(DISTINCT offer_id) as unique_active_offers FROM cdr_events WHERE offer_id IS NOT NULL"
    df_unique_active_offers = query_to_dataframe(unique_active_offers_query)
    stats['unique_active_offers'] = int(
        df_unique_active_offers['unique_active_offers'].iloc[0]) if not df_unique_active_offers.empty else 0

    # Unique Operation Types Used: Count of distinct operation types recorded in CDR events
    unique_operation_types_query = "SELECT COUNT(DISTINCT operation_type_id) as unique_operation_types FROM cdr_events WHERE operation_type_id IS NOT NULL"
    df_unique_operation_types = query_to_dataframe(unique_operation_types_query)
    stats['unique_operation_types'] = int(
        df_unique_operation_types['unique_operation_types'].iloc[0]) if not df_unique_operation_types.empty else 0

    # Unique Object Types Used: Count of distinct object types referenced in the 'objects' table
    unique_object_types_query = """
    SELECT COUNT(DISTINCT ot.object_type_id) as unique_object_types_used
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    """
    df_unique_object_types = query_to_dataframe(unique_object_types_query)
    stats['unique_object_types_used'] = int(
        df_unique_object_types['unique_object_types_used'].iloc[0]) if not df_unique_object_types.empty else 0

    # Most Popular Offer (by subscriber count): Identifies the offer with the highest number of unique subscribers
    top_offer_query = """
    SELECT o.offer_name, COUNT(DISTINCT ce.cust_num) as subscriber_count
    FROM offers o
    JOIN cdr_events ce ON o.offer_id = ce.offer_id
    GROUP BY o.offer_name
    ORDER BY subscriber_count DESC
    LIMIT 1
    """
    df_top_offer = query_to_dataframe(top_offer_query)
    stats['most_popular_offer'] = df_top_offer.iloc[0].to_dict() if not df_top_offer.empty else {'offer_name': 'N/A',
                                                                                                 'subscriber_count': 0}

    # Most Popular Operation Type (by event count): Identifies the operation type with the highest event count
    top_operation_query = """
    SELECT ot.description, COUNT(ce.cdr_event_id) as event_count
    FROM operation_types ot
    JOIN cdr_events ce ON ot.operation_type_id = ce.operation_type_id
    GROUP BY ot.description
    ORDER BY event_count DESC
    LIMIT 1
    """
    df_top_operation = query_to_dataframe(top_operation_query)
    stats['most_popular_operation'] = df_top_operation.iloc[0].to_dict() if not df_top_operation.empty else {
        'description': 'N/A', 'event_count': 0}

    # --- Derive Conclusions ---
    # These are simple textual interpretations of the statistics, providing
    # actionable insights for the company. They are designed to be informative
    # even with limited data, by checking for zero values.
    conclusions = []

    if stats['total_customers'] > 0:
        conclusions.append(f"The platform currently serves a total of {stats['total_customers']} unique subscribers.")
        if stats['active_customers_year'] > 0:
            active_percentage_year = (stats['active_customers_year'] / stats['total_customers']) * 100
            conclusions.append(
                f"Approximately {active_percentage_year:.2f}% of your customer base ({stats['active_customers_year']} customers) has shown activity in the last year, indicating a core active user segment.")
        else:
            conclusions.append(
                "No customer activity recorded in the last year, suggesting a need for re-engagement strategies or data review.")

        if stats['avg_revenue_per_customer'] > 0:
            conclusions.append(
                f"The average revenue generated per customer from usage operations is {stats['avg_revenue_per_customer']:.2f}, highlighting the value of each active user.")
        if stats['avg_events_per_customer'] > 0:
            conclusions.append(
                f"On average, each customer generates {stats['avg_events_per_customer']:.2f} CDR events, providing insight into overall engagement levels.")

    if stats['total_cdr_events'] > 0:
        conclusions.append(
            f"A substantial volume of {stats['total_cdr_events']} CDR events have been processed across all operations.")
        if stats['total_revenue'] > 0:
            conclusions.append(
                f"Total revenue from usage operations stands at {stats['total_revenue']:.2f}, which is a key performance indicator for service monetization.")
        else:
            conclusions.append(
                "While events are recorded, no revenue has been generated from usage operations, suggesting a potential issue with monetization models or data categorization.")
    else:
        conclusions.append(
            "No CDR events recorded in the system, which severely limits the ability to perform usage and revenue analysis. Data ingestion should be verified.")

    if stats['total_offers'] > 0:
        conclusions.append(f"There are {stats['total_offers']} distinct offers available in the system.")
        if stats['unique_active_offers'] > 0:
            active_offers_percentage = (stats['unique_active_offers'] / stats['total_offers']) * 100
            conclusions.append(
                f"Out of these, {stats['unique_active_offers']} ({active_offers_percentage:.2f}%) offers have recorded actual customer activity, indicating which offers are gaining traction.")
        else:
            conclusions.append(
                "Despite having offers, none have recorded customer activity, suggesting a need to promote existing offers or introduce more appealing options.")

    if stats['most_popular_offer']['subscriber_count'] > 0:
        conclusions.append(
            f"The most popular offer, '{stats['most_popular_offer']['offer_name']}', has attracted {stats['most_popular_offer']['subscriber_count']} unique subscribers. This offer could be a model for future product development.")

    if stats['most_popular_operation']['event_count'] > 0:
        conclusions.append(
            f"The most frequent operation type is '{stats['most_popular_operation']['description']}' with {stats['most_popular_operation']['event_count']} events. This highlights a primary area of customer interaction.")
        if stats['unique_operation_types'] > 0:
            conclusions.append(
                f"A total of {stats['unique_operation_types']} different operation types are being utilized by customers.")

    if stats['unique_object_types_used'] > 0:
        conclusions.append(
            f"Customers are actively using {stats['unique_object_types_used']} different types of balance objects, indicating diverse service consumption.")
    else:
        conclusions.append(
            "No object types are currently in use by customers, which might impact service delivery or customer experience related to balances.")

    return jsonify({
        'stats': stats,
        'conclusions': conclusions
    })


if __name__ == '__main__':
    # This block runs the Flask development server when the script is executed directly.
    # debug=True enables debug mode, which provides helpful error messages and auto-reloads
    # the server on code changes.
    # host='0.0.0.0' makes the server accessible from any IP address, useful in containerized
    # environments or when accessing from another machine on the same network.
    app.run(debug=True, host='0.0.0.0', port=5000)
