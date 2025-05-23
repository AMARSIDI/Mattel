import os
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
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
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import secrets
import string
from sqlalchemy import create_engine # Import create_engine


# --- Flask Application Initialization and Configuration ---

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

# Load application configuration from environment variables or provide defaults.
# This makes the application more flexible and secure, as sensitive
# information like database credentials are not hardcoded directly in the script.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
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


# --- Core Helper Functions (Database, Plotly, Authentication) ---

# Global SQLAlchemy engine to be created once
db_engine = None

def get_db_engine():
    """
    Creates and returns an SQLAlchemy engine for connecting to the MySQL database.
    This engine is more robust for complex queries with Pandas.
    """
    global db_engine
    if db_engine is None:
        try:
            # Construct the database URI for SQLAlchemy
            # Ensure mysqlclient is installed for better performance with SQLAlchemy
            # pip install mysqlclient # or pip install mysql-connector-python
            db_uri = (
                f"mysql+mysqlconnector://{DATABASE_CONFIG['user']}:"
                f"{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}/"
                f"{DATABASE_CONFIG['database']}"
            )
            db_engine = create_engine(db_uri)
            print("SQLAlchemy engine created successfully.")
        except Exception as e:
            print(f"Error creating SQLAlchemy engine: {e}")
            db_engine = None # Ensure engine is None if creation fails
    return db_engine


def query_to_dataframe(query, params=None):
    """
    Executes a SQL query using an SQLAlchemy engine and returns the results as a Pandas DataFrame.
    This function is a wrapper to simplify database interactions and improve compatibility
    with complex SQL queries like CTEs and Window Functions (requires MySQL 8.0+).
    """
    engine = get_db_engine()
    if engine:
        try:
            # Use pd.read_sql_query with the SQLAlchemy engine
            # Parameters are passed directly to read_sql_query for sanitization
            df = pd.read_sql_query(sql=query, con=engine, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            return pd.DataFrame()  # Return empty DataFrame on query execution error
    return pd.DataFrame()  # Return empty DataFrame if engine creation fails


def create_figure_json(fig):
    """
    Converts a Plotly figure object into a JSON string.
    This JSON string can then be sent to the frontend and parsed by Plotly.js
    to render the chart.
    """
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_verification_token():
    """Generate a random verification token"""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(50))


def login_required(f):
    """Decorator to require login for protected routes"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator to require admin role for protected routes"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        if session.get('role_name') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)

    return decorated_function


def get_employee_by_email(email):
    """Get employee information by email"""
    query = """
    SELECT e.employee_id, e.employee_gmail, e.password_hash, e.is_active, 
           e.is_email_verified, r.role_name
    FROM employees e
    JOIN roles r ON e.role_id = r.role_id
    WHERE e.employee_gmail = %s
    """
    conn = None # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG) # Use direct connector for auth operations
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        return result
    except Exception as e:
        print(f"Error getting employee: {e}")
        return None
    finally:
        if conn and conn.is_connected(): # Check if conn is not None before closing
            conn.close()
    return None


def create_employee(email, password, role_name='employee'):
    """Create a new employee account"""
    conn = None # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG) # Use direct connector for auth operations
        cursor = conn.cursor()

        # Get role_id
        cursor.execute("SELECT role_id FROM roles WHERE role_name = %s", (role_name,))
        role_result = cursor.fetchone()
        if not role_result:
            return False, "Invalid role"

        role_id = role_result[0]
        password_hash = generate_password_hash(password)
        verification_token = generate_verification_token()
        token_expires = datetime.now() + timedelta(hours=24)

        query = """
        INSERT INTO employees (employee_gmail, password_hash, role_id, 
                             email_verification_token, email_verification_token_expires_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (email, password_hash, role_id, verification_token, token_expires))
        conn.commit()

        return True, "Account created successfully"
    except mysql.connector.IntegrityError:
        return False, "Email already exists"
    except Exception as e:
        print(f"Error creating employee: {e}")
        return False, "Failed to create account"
    finally:
        if conn and conn.is_connected(): # Check if conn is not None before closing
            conn.close()
    return False, "Database connection failed"

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


# --- Authentication and Base Routes ---

@app.route('/auth/signup', methods=['GET', 'POST'])
def signup():
    """Employee signup page and handler"""
    if request.method == 'GET':
        return render_template('auth/signup.html')

    try:
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')

        # Basic validation
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        # Create employee account
        success, message = create_employee(email, password)

        if success:
            return jsonify({'message': message, 'redirect': '/auth/login'}), 201
        else:
            return jsonify({'error': message}), 400

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'error': 'An error occurred during signup'}), 500


@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    """Employee login page and handler"""
    if request.method == 'GET':
        return render_template('auth/login.html')

    try:
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        # Get employee from database
        employee = get_employee_by_email(email)

        if not employee:
            return jsonify({'error': 'Invalid email or password'}), 401

        if not employee['is_active']:
            return jsonify({'error': 'Account is deactivated'}), 401

        # Check password
        if not check_password_hash(employee['password_hash'], password):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Create session
        session['employee_id'] = employee['employee_id']
        session['employee_email'] = employee['employee_gmail']
        session['role_name'] = employee['role_name']
        session['is_email_verified'] = employee['is_email_verified']

        return jsonify({
            'message': 'Login successful',
            'redirect': '/dashboard',
            'user': {
                'email': employee['employee_gmail'],
                'role': employee['role_name']
            }
        }), 200

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'An error occurred during login'}), 500


@app.route('/auth/logout', methods=['POST'])
def logout():
    """Employee logout"""
    session.clear()
    return jsonify({'message': 'Logged out successfully', 'redirect': '/auth/login'}), 200


@app.route('/auth/profile')
@login_required
def profile():
    """Get current user profile"""
    return jsonify({
        'employee_id': session['employee_id'],
        'email': session['employee_email'],
        'role': session['role_name'],
        'is_email_verified': session.get('is_email_verified', False)
    })


@app.route('/auth/check')
def check_auth():
    """Check if user is authenticated"""
    if 'employee_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'email': session['employee_email'],
                'role': session['role_name']
            }
        })
    return jsonify({'authenticated': False})

@app.route('/')
def index():
    """Redirect to dashboard if logged in, otherwise to login"""
    if 'employee_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - requires authentication"""
    return render_template('index.html')


# --- Admin Routes ---

@app.route('/admin/employees')
@admin_required
def list_employees():
    """List all employees (admin only)"""
    query = """
    SELECT e.employee_id, e.employee_gmail, e.is_active, 
           e.is_email_verified, r.role_name, e.created_at
    FROM employees e
    JOIN roles r ON e.role_id = r.role_id
    ORDER BY e.created_at DESC
    """
    df = query_to_dataframe(query)
    return jsonify(df.to_dict('records'))


@app.route('/admin/employees/<int:employee_id>/toggle', methods=['POST'])
@admin_required
def toggle_employee_status(employee_id):
    """Toggle employee active status (admin only)"""
    conn = None # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG) # Use direct connector for auth operations
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE employees SET is_active = NOT is_active WHERE employee_id = %s",
            (employee_id,)
        )
        conn.commit()
        return jsonify({'message': 'Employee status updated'})
    except Exception as e:
        print(f"Error toggling employee status: {e}")
        return jsonify({'error': 'Failed to update employee status'}), 500
    finally:
        if conn and conn.is_connected(): # Check if conn is not None before closing
            conn.close()
    return jsonify({'error': 'Database connection failed'}), 500


# --- Utility Function for Initial Admin Creation ---

def create_first_admin():
    """Helper function to create the first admin user if none exists."""
    conn = None # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG) # Use direct connector for auth operations
        cursor = conn.cursor()
        # Check if any admin exists
        cursor.execute("""
                SELECT COUNT(*) FROM employees e 
                JOIN roles r ON e.role_id = r.role_id 
                WHERE r.role_name = 'admin'
            """)
        admin_count = cursor.fetchone()[0]

        if admin_count == 0:
            # Create first admin
            admin_email = "admin@mattel.com"  # <--- IMPORTANT: Change this in production
            admin_password = "admin123"  # <--- IMPORTANT: Change this in production

            cursor.execute("SELECT role_id FROM roles WHERE role_name = 'admin'")
            role_id = cursor.fetchone()[0]

            password_hash = generate_password_hash(admin_password)

            cursor.execute("""
                    INSERT INTO employees (employee_gmail, password_hash, role_id, is_email_verified)
                    VALUES (%s, %s, %s, TRUE)
                """, (admin_email, password_hash, role_id))
            conn.commit()

            print(f"First admin created: {admin_email} / {admin_password}")
            print("Please change the default password after first login in production!")

    except Exception as e:
        print(f"Error creating first admin: {e}")
    finally:
        if conn and conn.is_connected(): # Check if conn is not None before closing
            conn.close()


# --- API Endpoints ---


""" --- 1. OVERALL SYSTEM HEALTH & KEY PERFORMANCE INDICATORS (KPIs) --- """

@app.route('/api/health')
def health_check():
    """
    API endpoint to check the health of the Flask application and its
    connection to the MySQL database.
    Returns a JSON response indicating 'healthy' or 'unhealthy' status.
    """
    try:
        conn = mysql.connector.connect(**DATABASE_CONFIG) # Use direct connector for health check
        if conn:
            conn.close()
            return jsonify({'status': 'healthy', 'database': 'connected'})
        return jsonify({'status': 'unhealthy', 'database': 'disconnected'}), 500
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/overall_stats')
@login_required
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


""" --- 2. CUSTOMER BEHAVIOR AND SEGMENTATION ANALYSIS --- """

@app.route('/api/customer_type_distribution')
@login_required
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

    if df.empty or df['customer_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No customer type distribution data available.")
    else:
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


@app.route('/api/customer_segmentation_by_spending')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_spending_segments():
    """
    Segment customers into High/Medium/Low value based on total spending (DEDUCT_AMOUNT).
    Uses percentiles to create meaningful segments.
    NOTE: This function requires MySQL 8.0+ for PERCENTILE_CONT.
    """
    query = """
    WITH customer_spending AS (
        SELECT 
            ce.cust_num,
            c.cust_type,
            SUM(ce.deduct_amount) as total_spent,
            COUNT(*) as transaction_count,
            AVG(ce.deduct_amount) as avg_transaction_amount,
            MAX(ce.creation_time) as last_activity
        FROM cdr_events ce
        JOIN customers c ON ce.cust_num = c.cust_num_bigint
        GROUP BY ce.cust_num, c.cust_type
    )
    SELECT 
        CASE 
            WHEN cs.total_spent >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_spent) FROM customer_spending) THEN 'High Value'
            WHEN cs.total_spent >= (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_spent) FROM customer_spending) THEN 'Medium Value'
            ELSE 'Low Value'
        END as spending_segment,
        cs.cust_type,
        COUNT(*) as customer_count,
        AVG(cs.total_spent) as avg_total_spent,
        AVG(cs.transaction_count) as avg_transactions,
        AVG(cs.avg_transaction_amount) as avg_transaction_size
    FROM customer_spending cs
    GROUP BY spending_segment, cs.cust_type
    ORDER BY 
        CASE spending_segment 
            WHEN 'High Value' THEN 1 
            WHEN 'Medium Value' THEN 2 
            ELSE 3 
        END,
        cs.cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No customer spending segmentation data available.")
    else:
        fig = px.bar(
            df,
            x='spending_segment',
            y='customer_count',
            color='cust_type',
            title='Customer Segmentation by Spending Level',
            labels={
                'spending_segment': 'Spending Segment',
                'customer_count': 'Number of Customers',
                'cust_type': 'Customer Type'
            },
            hover_data=['avg_total_spent', 'avg_transactions', 'avg_transaction_size']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_behavior_by_type')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_behavior_by_type():
    """
    Compare behavior differences between Subscribers (S) and Accounts (A).
    Analyzes spending patterns, usage frequency, and preferences.
    """
    query = """
    SELECT 
        c.cust_type,
        COUNT(DISTINCT ce.cust_num) as active_customers,
        AVG(ce.deduct_amount) as avg_deduct_amount,
        AVG(ce.charge_amount) as avg_charge_amount,
        SUM(ce.deduct_amount) as total_revenue,
        COUNT(ce.cdr_event_id) as total_events,
        COUNT(ce.cdr_event_id) / COUNT(DISTINCT ce.cust_num) as avg_events_per_customer,
        AVG(ce.current_amount_post) as avg_balance_after_transaction
    FROM customers c
    JOIN cdr_events ce ON c.cust_num_bigint = ce.cust_num
    GROUP BY c.cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No customer behavior comparison data available.")
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Comparison', 'Usage Frequency', 'Average Transaction', 'Customer Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['total_revenue'], name='Total Revenue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['avg_events_per_customer'], name='Avg Events per Customer'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['avg_deduct_amount'], name='Avg Deduct Amount'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['active_customers'], name='Active Customers'),
            row=2, col=2
        )

        fig.update_layout(title_text="Customer Behavior: Subscribers vs Accounts")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_offer_preferences')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_offer_preferences():
    """
    Segment customers based on their most frequently used offers.
    Creates segments like 'Data-heavy users', 'Voice-centric users', 'Loan takers'.
    NOTE: This function requires MySQL 8.0+ for ROW_NUMBER().
    """
    query = """
    WITH customer_offer_usage AS (
        SELECT 
            ce.cust_num,
            o.offer_name,
            COUNT(*) as usage_count,
            SUM(ce.deduct_amount) as total_spent_on_offer,
            ROW_NUMBER() OVER (PARTITION BY ce.cust_num ORDER BY COUNT(*) DESC) as preference_rank
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        GROUP BY ce.cust_num, o.offer_name
    ),
    customer_primary_preference AS (
        SELECT 
            cust_num,
            offer_name as primary_offer,
            CASE 
                WHEN offer_name LIKE '%Data%' OR offer_name LIKE '%Internet%' OR offer_name LIKE '%Mo%' THEN 'Data Users'
                WHEN offer_name LIKE '%Voice%' OR offer_name LIKE '%min%' OR offer_name LIKE '%SMS%' THEN 'Voice-centric users'
                WHEN offer_name LIKE '%Loan%' THEN 'Loan takers'
                ELSE 'General users'
            END as user_segment
        FROM customer_offer_usage
        WHERE preference_rank = 1
    )
    SELECT 
        user_segment,
        COUNT(*) as customer_count,
        AVG(total_revenue.total_spent) as avg_customer_value
    FROM customer_primary_preference cpp
    JOIN (
        SELECT cust_num, SUM(deduct_amount) as total_spent
        FROM cdr_events 
        GROUP BY cust_num
    ) total_revenue ON cpp.cust_num = total_revenue.cust_num
    GROUP BY user_segment
    ORDER BY customer_count DESC
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No customer offer preference data available.")
    else:
        fig = px.pie(
            df,
            values='customer_count',
            names='user_segment',
            title='Customer Segmentation by Offer Preferences',
            hover_data=['avg_customer_value']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_lifecycle_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_lifecycle_analysis():
    """
    Analyze customer lifecycle using CHARGE_AMOUNT_EXPIRE_TIME and CURRENT_AMOUNT_EXPIRE_TIME.
    Understand how quickly customers consume what they purchase.
    """
    query = """
    SELECT 
        ce.cust_num,
        c.cust_type,
        AVG(TIMESTAMPDIFF(HOUR, ce.creation_time, ce.charge_expire_time)) as avg_charge_lifetime_hours,
        AVG(TIMESTAMPDIFF(HOUR, ce.creation_time, ce.current_expire_time)) as avg_current_lifetime_hours,
        AVG(ce.charge_amount) as avg_charge_amount,
        AVG(ce.current_amount_post) as avg_final_balance,
        COUNT(*) as transaction_count
    FROM cdr_events ce
    JOIN customers c ON ce.cust_num = c.cust_num_bigint
    WHERE ce.charge_expire_time IS NOT NULL 
    AND ce.current_expire_time IS NOT NULL
    AND ce.charge_amount > 0
    GROUP BY ce.cust_num, c.cust_type
    HAVING transaction_count >= 5  -- Only customers with sufficient data
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No customer lifecycle data available.")
    else:
        fig = px.scatter(
            df,
            x='avg_charge_lifetime_hours',
            y='avg_current_lifetime_hours',
            color='cust_type',
            size='avg_charge_amount',
            title='Customer Lifecycle Analysis: Charge vs Current Amount Lifetimes',
            labels={
                'avg_charge_lifetime_hours': 'Average Charge Lifetime (Hours)',
                'avg_current_lifetime_hours': 'Average Current Amount Lifetime (Hours)',
                'cust_type': 'Customer Type'
            },
            hover_data=['avg_final_balance', 'transaction_count']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/customer_journey_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_journey_analysis():
    """
    Track customer journey from first transaction to current state.
    Shows progression through different offers and spending patterns.
    NOTE: This function requires MySQL 8.0+ for ROW_NUMBER().
    """
    query = """
    WITH customer_journey AS (
        SELECT 
            ce.cust_num,
            ce.creation_time,
            o.offer_name,
            ce.deduct_amount,
            ROW_NUMBER() OVER (PARTITION BY ce.cust_num ORDER BY ce.creation_time) as transaction_sequence,
            DATEDIFF(ce.creation_time, MIN(ce.creation_time) OVER (PARTITION BY ce.cust_num)) as days_from_first_transaction
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
    ),
    journey_stages AS (
        SELECT 
            cust_num,
            CASE 
                WHEN days_from_first_transaction <= 7 THEN 'New Customer (0-7 days)'
                WHEN days_from_first_transaction <= 30 THEN 'Early Stage (8-30 days)'
                WHEN days_from_first_transaction <= 90 THEN 'Growing (31-90 days)'
                ELSE 'Mature (90+ days)'
            END as journey_stage,
            COUNT(*) as transactions_in_stage,
            SUM(deduct_amount) as total_spent_in_stage,
            COUNT(DISTINCT offer_name) as unique_offers_tried
        FROM customer_journey
        GROUP BY cust_num, journey_stage
    )
    SELECT 
        journey_stage,
        COUNT(DISTINCT cust_num) as customers_count,
        AVG(transactions_in_stage) as avg_transactions,
        AVG(total_spent_in_stage) as avg_spending,
        AVG(unique_offers_tried) as avg_offers_tried,
        SUM(total_spent_in_stage) as total_stage_revenue
    FROM journey_stages
    GROUP BY journey_stage
    ORDER BY 
        CASE journey_stage
            WHEN 'New Customer (0-7 days)' THEN 1
            WHEN 'Early Stage (8-30 days)' THEN 2
            WHEN 'Growing (31-90 days)' THEN 3
            ELSE 4
        END
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No customer journey data available.")
    else:
        fig = go.Figure(go.Funnel(
            y=df['journey_stage'],
            x=df['customers_count'],
            textinfo="value+percent initial",
            hovertemplate='<b>%{y}</b><br>' +
                          'Customers: %{x}<br>' +
                          'Avg Spending: %{customdata[0]:.2f}<br>' +
                          'Avg Transactions: %{customdata[1]:.1f}<extra></extra>',
            customdata=df[['avg_spending', 'avg_transactions']].values
        ))

        fig.update_layout(title_text="Customer Journey Analysis - Progression Funnel")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


""" --- 3. OFFER PERFORMANCE AND PRODUCT ANALYSIS --- """

@app.route('/api/offers_summary')
@login_required
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
    LEFT JOIN cdr_events ce ON o.offer_id = o.offer_id
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No offer summary data available.")
    else:
        fig = px.bar(
            df,
            x='offer_name',
            y='subscriber_count',
            title='Number of Subscribers by Offer',
            labels={'offer_name': 'Offer Name', 'subscriber_count': 'Number of Subscribers'},
            hover_data=['event_count', 'avg_deduct_amount', 'total_deduct_amount']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/offer_performance')
@login_required
@cache.cached(timeout=cache_timeout)
def offer_performance():
    """
    API endpoint for analyzing offer performance.
    Calculates subscriber count, event count, total usage (deduct_amount),
    and average offer lifetime in days for each offer.
    Generates a scatter plot to visualize offer performance, with marker size
    representing event count.
    NOTE: This function requires MySQL 8.0+ for the subquery in DATEDIFF.
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

    if df.empty or (
            df['subscriber_count'].sum() == 0 and df['total_usage'].sum() == 0 and df['event_count'].sum() == 0):
        fig = go.Figure()
        fig.update_layout(title_text="No offer performance data available.")
    else:
        if df['subscriber_count'].sum() == 0 and df['total_usage'].sum() == 0:
            fig = go.Figure()
            fig.update_layout(title_text="Offer Performance: No meaningful usage or subscriber data.")
        else:
            fig = px.scatter(
                df,
                x='subscriber_count',
                y='total_usage',
                size=df['event_count'].apply(lambda x: max(x, 1)),
                hover_name='offer_name',
                title='Offer Performance Analysis',
                labels={
                    'subscriber_count': 'Number of Subscribers',
                    'total_usage': 'Total Usage (Amount)',
                    'event_count': 'Number of Events'
                }
            )
            if len(df) < 20:
                fig.update_traces(text=df['offer_name'], textposition='top center')

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/offer_trend_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def offer_trend_analysis():
    """
    Track the trend of offer subscriptions over time (CREATION_TIME).
    Shows which offers are growing or declining in popularity.
    """
    query = """
    SELECT 
        DATE_FORMAT(ce.creation_time, '%Y-%m') as month,
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as unique_subscribers,
        COUNT(ce.cdr_event_id) as total_subscriptions,
        SUM(ce.deduct_amount) as monthly_revenue
    FROM cdr_events ce
    JOIN offers o ON ce.offer_id = o.offer_id
    WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
    GROUP BY month, o.offer_name
    ORDER BY month, o.offer_name
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No offer trend data available.")
    else:
        top_offers = df.groupby('offer_name')['unique_subscribers'].sum().nlargest(10).index
        df_filtered = df[df['offer_name'].isin(top_offers)]

        fig = px.line(
            df_filtered,
            x='month',
            y='unique_subscribers',
            color='offer_name',
            title='Offer Subscription Trends (Top 10 Offers)',
            labels={
                'month': 'Month',
                'unique_subscribers': 'Unique Subscribers',
                'offer_name': 'Offer Name'
            },
            hover_data=['total_subscriptions', 'monthly_revenue']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig),
        'top_offers': list(top_offers)
    })


@app.route('/api/data_bundle_analysis')
@login_required
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
    LEFT JOIN cdr_events ce ON o.offer_id = o.offer_id
    WHERE (o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Mo%' OR o.offer_name LIKE '%Internet%')
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """
    df = query_to_dataframe(query)

    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No data bundle analysis data available.")
    else:
        fig = px.bar(
            df,
            x='offer_name',
            y='subscriber_count',
            color='avg_usage',
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


@app.route('/api/cross_offer_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def cross_offer_analysis():
    """
    Analyze customers who subscribe to multiple offers.
    Shows cross-selling opportunities and customer behavior patterns.
    NOTE: This function requires MySQL 8.0+ for GROUP_CONCAT.
    """
    query = """
    WITH customer_offers AS (
        SELECT 
            ce.cust_num,
            COUNT(DISTINCT ce.offer_id) as unique_offers_used,
            GROUP_CONCAT(DISTINCT o.offer_name SEPARATOR ', ') as offers_used,
            SUM(ce.deduct_amount) as total_spent,
            COUNT(ce.cdr_event_id) as total_events
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
        GROUP BY ce.cust_num
    )
    SELECT 
        unique_offers_used,
        COUNT(*) as customer_count,
        AVG(total_spent) as avg_spending_per_customer,
        AVG(total_events) as avg_events_per_customer,
        SUM(total_spent) as total_segment_revenue
    FROM customer_offers
    GROUP BY unique_offers_used
    ORDER BY unique_offers_used
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No cross-offer analysis data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=df['unique_offers_used'], y=df['customer_count'], name="Customer Count"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['unique_offers_used'], y=df['avg_spending_per_customer'],
                       name="Avg Spending per Customer", mode='lines+markers'),
            secondary_y=True,
        )

        fig.update_layout(title_text="Cross-Offer Analysis: Multi-Offer Usage Patterns")
        fig.update_yaxes(title_text="Number of Customers", secondary_y=False)
        fig.update_yaxes(title_text="Average Spending per Customer", secondary_y=True)

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


""" --- 4. FINANCIAL AND REVENUE ANALYSIS --- """

@app.route('/api/revenue_analysis')
@login_required
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

    if df.empty or df['revenue'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No revenue analysis data available.")
    else:
        fig = px.bar(
            df,
            x='month',
            y='revenue',
            color='offer_name',
            title='Monthly Revenue by Offer',
            labels={'month': 'Month', 'revenue': 'Revenue', 'offer_name': 'Offer Name'}
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/revenue_trends_by_customer_type')
@login_required
@cache.cached(timeout=cache_timeout)
def revenue_trends_by_customer_type():
    """
    Track revenue trends over time broken down by customer type (CUST_TYPE).
    Shows growth patterns for different customer segments.
    """
    query = """
    SELECT 
        DATE_FORMAT(ce.creation_time, '%Y-%m') as month,
        c.cust_type,
        SUM(ce.deduct_amount) as monthly_revenue,
        COUNT(DISTINCT ce.cust_num) as active_customers,
        COUNT(ce.cdr_event_id) as total_transactions,
        AVG(ce.deduct_amount) as avg_transaction_value
    FROM cdr_events ce
    JOIN customers c ON ce.cust_num = c.cust_num_bigint
    WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 18 MONTH)
    GROUP BY month, c.cust_type
    ORDER BY month, c.cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No revenue trends data available.")
    else:
        fig = px.area(
            df,
            x='month',
            y='monthly_revenue',
            color='cust_type',
            title='Revenue Trends by Customer Type',
            labels={
                'month': 'Month',
                'monthly_revenue': 'Monthly Revenue',
                'cust_type': 'Customer Type'
            },
            hover_data=['active_customers', 'total_transactions', 'avg_transaction_value']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/arpu_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def arpu_analysis():
    """
    Calculate Average Revenue Per User (ARPU) by customer type and time period.
    Essential metric for telecom business analysis.
    """
    query = """
    SELECT 
        DATE_FORMAT(ce.creation_time, '%Y-%m') as month,
        c.cust_type,
        COUNT(DISTINCT ce.cust_num) as active_users,
        SUM(ce.deduct_amount) as total_revenue,
        SUM(ce.deduct_amount) / COUNT(DISTINCT ce.cust_num) as arpu,
        AVG(ce.deduct_amount) as avg_transaction_value
    FROM cdr_events ce
    JOIN customers c ON ce.cust_num = c.cust_num_bigint
    WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
    GROUP BY month, c.cust_type
    ORDER BY month, c.cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No ARPU data available.")
    else:
        fig = px.line(
            df,
            x='month',
            y='arpu',
            color='cust_type',
            title='Average Revenue Per User (ARPU) Trends',
            labels={
                'month': 'Month',
                'arpu': 'ARPU',
                'cust_type': 'Customer Type'
            },
            hover_data=['active_users', 'total_revenue', 'avg_transaction_value']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/balance_flow_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def balance_flow_analysis():
    """
    Analyze the flow of funds: CHARGE_AMOUNT vs DEDUCT_AMOUNT over time.
    Shows money in vs money out patterns.
    """
    query = """
    SELECT 
        DATE_FORMAT(creation_time, '%Y-%m') as month,
        SUM(IFNULL(charge_amount, 0)) as total_charged,
        SUM(IFNULL(deduct_amount, 0)) as total_deducted,
        SUM(IFNULL(charge_amount, 0)) - SUM(IFNULL(deduct_amount, 0)) as net_balance_change,
        AVG(current_amount_post) as avg_final_balance,
        COUNT(DISTINCT cust_num) as active_customers
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
    GROUP BY month
    ORDER BY month
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No balance flow data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df['month'], y=df['total_charged'], name="Total Charged", line=dict(color='green')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['month'], y=df['total_deducted'], name="Total Deducted", line=dict(color='red')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['month'], y=df['net_balance_change'], name="Net Balance Change", line=dict(color='blue')),
            secondary_y=True,
        )

        fig.update_layout(title_text="Balance Flow Analysis: Money In vs Money Out")
        fig.update_yaxes(title_text="Amount (Charged/Deducted)", secondary_y=False)
        fig.update_yaxes(title_text="Net Balance Change", secondary_y=True)

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/loan_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def loan_analysis():
    """
    Analyze loan offers performance, repayment patterns, and risk assessment.
    NOTE: This function requires MySQL 8.0+ for CTEs.
    """
    query = """
    WITH loan_events AS (
        SELECT 
            ce.cust_num,
            c.cust_type,
            o.offer_name,
            ce.creation_time,
            ce.charge_amount as loan_amount,
            ce.deduct_amount as repayment_amount,
            ce.current_amount_post as balance_after,
            CASE WHEN o.offer_name LIKE '%Loan%' THEN 'Loan_Taken' ELSE 'Regular_Usage' END as event_type
        FROM cdr_events ce
        JOIN customers c ON ce.cust_num = c.cust_num_bigint
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE o.offer_name LIKE '%Loan%' OR ce.cust_num IN (
            SELECT DISTINCT cust_num FROM cdr_events ce2 
            JOIN offers o2 ON ce2.offer_id = o2.offer_id 
            WHERE o2.offer_name LIKE '%Loan%'
        )
    ),
    loan_customers AS (
        SELECT 
            cust_num,
            cust_type,
            COUNT(CASE WHEN event_type = 'Loan_Taken' THEN 1 END) as total_loans_taken,
            SUM(CASE WHEN event_type = 'Loan_Taken' THEN loan_amount ELSE 0 END) as total_loan_amount,
            SUM(CASE WHEN event_type = 'Regular_Usage' THEN repayment_amount ELSE 0 END) as total_repayments,
            AVG(balance_after) as avg_balance,
            MAX(creation_time) as last_activity
        FROM loan_events
        GROUP BY cust_num, cust_type
    )
    SELECT 
        cust_type,
        COUNT(*) as loan_customers,
        AVG(total_loans_taken) as avg_loans_per_customer,
        AVG(total_loan_amount) as avg_total_loan_amount,
        AVG(total_repayments) as avg_total_repayments,
        AVG(total_repayments / NULLIF(total_loan_amount, 0)) as avg_repayment_ratio,
        COUNT(CASE WHEN total_repayments >= total_loan_amount THEN 1 END) as customers_fully_repaid,
        COUNT(CASE WHEN total_repayments < total_loan_amount * 0.5 THEN 1 END) as high_risk_customers
    FROM loan_customers
    WHERE total_loans_taken > 0
    GROUP BY cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No loan analysis data available.")
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loan Customers by Type', 'Repayment Ratios', 'Risk Assessment', 'Loan Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['loan_customers'], name='Loan Customers'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['avg_repayment_ratio'], name='Avg Repayment Ratio'),
            row=1, col=2
        )

        total_high_risk = df['high_risk_customers'].sum()
        total_customers = df['loan_customers'].sum()
        total_low_risk = total_customers - total_high_risk

        fig.add_trace(
            go.Pie(labels=['Low Risk', 'High Risk'],
                   values=[total_low_risk, total_high_risk],
                   name="Risk Distribution"),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=df['cust_type'], y=df['customers_fully_repaid'], name='Fully Repaid'),
            row=2, col=2
        )

        fig.update_layout(title_text="Comprehensive Loan Analysis Dashboard")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


""" --- 5. OPERATIONAL AND RESOURCE MANAGEMENT --- """

@app.route('/api/operations_breakdown')
@login_required
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
    LEFT JOIN cdr_events ce ON ot.operation_type_id = ot.operation_type_id
    GROUP BY ot.operation_type_id, ot.description
    ORDER BY event_count DESC
    """
    df = query_to_dataframe(query)

    if df.empty or df['event_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No operations breakdown data available.")
    else:
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
@login_required
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
    LEFT JOIN objects o ON ot.object_type_id = ot.object_type_id
    GROUP BY ot.object_type_code, ot.description
    """
    df = query_to_dataframe(query)

    if df.empty or df['object_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No object type analysis data available.")
    else:
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
@login_required
@cache.cached(timeout=cache_timeout)
def time_series_analysis():
    """
    API endpoint for time series analysis of events.
    Aggregates daily event counts, total deduction amounts, and active customers
    over the last 5 years.
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

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No daily events and active customers data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=df['event_date'], y=df['event_count'], name="Event Count"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df['event_date'], y=df['active_customers'], name="Active Customers"),
            secondary_y=True,
        )
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


@app.route('/api/usage_patterns_time')
@login_required
@cache.cached(timeout=cache_timeout)
def usage_patterns_by_time():
    """
    Analyze peak usage times using CREATION_TIME.
    Shows hourly and daily patterns for better resource planning.
    """
    query = """
    SELECT 
        HOUR(creation_time) as hour_of_day,
        DAYNAME(creation_time) as day_of_week,
        DAYOFWEEK(creation_time) as day_num,
        COUNT(*) as event_count,
        SUM(deduct_amount) as total_revenue,
        COUNT(DISTINCT cust_num) as active_customers,
        AVG(deduct_amount) as avg_transaction_amount
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
    GROUP BY hour_of_day, day_of_week, day_num
    ORDER BY day_num, hour_of_day
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No usage pattern data available.")
    else:
        pivot_df = df.pivot_table(
            index='hour_of_day',
            columns='day_of_week',
            values='event_count',
            fill_value=0
        )
        day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        pivot_df = pivot_df.reindex(columns=[d for d in day_order if d in pivot_df.columns])

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig.update_layout(
            title='Usage Patterns: Hour vs Day of Week Heatmap',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day'
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/expiry_forecast')
@login_required
@cache.cached(timeout=cache_timeout)
def expiry_forecast():
    """
    API endpoint for forecasting resource expirations.
    Aggregates expiring objects and affected customers by expiry date and object type
    for objects expiring within the next year.
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

    if df.empty or df['expiring_objects'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No upcoming object expirations data available.")
    else:
        fig = px.area(
            df,
            x='expiry_date',
            y='expiring_objects',
            color='object_type',
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


@app.route('/api/expiry_impact_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def expiry_impact_analysis():
    """
    Analyze the impact of CHARGE_AMOUNT_EXPIRE_TIME and CURRENT_AMOUNT_EXPIRE_TIME
    on customer behavior and potential revenue loss.
    """
    query = """
    WITH expiry_analysis AS (
        SELECT 
            ce.cust_num,
            SUM(CASE WHEN ce.charge_expire_time < NOW() AND ce.charge_amount > 0 THEN ce.charge_amount ELSE 0 END) as expired_charged_amount,
            SUM(CASE WHEN ce.current_expire_time < NOW() AND ce.current_amount_post > 0 THEN ce.current_amount_post ELSE 0 END) as expired_current_amount,
            COUNT(CASE WHEN ce.charge_expire_time < NOW() AND ce.charge_amount > 0 THEN 1 END) as expired_charge_events,
            COUNT(CASE WHEN ce.current_expire_time < NOW() AND ce.current_amount_post > 0 THEN 1 END) as expired_current_events,
            SUM(ce.charge_amount) as total_charged,
            SUM(ce.deduct_amount) as total_deducted
        FROM cdr_events ce
        WHERE ce.charge_expire_time IS NOT NULL OR ce.current_expire_time IS NOT NULL
        GROUP BY ce.cust_num
    )
    SELECT 
        CASE 
            WHEN expired_charged_amount > 0 OR expired_current_amount > 0 THEN 'Has Expired Balances'
            ELSE 'No Expired Balances'
        END as expiry_status,
        COUNT(*) as customer_count,
        AVG(expired_charged_amount) as avg_expired_charged,
        AVG(expired_current_amount) as avg_expired_current,
        SUM(expired_charged_amount + expired_current_amount) as total_expired_value,
        AVG(total_charged) as avg_total_charged,
        AVG(expired_charged_amount / NULLIF(total_charged, 0)) as expiry_rate
    FROM expiry_analysis
    GROUP BY expiry_status
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No expiry impact data available.")
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Customer Distribution', 'Expired Value Impact'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Pie(labels=df['expiry_status'], values=df['customer_count'], name="Customers"),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df['expiry_status'], y=df['total_expired_value'], name="Total Expired Value"),
            row=1, col=2
        )

        fig.update_layout(title_text="Expiry Impact Analysis: Revenue Loss from Expired Balances")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/network_capacity_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def network_capacity_analysis():
    """
    Analyze network usage patterns to identify peak capacity needs.
    NOTE: This function requires MySQL 8.0+ for CTEs.
    """
    query = """
    WITH hourly_usage AS (
        SELECT 
            DATE(creation_time) as usage_date,
            HOUR(creation_time) as usage_hour,
            COUNT(*) as event_count,
            COUNT(DISTINCT cust_num) as active_users,
            SUM(CASE WHEN o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Internet%' THEN 1 ELSE 0 END) as data_events,
            SUM(CASE WHEN o.offer_name LIKE '%Voice%' OR o.offer_name LIKE '%min%' THEN 1 ELSE 0 END) as voice_events
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY usage_date, usage_hour
    ),
    capacity_metrics AS (
        SELECT 
            usage_hour,
            AVG(event_count) as avg_events_per_hour,
            AVG(active_users) as avg_users_per_hour,
            AVG(data_events) as avg_data_events,
            AVG(voice_events) as avg_voice_events,
            MAX(event_count) as peak_events,
            MIN(event_count) as min_events
        FROM hourly_usage
        GROUP BY usage_hour
    )
    SELECT 
        usage_hour,
        avg_events_per_hour,
        avg_users_per_hour,
        avg_data_events,
        avg_voice_events,
        peak_events,
        min_events,
        CASE 
            WHEN avg_events_per_hour > (SELECT AVG(avg_events_per_hour) * 1.2 FROM capacity_metrics) THEN 'Peak Hours'
            WHEN avg_events_per_hour < (SELECT AVG(avg_events_per_hour) * 0.8 FROM capacity_metrics) THEN 'Low Usage'
            ELSE 'Normal Usage'
        END as capacity_status
    FROM capacity_metrics
    ORDER BY usage_hour
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No network capacity data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df['usage_hour'], y=df['avg_events_per_hour'],
                       name="Average Events", mode='lines+markers'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['usage_hour'], y=df['peak_events'],
                       name="Peak Events", mode='lines+markers', line=dict(dash='dash')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['usage_hour'], y=df['avg_users_per_hour'],
                       name="Active Users", mode='lines+markers'),
            secondary_y=True,
        )

        fig.update_layout(title_text="Network Capacity Analysis - Hourly Usage Patterns")
        fig.update_yaxes(title_text="Events per Hour", secondary_y=False)
        fig.update_yaxes(title_text="Active Users per Hour", secondary_y=True)
        fig.update_xaxes(title_text="Hour of Day")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/data_quality_report')
@login_required
@cache.cached(timeout=cache_timeout)
def data_quality_report():
    """
    Data quality checks including consistency between OFFER_ID and OFfERID,
    anomalies, and outliers in amounts and timestamps.
    """
    queries = {
        'offer_consistency': """
            SELECT 
                'Offer ID Consistency' as check_type,
                COUNT(*) as total_records,
                COUNT(CASE WHEN ce.offer_id != o.offer_id THEN 1 END) as inconsistent_records,
                (COUNT(CASE WHEN ce.offer_id != o.offer_id THEN 1 END) / COUNT(*)) * 100 as inconsistency_rate
            FROM cdr_events ce
            LEFT JOIN offers o ON ce.offer_id = o.offer_id
        """,

        'amount_anomalies': """
            SELECT 
                'Amount Anomalies' as check_type,
                COUNT(*) as total_records,
                COUNT(CASE WHEN deduct_amount < 0 THEN 1 END) as negative_deductions,
                COUNT(CASE WHEN charge_amount < 0 THEN 1 END) as negative_charges,
                COUNT(CASE WHEN deduct_amount > 1000 THEN 1 END) as high_deductions,
                AVG(deduct_amount) as avg_deduct,
                STDDEV(deduct_amount) as stddev_deduct
            FROM cdr_events
        """,

        'timestamp_issues': """
            SELECT 
                'Timestamp Issues' as check_type,
                COUNT(*) as total_records,
                COUNT(CASE WHEN creation_time > NOW() THEN 1 END) as future_timestamps,
                COUNT(CASE WHEN creation_time < '2020-01-01' THEN 1 END) as very_old_timestamps,
                COUNT(CASE WHEN charge_expire_time < creation_time THEN 1 END) as invalid_expiry_times
            FROM cdr_events
        """,

        'customer_consistency': """
            SELECT 
                'Customer Consistency' as check_type,
                COUNT(DISTINCT ce.cust_num) as customers_in_events,
                COUNT(DISTINCT c.cust_num_bigint) as customers_in_master,
                COUNT(DISTINCT ce.cust_num) - COUNT(DISTINCT c.cust_num_bigint) as orphaned_customers
            FROM cdr_events ce
            LEFT JOIN customers c ON ce.cust_num = c.cust_num_bigint
        """
    }

    results = []
    for check_name, query in queries.items():
        df = query_to_dataframe(query)
        if not df.empty:
            results.extend(df.to_dict('records'))

    if not results:
        fig = go.Figure()
        fig.update_layout(title_text="No data quality report available.")
    else:
        fig = go.Figure(data=[
            go.Table(
                header=dict(values=list(results[0].keys()) if results else [],
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[[row[col] for row in results] for col in results[0].keys()] if results else [],
                           fill_color='lavender',
                           align='left'))
        ])
        fig.update_layout(title_text="Data Quality Report Summary")

    return jsonify({
        'data': results,
        'chart': create_figure_json(fig)
    })


""" --- 6. PREDICTIVE ANALYSIS AND RECOMMENDATION --- """

@app.route('/api/churn_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def churn_analysis():
    """
    API endpoint for churn analysis.
    Identifies churned customers (inactive for > 90 days) and aggregates
    churn count by quarter. Also provides a breakdown of churned customers by offer.
    NOTE: This function requires MySQL 8.0+ for CTEs.
    """
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
        WHERE ca.days_since_last_activity > 90
    )
    SELECT 
        CONCAT(churn_year, '-Q', churn_quarter) as period,
        COUNT(*) as churn_count
    FROM churned
    GROUP BY churn_year, churn_quarter
    ORDER BY churn_year, churn_quarter
    """
    df = query_to_dataframe(query)

    if df.empty or df['churn_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No customer churn data available.")
    else:
        fig = px.bar(
            df,
            x='period',
            y='churn_count',
            title='Customer Churn by Quarter',
            labels={'period': 'Time Period', 'churn_count': 'Number of Churned Customers'}
        )

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
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig),
        'offers_churn': offers_churn_df.to_dict('records')
    })


@app.route('/api/churn_risk_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def churn_risk_analysis():
    """
    Identify customers at risk of churning based on usage patterns and activity.
    NOTE: This function requires MySQL 8.0+ for CTEs.
    """
    query = """
    WITH customer_activity AS (
        SELECT 
            ce.cust_num,
            c.cust_type,
            COUNT(*) as total_transactions,
            SUM(ce.deduct_amount) as total_spent,
            MAX(ce.creation_time) as last_activity,
            DATEDIFF(NOW(), MAX(ce.creation_time)) as days_since_last_activity,
            AVG(ce.deduct_amount) as avg_transaction_amount,
            COUNT(DISTINCT DATE(ce.creation_time)) as active_days
        FROM cdr_events ce
        JOIN customers c ON ce.cust_num = c.cust_num_bigint
        WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
        GROUP BY ce.cust_num, c.cust_type
    ),
    churn_indicators AS (
        SELECT 
            *,
            CASE 
                WHEN days_since_last_activity > 30 THEN 'High Risk'
                WHEN days_since_last_activity > 14 THEN 'Medium Risk'
                WHEN total_transactions < 5 THEN 'Low Engagement'
                ELSE 'Active'
            END as churn_risk_level,
            CASE
                WHEN days_since_last_activity > 30 OR total_transactions < 3 THEN 1
                ELSE 0
            END as churn_flag
        FROM customer_activity
    )
    SELECT 
        churn_risk_level,
        cust_type,
        COUNT(*) as customer_count,
        AVG(total_spent) as avg_spent,
        AVG(days_since_last_activity) as avg_days_inactive,
        AVG(total_transactions) as avg_transactions,
        SUM(total_spent) as total_revenue_at_risk
    FROM churn_indicators
    GROUP BY churn_risk_level, cust_type
    ORDER BY 
        CASE churn_risk_level 
            WHEN 'High Risk' THEN 1 
            WHEN 'Medium Risk' THEN 2 
            WHEN 'Low Engagement' THEN 3
            ELSE 4 
        END,
        cust_type
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No churn risk data available.")
    else:
        fig = px.bar(
            df,
            x='churn_risk_level',
            y='customer_count',
            color='cust_type',
            title='Customer Churn Risk Analysis',
            labels={
                'churn_risk_level': 'Churn Risk Level',
                'customer_count': 'Number of Customers',
                'cust_type': 'Customer Type'
            },
            hover_data=['avg_spent', 'avg_days_inactive', 'total_revenue_at_risk']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/seasonal_usage_prediction')
@login_required
@cache.cached(timeout=cache_timeout)
def seasonal_usage_prediction():
    """
    Analyze seasonal patterns and predict future usage based on historical data.
    NOTE: This function requires MySQL 8.0+ for CTEs.
    """
    query = """
    WITH monthly_usage AS (
        SELECT 
            YEAR(creation_time) as year,
            MONTH(creation_time) as month,
            MONTHNAME(creation_time) as month_name,
            COUNT(*) as total_events,
            SUM(deduct_amount) as total_revenue,
            COUNT(DISTINCT cust_num) as active_customers,
            AVG(deduct_amount) as avg_transaction_value
        FROM cdr_events
        WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 24 MONTH)
        GROUP BY year, month, month_name
    ),
    seasonal_patterns AS (
        SELECT 
            month,
            month_name,
            AVG(total_events) as avg_monthly_events,
            AVG(total_revenue) as avg_monthly_revenue,
            AVG(active_customers) as avg_monthly_customers,
            STDDEV(total_revenue) as revenue_volatility,
            COUNT(*) as data_points
        FROM monthly_usage
        GROUP BY month, month_name
    )
    SELECT 
        month,
        month_name,
        avg_monthly_events,
        avg_monthly_revenue,
        avg_monthly_customers,
        revenue_volatility,
        CASE 
            WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN 'Peak Season'
            WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.9 FROM seasonal_patterns) THEN 'Low Season'
            ELSE 'Normal Season'
        END as season_type,
        data_points
    FROM seasonal_patterns
    ORDER BY month
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No seasonal usage data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df['month_name'], y=df['avg_monthly_revenue'],
                       name="Average Monthly Revenue", mode='lines+markers'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df['month_name'], y=df['avg_monthly_customers'],
                       name="Average Monthly Customers", mode='lines+markers'),
            secondary_y=True,
        )

        fig.update_layout(title_text="Seasonal Usage Patterns and Predictions")
        fig.update_yaxes(title_text="Average Monthly Revenue", secondary_y=False)
        fig.update_yaxes(title_text="Average Monthly Customers", secondary_y=True)

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


@app.route('/api/next_offer_recommendation')
@login_required
@cache.cached(timeout=cache_timeout)
def next_offer_recommendation():
    """
    Recommend next offers based on customer's current usage and offer history.
    Uses collaborative filtering approach.
    NOTE: This function requires MySQL 8.0+ for CTEs and ROW_NUMBER().
    """
    query = """
    WITH customer_offer_matrix AS (
        SELECT 
            ce.cust_num,
            o.offer_name,
            COUNT(*) as usage_frequency,
            SUM(ce.deduct_amount) as total_spent,
            MAX(ce.creation_time) as last_used
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
        GROUP BY ce.cust_num, o.offer_name
    ),
    customer_segments AS (
        SELECT 
            cust_num,
            CASE 
                WHEN offer_name LIKE '%Data%' OR offer_name LIKE '%Internet%' OR offer_name LIKE '%Mo%' THEN 'Data Users'
                WHEN offer_name LIKE '%Voice%' OR offer_name LIKE '%min%' THEN 'Voice Users'
                WHEN offer_name LIKE '%Loan%' THEN 'Loan Users'
                ELSE 'Mixed Users'
            END as user_segment,
            offer_name as preferred_offer
        FROM customer_offer_matrix
        WHERE usage_frequency = (
            SELECT MAX(usage_frequency) 
            FROM customer_offer_matrix com2 
            WHERE com2.cust_num = customer_offer_matrix.cust_num
        )
    ),
    recommendations AS (
        SELECT 
            cs.user_segment,
            o.offer_name as recommended_offer,
            COUNT(DISTINCT ce.cust_num) as users_count,
            AVG(ce.deduct_amount) as avg_revenue_per_use,
            SUM(ce.deduct_amount) as total_revenue
        FROM customer_segments cs
        JOIN cdr_events ce ON ce.cust_num != cs.cust_num
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE CASE 
            WHEN cs.user_segment = 'Data Users' THEN o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Internet%'
            WHEN cs.user_segment = 'Voice Users' THEN o.offer_name LIKE '%Voice%' OR o.offer_name LIKE '%min%'
            WHEN cs.user_segment = 'Loan Users' THEN o.offer_name LIKE '%Loan%'
            ELSE TRUE
        END
        AND o.offer_name != cs.preferred_offer
        GROUP BY cs.user_segment, o.offer_name
    )
    SELECT 
        user_segment,
        recommended_offer,
        users_count,
        avg_revenue_per_use,
        total_revenue,
        ROW_NUMBER() OVER (PARTITION BY user_segment ORDER BY total_revenue DESC) as recommendation_rank
    FROM recommendations
    WHERE recommendation_rank <= 3
    ORDER BY user_segment, recommendation_rank
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No recommendation data available.")
    else:
        fig = px.bar(
            df,
            x='user_segment',
            y='total_revenue',
            color='recommended_offer',
            title='Next Offer Recommendations by User Segment',
            labels={
                'user_segment': 'User Segment',
                'total_revenue': 'Potential Revenue',
                'recommended_offer': 'Recommended Offer'
            },
            hover_data=['users_count', 'avg_revenue_per_use']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


""" --- 7. DETAILED ENTITY LOOKUPS --- """

@app.route('/api/customer_details/<int:cust_num>')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_details(cust_num):
    """
    API endpoint for detailed information about a specific customer.
    Fetches customer's basic info, associated objects/balances, and recent CDR events.
    This is a lookup endpoint, returning raw data, not a chart.
    """
    customer_query = """
    SELECT cust_num_bigint, cust_type FROM customers WHERE cust_num_bigint = %s
    """
    customer_df = query_to_dataframe(customer_query, params=(cust_num,))
    if customer_df.empty:
        return jsonify({'error': 'Customer not found'}), 404

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
@login_required
@cache.cached(timeout=cache_timeout)
def object_details(object_id):
    """
    API endpoint for detailed information about a specific object.
    Fetches object's basic info and related CDR events.
    This is a lookup endpoint, returning raw data, not a chart.
    """
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


""" --- 8. REAL-TIME MONITORING --- """

@app.route('/api/realtime_dashboard')
@login_required
@cache.cached(timeout=60)  # Shorter cache for real-time data
def realtime_dashboard():
    """
    Real-time dashboard showing current system activity and key metrics.
    """
    query = """
    SELECT 
        'current_hour' as metric_type,
        COUNT(*) as current_hour_events,
        COUNT(DISTINCT cust_num) as current_hour_users,
        SUM(deduct_amount) as current_hour_revenue,
        AVG(deduct_amount) as avg_transaction_value
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

    UNION ALL

    SELECT 
        'today' as metric_type,
        COUNT(*) as current_hour_events,
        COUNT(DISTINCT cust_num) as current_hour_users,
        SUM(deduct_amount) as current_hour_revenue,
        AVG(deduct_amount) as avg_transaction_value
    FROM cdr_events
    WHERE DATE(creation_time) = CURDATE()

    UNION ALL

    SELECT 
        'yesterday' as metric_type,
        COUNT(*) as current_hour_events,
        COUNT(DISTINCT cust_num) as current_hour_users,
        SUM(deduct_amount) as current_hour_revenue,
        AVG(deduct_amount) as avg_transaction_value
    FROM cdr_events
    WHERE DATE(creation_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No real-time data available.")
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Events', 'Users', 'Revenue', 'Avg Transaction'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=df['metric_type'], y=df['current_hour_events'], name='Events'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df['metric_type'], y=df['current_hour_users'], name='Users'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=df['metric_type'], y=df['current_hour_revenue'], name='Revenue'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=df['metric_type'], y=df['avg_transaction_value'], name='Avg Transaction'),
            row=2, col=2
        )

        fig.update_layout(title_text="Real-Time System Dashboard")

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/system_health_check')
@login_required
@cache.cached(timeout=300)
def system_health_check():
    """
    System health monitoring including data freshness and anomaly detection.
    """
    query = """
    SELECT 
        'data_freshness' as health_metric,
        MAX(creation_time) as latest_record,
        TIMESTAMPDIFF(MINUTE, MAX(creation_time), NOW()) as minutes_since_last_record,
        COUNT(*) as total_records,
        COUNT(DISTINCT cust_num) as total_customers
    FROM cdr_events

    UNION ALL

    SELECT 
        'recent_activity' as health_metric,
        NOW() as latest_record,
        0 as minutes_since_last_record,
        COUNT(*) as total_records,
        COUNT(DISTINCT cust_num) as total_customers
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
    """
    df = query_to_dataframe(query)

    health_status = "Healthy"
    if not df.empty:
        freshness_row = df[df['health_metric'] == 'data_freshness']
        if not freshness_row.empty and freshness_row.iloc[0]['minutes_since_last_record'] > 60:
            health_status = "Warning - Data not fresh"

    return jsonify({
        'data': df.to_dict('records'),
        'health_status': health_status,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/competitive_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def competitive_analysis():
    """
    Analyze offer performance against market benchmarks and identify competitive gaps.
    NOTE: This function requires MySQL 8.0+ for CTEs and RANK().
    """
    query = """
    WITH offer_performance AS (
        SELECT 
            o.offer_name,
            COUNT(DISTINCT ce.cust_num) as unique_users,
            COUNT(*) as total_usage,
            SUM(ce.deduct_amount) as total_revenue,
            AVG(ce.deduct_amount) as avg_revenue_per_use,
            DATEDIFF(NOW(), MIN(ce.creation_time)) as days_in_market
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        GROUP BY o.offer_name
    ),
    performance_benchmarks AS (
        SELECT 
            *,
            total_revenue / NULLIF(days_in_market, 0) as daily_revenue,
            unique_users / NULLIF(days_in_market, 0) as daily_user_acquisition,
            CASE 
                WHEN offer_name LIKE '%Data%' OR offer_name LIKE '%Internet%' OR offer_name LIKE '%Mo%' THEN 'Data Products'
                WHEN offer_name LIKE '%Voice%' OR offer_name LIKE '%min%' OR offer_name LIKE '%Call%' THEN 'Voice Products'
                WHEN offer_name LIKE '%Loan%' THEN 'Financial Products'
                ELSE 'Other Products'
            END as product_category
        FROM offer_performance
    )
    SELECT 
        product_category,
        offer_name,
        unique_users,
        total_revenue,
        avg_revenue_per_use,
        daily_revenue,
        daily_user_acquisition,
        RANK() OVER (PARTITION BY product_category ORDER BY total_revenue DESC) as revenue_rank,
        RANK() OVER (PARTITION BY product_category ORDER BY unique_users DESC) as user_rank
    FROM performance_benchmarks
    ORDER BY product_category, total_revenue DESC
    """
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No competitive analysis data available.")
    else:
        fig = px.scatter(
            df,
            x='unique_users',
            y='total_revenue',
            color='product_category',
            size='avg_revenue_per_use',
            hover_name='offer_name',
            title='Competitive Analysis: Offer Performance Matrix',
            labels={
                'unique_users': 'Unique Users',
                'total_revenue': 'Total Revenue',
                'product_category': 'Product Category'
            },
            hover_data=['daily_revenue', 'revenue_rank', 'user_rank']
        )

    return jsonify({
        'data': df.to_dict('records'),
        'chart': create_figure_json(fig)
    })


if __name__ == '__main__':
    # This block runs the Flask development server when the script is executed directly.
    # debug=True enables debug mode, which provides helpful error messages and auto-reloads
    # the server on code changes.
    # host='0.0.0.0' makes the server accessible from any IP address, useful in containerized
    # environments or when accessing from another machine on the same network.
    app.run(debug=True, host='0.0.0.0', port=5000)
