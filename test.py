# ==============================================================================
# 1. IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import json
import secrets
import string
import logging
import smtplib
from functools import wraps
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
from flask_cors import CORS
from flask_caching import Cache
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector  # Keep this for direct connection in auth functions if needed, though SQLAlchemy is preferred
from sqlalchemy.exc import IntegrityError  # Import IntegrityError for specific error handling
import re  # Added for potential email validation, though not directly used in provided snippets

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 2. FLASK APPLICATION CONFIGURATION
# ==============================================================================
# For local development, create a file named .env in the root directory
# and add your secret values there. Example:
#
# SECRET_KEY=your_super_secret_key
# DATABASE_PASSWORD=your_db_password
# EMAIL_USER=your_email@example.com
# EMAIL_PASSWORD=your_email_app_password

app = Flask(__name__)

# --- Core App Configuration ---
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-default-secret-key-that-you-should-change')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit

# --- Database Configuration (from environment variables) ---
DB_HOST = os.getenv('DATABASE_HOST', 'localhost')
DB_USER = os.getenv('DATABASE_USER', 'root')
DB_PASSWORD = os.getenv('DATABASE_PASSWORD', 'SidiAmar23635')  # Default for convenience
DB_NAME = os.getenv('DATABASE_NAME', 'mattel')

# SQLAlchemy Database URI for robust Pandas integration
# IMPORTANT: The advanced queries in this app require MySQL 8.0 or higher.
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# --- Caching Configuration ---
app.config['CACHE_TYPE'] = 'SimpleCache'  # Simple in-memory cache
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Default cache timeout: 5 minutes
cache = Cache(app)

# --- Email Configuration (from environment variables) ---
app.config['EMAIL_HOST'] = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
app.config['EMAIL_PORT'] = int(os.getenv('EMAIL_PORT', 587))
app.config['EMAIL_USER'] = os.getenv('EMAIL_USER', '23612@isms.esp.mr')  # Default for convenience
app.config['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD', 'Lalle23612')  # Default for convenience
app.config['EMAIL_FROM'] = os.getenv('EMAIL_FROM', 'no-reply@mattel.com')
app.config['APP_BASE_URL'] = os.getenv('APP_BASE_URL', 'http://127.0.0.1:5000')

# Store database configuration in a dictionary for direct mysql.connector usage (e.g., auth functions)
# FIXED: Use the DB_* variables directly, as they are already populated from os.getenv.
# This prevents KeyError as app.config does not explicitly store these values.
DATABASE_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME
}

# Retrieve cache timeout from app config
cache_timeout = app.config['CACHE_DEFAULT_TIMEOUT']

# --- Initialize Extensions ---
CORS(app)

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
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            db_engine = create_engine(db_uri)
            app.logger.info("SQLAlchemy engine created successfully.")
        except Exception as e:
            app.logger.error(f"Error creating SQLAlchemy engine: {e}", exc_info=True)
            db_engine = None  # Ensure engine is None if creation fails
    return db_engine


# ==============================================================================
# 3. CORE HELPER & UTILITY FUNCTIONS
# ==============================================================================

def query_to_dataframe(query: str, params: dict = None) -> pd.DataFrame:
    """Executes a SQL query using the global engine and returns a Pandas DataFrame."""
    engine = get_db_engine()
    if engine:
        try:
            df = pd.read_sql_query(sql=query, con=engine, params=params)
            return df
        except Exception as e:
            app.logger.error(f"Error executing query: {e}\nQuery: {query}", exc_info=True)
            return pd.DataFrame()
    app.logger.error("Database engine not initialized. Cannot execute query.")
    return pd.DataFrame()


def create_figure_json(fig: go.Figure) -> str:
    """Converts a Plotly figure to a JSON string for frontend rendering."""
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def send_email(to_email: str, subject: str, body_html: str):
    """Sends an HTML email using configured settings."""
    if not all([app.config['EMAIL_USER'], app.config['EMAIL_PASSWORD']]):
        app.logger.warning("Email credentials not set in environment. Skipping email sending.")
        return False

    msg = MIMEMultipart('alternative')
    msg['From'] = f"Mattel Analytics <{app.config['EMAIL_FROM']}>"
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body_html, 'html'))

    try:
        with smtplib.SMTP(app.config['EMAIL_HOST'], app.config['EMAIL_PORT']) as smtp:
            smtp.starttls()
            smtp.login(app.config['EMAIL_USER'], app.config['EMAIL_PASSWORD'])
            smtp.send_message(msg)
        app.logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        app.logger.error(f"Email sending to {to_email} failed: {e}", exc_info=True)
        return False


def generate_verification_token():
    """Generate a random verification token"""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(50))


@app.template_filter('format_number')
def format_number_filter(value):
    """Jinja2 filter to format numbers with commas for display in templates."""
    try:
        num_value = float(value)
        # Format floats with 2 decimal places, integers without
        if isinstance(value, float) and value % 1 != 0:  # Check if it's a float with decimal part
            return f"{num_value:,.1f}"
        return f"{int(num_value):,}"
    except (ValueError, TypeError):
        return value


# ==============================================================================
# 4. AUTHENTICATION & USER MANAGEMENT
# ==============================================================================

# --- Auth Decorators ---

def login_required(f):
    """Decorator to protect routes that require a logged-in user."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_id' not in session:
            # For API calls, return JSON. For page loads, redirect.
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Authentication required', 'redirect': url_for('login')}), 401
            flash("You must be logged in to view this page.", "info")
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator to protect routes that require an admin user."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First, ensure user is logged in
        if 'employee_id' not in session:
            flash("You must be logged in to view this page.", "info")
            return redirect(url_for('login'))
        # Then, check for admin role
        if session.get('role_name') != 'admin':
            flash("Admin access is required to view this page.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)

    return decorated_function


# --- Auth Helper Functions ---

def get_employee_by_email(email: str) -> dict:
    """Retrieves an employee's details from the database by their email using SQLAlchemy."""
    query = text("""
        SELECT e.employee_id, e.employee_gmail, e.password_hash, e.is_active, e.is_email_verified, r.role_name, r.role_id
        FROM employees e JOIN roles r ON e.role_id = r.role_id
        WHERE e.employee_gmail = :email
    """)
    engine = get_db_engine()
    if not engine:
        app.logger.error("Database engine not available for get_employee_by_email.")
        return None
    with engine.connect() as conn:
        result = conn.execute(query, {'email': email}).mappings().fetchone()
    return result


def create_employee(email: str, password: str, phone: str = None, role_name='employee') -> (bool, str):
    """Creates a new, inactive employee account."""
    if get_employee_by_email(email):
        return False, "An account with this email already exists."

    password_hash = generate_password_hash(password)
    verification_token = secrets.token_urlsafe(32)
    token_expires = datetime.now() + timedelta(hours=24)

    engine = get_db_engine()
    if not engine:
        return False, "Database connection not available."

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            role_query = text("SELECT role_id FROM roles WHERE role_name = :role_name")
            role_id = conn.execute(role_query, {'role_name': role_name}).scalar()
            if not role_id:
                return False, f"Default '{role_name}' role not found in database."

            insert_query = text("""
                INSERT INTO employees (employee_gmail, employee_phone, password_hash, role_id, is_active, email_verification_token, email_verification_token_expires_at, is_email_verified, created_at, updated_at)
                VALUES (:email, :phone, :p_hash, :role_id, FALSE, :token, :expires, FALSE, NOW(), NOW())
            """)
            conn.execute(insert_query, {
                'email': email, 'phone': phone, 'p_hash': password_hash, 'role_id': role_id,
                'token': verification_token, 'expires': token_expires
            })
            trans.commit()
            return True, "Account created. Please wait for an administrator to activate it."
        except Exception as e:
            trans.rollback()
            app.logger.error(f"Error creating employee {email}: {e}", exc_info=True)
            return False, "Failed to create account due to a database error."


# --- Authentication Routes ---

@app.route('/auth/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')

        if not email or len(password) < 8 or password != confirm_password:
            return jsonify({'error': 'Please provide a valid email and matching passwords (min 8 chars).'}), 400

        success, message = create_employee(email, password)
        if success:
            return jsonify({'message': message, 'redirect': url_for('login')}), 201
        else:
            return jsonify({'error': message}), 400
    return render_template('pages-sign-up.html')


@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        employee = get_employee_by_email(email)
        if not employee or not check_password_hash(employee['password_hash'], password):
            return jsonify({'error': 'Invalid email or password.'}), 401
        if not employee['is_active']:
            return jsonify({'error': 'This account is not active. Please contact an administrator.'}), 403

        session.clear()
        session['employee_id'] = employee['employee_id']
        session['employee_email'] = employee['employee_gmail']
        session['role_name'] = employee['role_name']
        session['is_email_verified'] = employee['is_email_verified']
        session['role_id'] = employee['role_id']  # Store role_id for admin checks
        app.logger.info(f"User {email} logged in successfully.")
        return jsonify({'message': 'Login successful', 'redirect': url_for('dashboard')}), 200
    return render_template('pages-sign-in.html')


@app.route('/auth/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully', 'redirect': url_for('login')}), 200


@app.route('/auth/check')
def check_auth_status():
    """Endpoint for the frontend to check if a user is currently authenticated."""
    if 'employee_id' in session:
        return jsonify(
            {'authenticated': True, 'user': {'email': session.get('employee_email'), 'role': session.get('role_name')}})
    return jsonify({'authenticated': False})


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


# ==============================================================================
# 5. CORE APPLICATION & DASHBOARD ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Redirects to the appropriate page based on login status."""
    return redirect(url_for('dashboard')) if 'employee_id' in session else redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """
    Renders the main dashboard shell. Data is loaded asynchronously by JavaScript
    calling the /api/kpi_summary endpoint.
    """
    # Fetch overall stats directly here for initial render of template KPI cards
    # Query for total revenue, charges, operations
    query_kpis = "SELECT SUM(deduct_amount) as total_revenue, SUM(charge_amount) as total_charges, COUNT(*) as number_of_operations FROM cdr_events"
    df_kpis = query_to_dataframe(query_kpis)

    # Query for active customers (using existing logic from overall_stats)
    query_active_customers = """
    SELECT COUNT(DISTINCT cust_num) as active_customers_year
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
    """
    df_active_customers = query_to_dataframe(query_active_customers)

    # Extract values, handling potential None from empty tables
    total_revenue = 0
    if not df_kpis.empty and pd.notna(df_kpis['total_revenue'].iloc[0]):
        total_revenue = int(df_kpis['total_revenue'].iloc[0])

    total_charges = 0
    if not df_kpis.empty and pd.notna(df_kpis['total_charges'].iloc[0]):
        total_charges = int(df_kpis['total_charges'].iloc[0])

    total_deducted = total_revenue  # Assuming total_deducted is same as total_revenue from usage

    number_of_operations = 0
    if not df_kpis.empty and pd.notna(df_kpis['number_of_operations'].iloc[0]):
        number_of_operations = int(df_kpis['number_of_operations'].iloc[0])

    active_customers = 0
    if not df_active_customers.empty and pd.notna(df_active_customers['active_customers_year'].iloc[0]):
        active_customers = int(df_active_customers['active_customers_year'].iloc[0])

    # Calculate ARPU (requires total_revenue and total_customers from 'customers' table)
    query_total_customers = "SELECT COUNT(*) as total_customers FROM customers"
    df_total_customers = query_to_dataframe(query_total_customers)
    total_customers_overall = 0
    if not df_total_customers.empty and pd.notna(df_total_customers['total_customers'].iloc[0]):
        total_customers_overall = int(df_total_customers['total_customers'].iloc[0])

    arpu = total_revenue / total_customers_overall if total_customers_overall > 0 else 0.0

    # Pass the role_name and all KPI values to the template
    user_role = session.get('role_name')

    return render_template(
        'INDEX22.html',
        page_title="Mattel Analytics Dashboard",
        current_year=datetime.now().year,
        user_role=user_role,
        total_revenue=total_revenue,
        total_charges=total_charges,
        total_deducted=total_deducted,
        number_of_operations=number_of_operations,
        active_customers=active_customers,
        arpu=arpu
    )


@app.route('/analytics_modules')
@login_required
def analytics_modules_page():
    """
    Renders the analytics modules shell. All charts are loaded asynchronously
    by JavaScript making calls to the various /api/... endpoints. This makes the
    initial page load very fast.
    """
    user_role = session.get('role_name')

    # Fetch data from all relevant API endpoints
    # Use .get('chart', '') to safely retrieve chart data, defaulting to an empty string if None or missing
    # And .get('data', []) for table data, defaulting to an empty list
    overall_stats_response = overall_stats().json
    offers_summary_response = offers_summary().json
    offer_performance_response = offer_performance().json
    offer_trend_analysis_response = offer_trend_analysis().json
    data_bundle_analysis_response = data_bundle_analysis().json
    cross_offer_analysis_response = cross_offer_analysis().json
    revenue_analysis_response = revenue_analysis().json
    revenue_trends_by_customer_type_response = revenue_trends_by_customer_type().json
    arpu_analysis_response = arpu_analysis().json
    balance_flow_analysis_response = balance_flow_analysis().json
    loan_analysis_response = loan_analysis().json
    operations_breakdown_response = operations_breakdown().json
    object_types_analysis_response = object_types_analysis().json
    time_series_analysis_response = time_series_analysis().json
    usage_patterns_by_time_response = usage_patterns_by_time().json
    expiry_forecast_response = expiry_forecast().json
    expiry_impact_analysis_response = expiry_impact_analysis().json
    network_capacity_analysis_response = network_capacity_analysis().json
    data_quality_report_response = data_quality_report().json
    churn_analysis_response = churn_analysis().json
    churn_risk_analysis_response = churn_risk_analysis().json
    seasonal_usage_prediction_response = seasonal_usage_prediction().json
    next_offer_recommendation_response = next_offer_recommendation().json
    customer_type_distribution_response = customer_type_distribution().json
    customer_spending_segments_response = customer_spending_segments().json
    customer_behavior_by_type_response = customer_behavior_by_type().json
    customer_offer_preferences_response = customer_offer_preferences().json
    customer_lifecycle_analysis_response = customer_lifecycle_analysis().json
    customer_journey_analysis_response = customer_journey_analysis().json

    return render_template(
        'analytics_modules.html',
        user_role=user_role,
        page_title="Analytics Modules",
        overall_stats=overall_stats_response,
        offers_summary=offers_summary_response,
        offer_performance=offer_performance_response,
        offer_trend_analysis=offer_trend_analysis_response,
        data_bundle_analysis=data_bundle_analysis_response,
        cross_offer_analysis=cross_offer_analysis_response,
        revenue_analysis=revenue_analysis_response,
        revenue_trends_by_customer_type=revenue_trends_by_customer_type_response,
        arpu_analysis=arpu_analysis_response,
        balance_flow_analysis=balance_flow_analysis_response,
        loan_analysis=loan_analysis_response,
        operations_breakdown=operations_breakdown_response,
        object_types_analysis=object_types_analysis_response,
        time_series_analysis=time_series_analysis_response,
        usage_patterns_by_time=usage_patterns_by_time_response,
        expiry_forecast=expiry_forecast_response,
        expiry_impact_analysis=expiry_impact_analysis_response,
        network_capacity_analysis=network_capacity_analysis_response,
        data_quality_report=data_quality_report_response,
        churn_analysis=churn_analysis_response,
        churn_risk_analysis=churn_risk_analysis_response,
        seasonal_usage_prediction=seasonal_usage_prediction_response,
        next_offer_recommendation=next_offer_recommendation_response,
        customer_type_distribution=customer_type_distribution_response,
        customer_spending_segments=customer_spending_segments_response,
        customer_behavior_by_type=customer_behavior_by_type_response,
        customer_offer_preferences=customer_offer_preferences_response,
        customer_lifecycle_analysis=customer_lifecycle_analysis_response,
        customer_journey_analysis=customer_journey_analysis_response
    )


@app.route('/blank')
@login_required
def blank():
    return render_template('introduction.html', user_role=session.get('role_name'))


@app.route('/settings')
@login_required
def settings():
    return render_template('pages-settings.html', user_role=session.get('role_name'))


# ==============================================================================
# 6. ADMIN-ONLY ROUTES
# ==============================================================================

@app.route("/admin/upload", methods=["GET"])
@admin_required
def upload_page():
    """Renders the data upload page."""
    user_role = session.get('role_name')
    return render_template("uploads.html", user_role=user_role)


@app.route("/admin/upload", methods=["POST"])
@admin_required
def handle_upload():
    """
    Handles the Excel file upload, processes all relevant sheets,
    and populates the database tables in the correct order.
    """
    if "file" not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({'error': 'No file selected.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Invalid file type. Please upload an Excel file (.xlsx, .xls).'}), 400

    engine = get_db_engine()
    if not engine:
        return jsonify({'error': 'Database connection could not be established.'}), 500

    try:
        # Read all necessary sheets from the uploaded Excel file
        xls = pd.ExcelFile(file)
        # The 'DESCRIPTION' sheet is a data dictionary and should not be processed for insertion.
        # We only need the data from 'DONNEES' and the offer names from 'NOM DES OFFRES'.
        df_main = pd.read_excel(xls, sheet_name='DONNEES')
        df_offers = pd.read_excel(xls, sheet_name='NOM DES OFFRES')

    except Exception as e:
        # Update the error message to reflect the needed sheet names
        return jsonify({
            'error': f"Error reading Excel file. Make sure sheets 'DONNEES' and 'NOM DES OFFRES' exist. Details: {e}"}), 400

    # Using a transaction ensures that if any part of the process fails, all changes are rolled back.
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            inserted_counts = {
                'customers': 0, 'object_types': 0, 'operation_types': 0,
                'offers': 0, 'objects': 0, 'cdr_events': 0
            }

            # 1. NORMALIZE COLUMN NAMES
            df_main.columns = [str(col).strip().lower().replace('.', '_') for col in df_main.columns]
            df_offers.columns = [str(col).strip().lower() for col in df_offers.columns]

            # Rename columns to match database schema
            df_main.rename(columns={
                'curent_amount': 'current_amount_pre',
                'curent_amount_2': 'current_amount_post',
                'charge_amount_expire_time': 'charge_expire_time',
                'curent_amount_expire_time': 'current_amount_expire_time',
                'operation_type': 'operation_type_id'
                # Assuming operation_type in excel corresponds to operation_type_id
            }, inplace=True)
            df_offers.rename(columns={"nom de l'offre": 'offer_name', 'offerid': 'offer_id'}, inplace=True)

            # --- 2. POPULATE LOOKUP TABLES ---

            # Customers
            if 'cust_num' in df_main.columns and 'cust_type' in df_main.columns:
                unique_customers = df_main[['cust_num', 'cust_type']].drop_duplicates().dropna()
                unique_customers.rename(columns={'cust_num': 'cust_num_bigint'}, inplace=True)
                if not unique_customers.empty:
                    result = conn.execute(text("""
                        INSERT IGNORE INTO customers (cust_num_bigint, cust_type)
                        VALUES (:cust_num_bigint, :cust_type)
                    """), unique_customers.to_dict(orient='records'))
                    inserted_counts['customers'] = result.rowcount

            # Object Types (from data in your DB schema) - Insert if they don't exist
            object_type_mapping = {'F': 'Free Resource', 'B': 'Main Balance', 'C': 'Credit Balance'}
            for code, desc in object_type_mapping.items():
                try:
                    conn.execute(text("""
                        INSERT IGNORE INTO object_types (object_type_code, description)
                        VALUES (:code, :desc)
                    """), {'code': code, 'desc': desc})
                    # Use a separate counter for object_types if you want to track ignored vs inserted
                    # For simplicity, we just count if an insert was attempted
                except Exception as e:
                    app.logger.warning(f"Could not insert object_type {code}: {e}. May already exist.")
            inserted_counts['object_types'] = len(object_type_mapping)  # This is a conceptual count, not actual inserts

            # Operation Types (from data in your DB schema) - Insert if they don't exist
            operation_type_mapping = {1: 'Purchase', 2: 'Transfer', 3: 'Refund', 4: 'Adjustment', 5: 'Loan',
                                      6: 'Expiration'}
            for id, desc in operation_type_mapping.items():
                try:
                    conn.execute(text("""
                        INSERT IGNORE INTO operation_types (operation_type_id, description)
                        VALUES (:id, :desc)
                    """), {'id': id, 'desc': desc})
                except Exception as e:
                    app.logger.warning(f"Could not insert operation_type {id}: {e}. May already exist.")
            inserted_counts['operation_types'] = len(operation_type_mapping)

            # Offers (from NOM DES OFFRES sheet)
            if not df_offers.empty:
                result = conn.execute(text("""
                    INSERT IGNORE INTO offers (offer_id, offer_name)
                    VALUES (:offer_id, :offer_name)
                """), df_offers.to_dict(orient='records'))
                inserted_counts['offers'] = result.rowcount

            # Objects
            object_types_db = pd.read_sql("SELECT object_type_id, object_type_code FROM object_types", conn)
            code_to_id_map = dict(zip(object_types_db['object_type_code'], object_types_db['object_type_id']))

            if 'object_type' in df_main.columns:
                df_main['object_type_id'] = df_main['object_type'].map(code_to_id_map)

                # Filter out rows where object_type_id could not be mapped
                df_main_filtered_objects = df_main.dropna(subset=['object_type_id', 'object_id']).copy()

                unique_objects = df_main_filtered_objects[
                    ['object_id', 'cust_num', 'object_type_id', 'current_amount_expire_time']
                ].drop_duplicates().dropna(subset=['object_id'])  # object_id must be present

                unique_objects.rename(columns={
                    'object_id': 'object_id_bigint',
                    'cust_num': 'cust_num_bigint',
                    'current_amount_expire_time': 'expiry_date'
                }, inplace=True)

                if not unique_objects.empty:
                    # Convert expiry_date to datetime, coercing errors to NaT, then handle NaT
                    unique_objects['expiry_date'] = pd.to_datetime(unique_objects['expiry_date'], errors='coerce')
                    # Replace NaT with None or a suitable default for database insertion
                    unique_objects['expiry_date'] = unique_objects['expiry_date'].apply(
                        lambda x: x.isoformat() if pd.notna(x) else None)

                    result = conn.execute(text("""
                        INSERT IGNORE INTO objects (object_id_bigint, cust_num_bigint, object_type_id, expiry_date)
                        VALUES (:object_id_bigint, :cust_num_bigint, :object_type_id, :expiry_date)
                    """), unique_objects.to_dict(orient='records'))
                    inserted_counts['objects'] = result.rowcount

            # --- 3. PREPARE AND INSERT MAIN DATA (cdr_events) ---

            cdr_columns = [
                'cdr_id', 'creation_time', 'cust_num', 'object_id', 'offer_id',
                'operation_type_id', 'deduct_amount', 'charge_amount',
                'current_amount_pre', 'current_amount_post',
                'charge_expire_time', 'current_amount_expire_time'
            ]
            df_cdr_to_insert = df_main[[col for col in cdr_columns if col in df_main.columns]].copy()

            for dt_col in ['creation_time', 'charge_expire_time', 'current_amount_expire_time']:
                if dt_col in df_cdr_to_insert.columns:
                    df_cdr_to_insert[dt_col] = pd.to_datetime(df_cdr_to_insert[dt_col], errors='coerce').where(
                        pd.notnull(df_cdr_to_insert[dt_col]), None)

            # Ensure integer columns are treated correctly, especially before to_sql
            int_cols = ['cdr_id', 'cust_num', 'object_id', 'offer_id', 'operation_type_id']
            for col in int_cols:
                if col in df_cdr_to_insert.columns:
                    # Convert to numeric, coerce errors to NaN, then fill NaN with a sentinel like -1 or drop
                    # For insertion, ensure None for NaN is handled by the database schema or `to_sql`
                    df_cdr_to_insert[col] = pd.to_numeric(df_cdr_to_insert[col], errors='coerce')
                    # Convert to Int64 (pandas nullable integer type) to allow NaN/None
                    df_cdr_to_insert[col] = df_cdr_to_insert[col].astype('Int64')

            # Handle float columns (amounts)
            float_cols = ['deduct_amount', 'charge_amount', 'current_amount_pre', 'current_amount_post']
            for col in float_cols:
                if col in df_cdr_to_insert.columns:
                    df_cdr_to_insert[col] = pd.to_numeric(df_cdr_to_insert[col], errors='coerce')
                    df_cdr_to_insert[col] = df_cdr_to_insert[col].fillna(0.0)  # Fill NaN amounts with 0.0

            if not df_cdr_to_insert.empty:
                # Use execute directly with a list of dictionaries for performance with sqlalchemy engine
                # Also, this avoids issues with `to_sql` and `if_exists='append'` when column types mismatch
                # if the DB schema is strict.
                # It's better to explicitly build the insert statement and parameters.

                # Filter out rows with NaN in critical columns if they are NOT NULL in DB
                df_cdr_to_insert.dropna(subset=['cdr_id', 'creation_time', 'cust_num', 'operation_type_id'],
                                        inplace=True)

                if not df_cdr_to_insert.empty:
                    # Prepare data for insertion (list of dicts)
                    cdr_records = df_cdr_to_insert.to_dict(orient='records')

                    # Create a prepared statement
                    insert_cdr_query = text("""
                        INSERT IGNORE INTO cdr_events (
                            cdr_id, creation_time, cust_num, object_id, offer_id,
                            operation_type_id, deduct_amount, charge_amount,
                            current_amount_pre, current_amount_post,
                            charge_expire_time, current_amount_expire_time
                        ) VALUES (
                            :cdr_id, :creation_time, :cust_num, :object_id, :offer_id,
                            :operation_type_id, :deduct_amount, :charge_amount,
                            :current_amount_pre, :current_amount_post,
                            :charge_expire_time, :current_amount_expire_time
                        )
                    """)

                    # Execute in chunks if the dataframe is very large
                    chunk_size = 5000
                    for i in range(0, len(cdr_records), chunk_size):
                        chunk = cdr_records[i:i + chunk_size]
                        conn.execute(insert_cdr_query, chunk)

                    inserted_counts['cdr_events'] = len(df_cdr_to_insert)

            trans.commit()
            app.logger.info(f"Admin {session['employee_email']} uploaded file. Details: {inserted_counts}")
            return jsonify({
                'message': 'File processed successfully!',
                'details': inserted_counts
            }), 200

        except Exception as e:
            trans.rollback()
            app.logger.error(f"Upload failed for admin {session['employee_email']}: {e}", exc_info=True)
            return jsonify({
                'error': f'A database error occurred during insertion. The process has been rolled back. Error: {e}'
            }), 500


@app.route('/admin/pending_users')
@admin_required
def pending_users():
    """Shows a list of users pending activation."""
    query = text("SELECT employee_id, employee_gmail FROM employees WHERE is_active = FALSE")
    engine = get_db_engine()
    if not engine:
        flash("Database connection not available.", "danger")
        return redirect(url_for('dashboard'))  # Or render an error page
    with engine.connect() as conn:
        users = conn.execute(query).mappings().fetchall()
    return render_template("liste_employes.html", users=users, user_role=session.get('role_name'))


@app.route('/admin/activate/<int:employee_id>', methods=['POST'])
@admin_required
def activate_employee(employee_id):
    """Activates a user account and sends a notification email."""
    engine = get_db_engine()
    if not engine:
        flash("Database connection not available.", "danger")
        return redirect(url_for('pending_users'))

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            email_query = text("SELECT employee_gmail FROM employees WHERE employee_id = :id")
            email = conn.execute(email_query, {'id': employee_id}).scalar_one_or_none()
            if not email:
                flash("User not found.", "danger")
                return redirect(url_for('pending_users'))

            update_query = text("UPDATE employees SET is_active = TRUE WHERE employee_id = :id")
            conn.execute(update_query, {'id': employee_id})
            trans.commit()

            email_body = "<p>Hello,</p><p>Your account for the Mattel Analytics platform has been activated. You can now log in.</p>"
            send_email(to_email=email, subject="Your Mattel Analytics Account is Active", body_html=email_body)
            flash(f"User {email} has been activated successfully.", "success")
        except Exception as e:
            trans.rollback()
            app.logger.error(f"Error activating user {employee_id}: {e}", exc_info=True)
            flash("An error occurred while activating the account.", "danger")
    return redirect(url_for('pending_users'))


@app.route('/admin/reset_analysis', methods=['POST'])
@admin_required
def reset_analysis():
    """DESTRUCTIVE: Truncates the cdr_events table. Use with extreme caution."""
    engine = get_db_engine()
    if not engine:
        return jsonify({'error': 'Database connection not available.'}), 500
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            conn.execute(text("TRUNCATE TABLE cdr_events"))
            trans.commit()
        app.logger.warning(f"Admin {session['employee_email']} reset the analysis data (truncated cdr_events).")
        return jsonify({'message': 'Analysis data has been successfully reset.'}), 200
    except Exception as e:
        app.logger.error(f"Error resetting analysis data: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# 7. API ENDPOINTS (Grouped by Category)
# ==============================================================================

# --- API Group: System & Overall Stats ---

@app.route('/api/health')
def health_check():
    """Checks API and database health."""
    engine = get_db_engine()
    if not engine:
        return jsonify({'status': 'unhealthy', 'database_error': 'Engine not initialized'}), 500
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))  # Simple query to check connection
            return jsonify({'status': 'healthy', 'database': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'database_error': str(e)}), 500


@app.route('/api/kpi_summary')
@login_required
@cache.cached(timeout=cache_timeout)
def kpi_summary():
    """Provides a summary of key performance indicators for the main dashboard."""
    query = text("""
        SELECT
            (SELECT COUNT(*) FROM customers) AS total_customers,
            (SELECT COUNT(*) FROM cdr_events) AS number_of_operations,
            (SELECT COUNT(DISTINCT cust_num) FROM cdr_events WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)) AS active_customers,
            IFNULL(SUM(CASE WHEN operation_type_id IN (1, 2, 3) THEN deduct_amount ELSE 0 END), 0) AS total_revenue,
            IFNULL(SUM(charge_amount), 0) AS total_charges
        FROM cdr_events LIMIT 1
    """)  # LIMIT 1 is a trick to make it return a single row, even if cdr_events is empty
    df = query_to_dataframe(query)

    # Ensure df is not empty before accessing data
    if df.empty or df.iloc[0].isnull().all():
        # Return default zero values if no data or all nulls
        summary = {
            'total_revenue': 0.0,
            'total_charges': 0.0,
            'total_deducted': 0.0,
            'number_of_operations': 0,
            'active_customers': 0,
            'arpu': 0.0
        }
        return jsonify(summary)

    stats = df.iloc[0]
    total_customers = int(stats['total_customers']) if pd.notna(stats['total_customers']) else 0
    total_revenue = float(stats['total_revenue']) if pd.notna(stats['total_revenue']) else 0.0
    arpu = total_revenue / total_customers if total_customers > 0 else 0.0

    summary = {
        'total_revenue': total_revenue,
        'total_charges': float(stats['total_charges']) if pd.notna(stats['total_charges']) else 0.0,
        'total_deducted': total_revenue,  # Assuming deduct_amount IS revenue
        'number_of_operations': int(stats['number_of_operations']) if pd.notna(stats['number_of_operations']) else 0,
        'active_customers': int(stats['active_customers']) if pd.notna(stats['active_customers']) else 0,
        'arpu': arpu
    }
    return jsonify(summary)


@app.route('/api/overall_stats')
@login_required
@cache.cached(timeout=cache_timeout)
def overall_stats():
    """
    API endpoint for overall key statistics and conclusions about the entire system.
    This endpoint gathers various high-level metrics and generates textual insights.
    """
    stats = {}

    total_customers_query = "SELECT COUNT(*) as total_customers FROM customers"
    df_total_customers = query_to_dataframe(total_customers_query)
    total_customers = 0
    if not df_total_customers.empty and pd.notna(df_total_customers['total_customers'].iloc[0]):
        total_customers = int(df_total_customers['total_customers'].iloc[0])
    stats['total_customers'] = total_customers

    total_cdr_events_query = "SELECT COUNT(*) as total_events FROM cdr_events"
    df_total_events = query_to_dataframe(total_cdr_events_query)
    total_cdr_events = 0
    if not df_total_events.empty and pd.notna(df_total_events['total_events'].iloc[0]):
        total_cdr_events = int(df_total_events['total_events'].iloc[0])
    stats['total_cdr_events'] = total_cdr_events

    total_revenue_query = """
    SELECT IFNULL(SUM(deduct_amount), 0) as total_revenue
    FROM cdr_events
    WHERE operation_type_id IN (1, 2, 3)
    """
    df_total_revenue = query_to_dataframe(total_revenue_query)
    total_revenue = 0.0
    if not df_total_revenue.empty and pd.notna(df_total_revenue['total_revenue'].iloc[0]):
        total_revenue = float(df_total_revenue['total_revenue'].iloc[0])
    stats['total_revenue'] = total_revenue

    total_charges_query = "SELECT IFNULL(SUM(charge_amount), 0) as total_charges FROM cdr_events"
    df_total_charges = query_to_dataframe(total_charges_query)
    total_charges = 0.0
    if not df_total_charges.empty and pd.notna(df_total_charges['total_charges'].iloc[0]):
        total_charges = float(df_total_charges['total_charges'].iloc[0])
    stats['total_charges'] = total_charges

    active_customers_year_query = """
    SELECT COUNT(DISTINCT cust_num) as active_customers_year
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
    """
    df_active_customers_year = query_to_dataframe(active_customers_year_query)
    active_customers_year = 0
    if not df_active_customers_year.empty and pd.notna(df_active_customers_year['active_customers_year'].iloc[0]):
        active_customers_year = int(df_active_customers_year['active_customers_year'].iloc[0])
    stats['active_customers_year'] = active_customers_year

    # Total Number of Offers: Count of all distinct offers defined in the 'offers' table
    total_offers_query = "SELECT COUNT(*) as total_offers FROM offers"
    df_total_offers = query_to_dataframe(total_offers_query)
    total_offers = 0
    if not df_total_offers.empty and pd.notna(df_total_offers['total_offers'].iloc[0]):
        total_offers = int(df_total_offers['total_offers'].iloc[0])
    stats['total_offers'] = total_offers

    # Average Revenue Per Customer: Total revenue divided by total customers
    stats['avg_revenue_per_customer'] = stats['total_revenue'] / stats['total_customers'] if stats[
                                                                                                 'total_customers'] > 0 else 0.0

    # Average Events Per Customer: Total CDR events divided by total customers
    stats['avg_events_per_customer'] = stats['total_cdr_events'] / stats['total_customers'] if stats[
                                                                                                   'total_customers'] > 0 else 0.0

    # Unique Offers with Activity: Count of distinct offers that have appeared in CDR events
    unique_active_offers_query = "SELECT COUNT(DISTINCT offer_id) as unique_active_offers FROM cdr_events WHERE offer_id IS NOT NULL"
    df_unique_active_offers = query_to_dataframe(unique_active_offers_query)
    unique_active_offers = 0
    if not df_unique_active_offers.empty and pd.notna(df_unique_active_offers['unique_active_offers'].iloc[0]):
        unique_active_offers = int(df_unique_active_offers['unique_active_offers'].iloc[0])
    stats['unique_active_offers'] = unique_active_offers

    # Unique Operation Types Used: Count of distinct operation types recorded in CDR events
    unique_operation_types_query = "SELECT COUNT(DISTINCT operation_type_id) as unique_operation_types FROM cdr_events WHERE operation_type_id IS NOT NULL"
    df_unique_operation_types = query_to_dataframe(unique_operation_types_query)
    unique_operation_types = 0
    if not df_unique_operation_types.empty and pd.notna(df_unique_operation_types['unique_operation_types'].iloc[0]):
        unique_operation_types = int(df_unique_operation_types['unique_operation_types'].iloc[0])
    stats['unique_operation_types'] = unique_operation_types

    # Unique Object Types Used: Count of distinct object types referenced in the 'objects' table
    unique_object_types_query = """
    SELECT COUNT(DISTINCT ot.object_type_id) as unique_object_types_used
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    """
    df_unique_object_types = query_to_dataframe(unique_object_types_query)
    unique_object_types_used = 0
    if not df_unique_object_types.empty and pd.notna(df_unique_object_types['unique_object_types_used'].iloc[0]):
        unique_object_types_used = int(df_unique_object_types['unique_object_types_used'].iloc[0])
    stats['unique_object_types_used'] = unique_object_types_used

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


# --- API Group: Customer Behavior & Segmentation ---
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
        COUNT(DISTINCT c.cust_num_bigint) as customer_count,
        COUNT(o.object_id_bigint) as object_count
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
    ),
    ranked_spending AS (
        SELECT
            total_spent,
            ROW_NUMBER() OVER (ORDER BY total_spent) as rn,
            COUNT(*) OVER () as total_customers_in_spending
        FROM customer_spending
    ),
    percentiles AS (
        SELECT
            MAX(CASE WHEN rn = CEIL(total_customers_in_spending * 0.8) THEN total_spent END) as p80,
            MAX(CASE WHEN rn = CEIL(total_customers_in_spending * 0.5) THEN total_spent END) as p50
        FROM ranked_spending
    )
    SELECT 
        CASE 
            WHEN cs.total_spent >= (SELECT p80 FROM percentiles) THEN 'High Value'
            WHEN cs.total_spent >= (SELECT p50 FROM percentiles) THEN 'Medium Value'
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
        IFNULL(AVG(ce.deduct_amount), 0) as avg_deduct_amount,
        IFNULL(AVG(ce.charge_amount), 0) as avg_charge_amount,
        IFNULL(SUM(ce.deduct_amount), 0) as total_revenue,
        COUNT(ce.cdr_event_id) as total_events,
        (COUNT(ce.cdr_event_id) / COUNT(DISTINCT ce.cust_num)) as avg_events_per_customer,
        IFNULL(AVG(ce.current_amount_post), 0) as avg_balance_after_transaction
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
            subplot_titles=('Total Revenue Comparison', 'Usage Frequency (Events/Customer)', 'Average Deduct Amount',
                            'Active Customer Count'),
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
        COUNT(DISTINCT cpp.cust_num) as customer_count,
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
        AVG(TIMESTAMPDIFF(HOUR, ce.creation_time, ce.current_amount_expire_time)) as avg_current_lifetime_hours,
        AVG(ce.charge_amount) as avg_charge_amount,
        AVG(ce.current_amount_post) as avg_final_balance,
        COUNT(*) as transaction_count
    FROM cdr_events ce
    JOIN customers c ON ce.cust_num = c.cust_num_bigint
    WHERE ce.charge_expire_time IS NOT NULL 
    AND ce.current_amount_expire_time IS NOT NULL
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


# --- API Group: Offer & Product Analysis ---
@app.route('/api/offers_summary')
@login_required
@cache.cached(timeout=cache_timeout)
def offers_summary():
    """
    API endpoint to provide a summary of offers and their usage.
    Calculates subscriber count, event count, average, and total deduction amounts per offer.
    Generates a bar chart showing the number of subscribers by offer.
    """
    query = text("""
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
    """)
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
    """
    query = text("""
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
    """)
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
                size=df['event_count'].apply(lambda x: max(x, 1)),  # Ensure size is at least 1 for visibility
                hover_name='offer_name',
                title='Offer Performance Analysis',
                labels={
                    'subscriber_count': 'Number of Subscribers',
                    'total_usage': 'Total Usage (Amount)',
                    'event_count': 'Number of Events'
                }
            )
            if len(df) < 20:  # Only add text labels if there aren't too many points
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
    query = text("""
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
    """)
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No offer trend data available.")
    else:
        # Filter for top N offers to keep the chart readable
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
    query = text("""
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ce.cust_num) as subscriber_count,
        IFNULL(AVG(ce.deduct_amount), 0) as avg_usage,
        IFNULL(AVG(TIMESTAMPDIFF(HOUR, ce.creation_time, ce.charge_expire_time)), 0) as avg_duration_hours
    FROM offers o
    LEFT JOIN cdr_events ce ON o.offer_id = ce.offer_id 
    WHERE o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Mo%' OR o.offer_name LIKE '%Internet%'
    GROUP BY o.offer_id, o.offer_name
    ORDER BY subscriber_count DESC
    """)
    df = query_to_dataframe(query)

    if df.empty or df['subscriber_count'].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title_text="No data bundle analysis data available.")
    else:
        fig = px.bar(
            df,
            x='offer_name',
            y='subscriber_count',
            color='avg_usage',  # Color by average usage to show usage intensity
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
    """
    query = text("""
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
    """)
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


# --- API Group: Financial & Revenue Analysis ---
@app.route('/api/revenue_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def revenue_analysis():
    """
    API endpoint for revenue analysis.
    Calculates monthly revenue broken down by offer, focusing on usage operation types.
    Generates a stacked bar chart for monthly revenue by offer.
    """
    query = text("""
    SELECT DATE_FORMAT(ce.creation_time, '%Y-%m') as month,
           o.offer_name,
           IFNULL(SUM(ce.deduct_amount), 0) as revenue
    FROM cdr_events ce
    LEFT JOIN offers o ON ce.offer_id = o.offer_id
    WHERE ce.operation_type_id IN (1, 2, 3) # Assuming 1,2,3 are usage-related operation types
    GROUP BY month, o.offer_name
    ORDER BY month, o.offer_name
    """)
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
    query = text("""
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
    """)
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
    query = text("""
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
    """)
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
    query = text("""
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
    """)
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
    """
    query = text("""
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
    """)
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


# --- API Group: Operational & Resource Management ---
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
    query = text("""
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
    """)
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
    query = text("""
    SELECT 
        ot.object_type_code,
        ot.description,
        COUNT(DISTINCT o.object_id_bigint) as object_count,
        COUNT(DISTINCT o.cust_num_bigint) as customer_count
    FROM object_types ot
    LEFT JOIN objects o ON ot.object_type_id = o.object_type_id
    GROUP BY ot.object_type_code, ot.description
    ORDER BY object_count DESC
    """)
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
    query = text("""
    SELECT 
        DATE(creation_time) as event_date,
        COUNT(*) as event_count,
        IFNULL(SUM(deduct_amount), 0) as total_deduct_amount,
        COUNT(DISTINCT cust_num) as active_customers
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 5 YEAR) 
    GROUP BY DATE(creation_time)
    ORDER BY event_date
    """)
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
    query = text("""
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
    """)
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
    query = text("""
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
    """)
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
    query = text("""
    WITH expiry_analysis AS (
        SELECT 
            ce.cust_num,
            SUM(CASE WHEN ce.charge_expire_time < NOW() AND ce.charge_amount > 0 THEN ce.charge_amount ELSE 0 END) as expired_charged_amount,
            SUM(CASE WHEN ce.current_amount_expire_time < NOW() AND ce.current_amount_post > 0 THEN ce.current_amount_post ELSE 0 END) as expired_current_amount,
            COUNT(CASE WHEN ce.charge_expire_time < NOW() AND ce.charge_amount > 0 THEN 1 END) as expired_charge_events,
            COUNT(CASE WHEN ce.current_amount_expire_time < NOW() AND ce.current_amount_post > 0 THEN 1 END) as expired_current_events,
            SUM(ce.charge_amount) as total_charged,
            SUM(ce.deduct_amount) as total_deducted
        FROM cdr_events ce
        WHERE ce.charge_expire_time IS NOT NULL OR ce.current_amount_expire_time IS NOT NULL
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
    """)
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No expiry impact data available.")
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Customer Distribution by Expiry Status', 'Total Expired Value Impact'),
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
    """
    query = text("""
    WITH hourly_usage AS (
        SELECT 
            DATE(creation_time) as usage_date,
            HOUR(creation_time) as usage_hour,
            COUNT(*) as event_count,
            COUNT(DISTINCT cust_num) as active_users,
            SUM(CASE WHEN o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Internet%' THEN 1 ELSE 0 END) as data_events,
            SUM(CASE WHEN o.offer_name LIKE '%Voice%' OR o.offer_name LIKE '%min%' THEN 1 ELSE 0 END) as voice_events
        FROM cdr_events ce
        LEFT JOIN offers o ON ce.offer_id = o.offer_id -- Use LEFT JOIN to include events without offers
        WHERE ce.creation_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
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
    """)
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
                COUNT(ce.cdr_event_id) as total_records,
                COUNT(CASE WHEN ce.offer_id IS NOT NULL AND o.offer_id IS NULL THEN 1 END) as inconsistent_records_no_match,
                COUNT(CASE WHEN ce.offer_id IS NOT NULL AND o.offer_id IS NOT NULL AND ce.offer_id != o.offer_id THEN 1 END) as inconsistent_records_mismatch,
                (COUNT(CASE WHEN ce.offer_id IS NOT NULL AND o.offer_id IS NULL THEN 1 END) + 
                 COUNT(CASE WHEN ce.offer_id IS NOT NULL AND o.offer_id IS NOT NULL AND ce.offer_id != o.offer_id THEN 1 END)) / COUNT(ce.cdr_event_id) * 100 as inconsistency_rate
            FROM cdr_events ce
            LEFT JOIN offers o ON ce.offer_id = o.offer_id
        """,

        'amount_anomalies': """
            SELECT 
                'Amount Anomalies' as check_type,
                COUNT(*) as total_records,
                COUNT(CASE WHEN deduct_amount < 0 THEN 1 END) as negative_deductions,
                COUNT(CASE WHEN charge_amount < 0 THEN 1 END) as negative_charges,
                COUNT(CASE WHEN deduct_amount > (SELECT AVG(deduct_amount) * 10 FROM cdr_events) THEN 1 END) as high_deductions,
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
        # Ensure consistent structure for results, even if a query returns empty
        if not df.empty:
            row_dict = df.iloc[0].to_dict()
            # Fill missing keys with default values if they are not in the query result
            for key in ['total_records', 'inconsistent_records_no_match', 'inconsistent_records_mismatch',
                        'negative_deductions', 'negative_charges', 'high_deductions', 'avg_deduct', 'stddev_deduct',
                        'future_timestamps', 'very_old_timestamps', 'invalid_expiry_times',
                        'customers_in_events', 'customers_in_master', 'orphaned_customers', 'inconsistency_rate']:
                row_dict.setdefault(key, 0 if key not in ['avg_deduct', 'stddev_deduct'] else None)
            results.append(row_dict)
        else:
            # Append a default structure for empty results to keep table consistent
            results.append({
                'check_type': check_name.replace('_', ' ').title(),
                'total_records': 0,
                'inconsistent_records_no_match': 0,
                'inconsistent_records_mismatch': 0,
                'inconsistency_rate': 0.0,
                'negative_deductions': 0,
                'negative_charges': 0,
                'high_deductions': 0,
                'avg_deduct': None,
                'stddev_deduct': None,
                'future_timestamps': 0,
                'very_old_timestamps': 0,
                'invalid_expiry_times': 0,
                'customers_in_events': 0,
                'customers_in_master': 0,
                'orphaned_customers': 0
            })

    if not results:
        fig = go.Figure()
        fig.update_layout(title_text="No data quality report available.")
    else:
        # Ensure all dictionaries in results have the same keys for table rendering
        all_keys = sorted(list(set(k for d in results for k in d.keys())))
        table_header_values = [k.replace('_', ' ').title() for k in all_keys]
        # Use .get with default 'N/A' for cell values, converting floats to formatted strings
        table_cell_values = [[(f"{row.get(k, 'N/A'):.2f}" if isinstance(row.get(k), (float)) and k not in [
            'inconsistency_rate'] else row.get(k, 'N/A')) for k in all_keys] for row in results]

        fig = go.Figure(data=[
            go.Table(
                header=dict(values=table_header_values,
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=table_cell_values,
                           fill_color='lavender',
                           align='left'))
        ])
        fig.update_layout(title_text="Data Quality Report Summary")

    return jsonify({
        'data': results,
        'chart': create_figure_json(fig)
    })


# --- API Group: Predictive Analysis & Recommendation ---
@app.route('/api/churn_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def churn_analysis():
    """
    API endpoint for churn analysis.
    Identifies churned customers (inactive for > 90 days) and aggregates
    churn count by quarter. Also provides a breakdown of churned customers by offer.
    """
    query = text("""
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
    """)
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

    offers_churn_query = text("""
    WITH customer_last_offer AS (
        SELECT 
            ce.cust_num,
            ce.offer_id,
            MAX(ce.creation_time) as last_activity_date
        FROM cdr_events ce
        GROUP BY ce.cust_num, ce.offer_id
    ),
    churned_cust_offer AS ( -- Renamed to avoid conflict
        SELECT 
            clo.cust_num,
            clo.offer_id
        FROM customer_last_offer clo
        WHERE DATEDIFF(NOW(), clo.last_activity_date) > 90
    )
    SELECT 
        o.offer_id,
        o.offer_name,
        COUNT(DISTINCT ch.cust_num) as churned_customers
    FROM churned_cust_offer ch -- Using the renamed CTE
    JOIN offers o ON ch.offer_id = o.offer_id
    GROUP BY o.offer_id, o.offer_name
    ORDER BY churned_customers DESC
    """)
    offers_churn_df = query_to_dataframe(offers_churn_query)

    return jsonify({
        'data': offers_churn_df.to_dict('records'),  # Data for the table
        'chart': create_figure_json(fig),  # Chart for churn over time
        'offers_churn': offers_churn_df.to_dict('records')
        # Separate data for offers churn breakdown (can be table/list on frontend)
    })


@app.route('/api/churn_risk_analysis')
@login_required
@cache.cached(timeout=cache_timeout)
def churn_risk_analysis():
    """
    Identify customers at risk of churning based on usage patterns and activity.
    """
    query = text("""
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
    """)
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
    """
    query = text("""
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
    """)
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
    """
    query = text("""
    WITH customer_offer_matrix AS (
        SELECT
            ce.cust_num,
            o.offer_name,
            COUNT(*) as usage_frequency,
            SUM(ce.deduct_amount) as total_spent,
            MAX(ce.creation_time) as last_used
        FROM cdr_events ce
        JOIN offers o ON ce.offer_id = o.offer_id
        GROUP BY ce.cust_num, o.offer_name
    ),
    customer_segments AS (
        SELECT
            cust_num,
            offer_name as primary_offer,
            CASE
                WHEN offer_name LIKE '%Data%' OR offer_name LIKE '%Internet%' OR offer_name LIKE '%Mo%' THEN 'Data Users'
                WHEN offer_name LIKE '%Voice%' OR offer_name LIKE '%min%' OR offer_name LIKE '%SMS%' THEN 'Voice-centric users'
                WHEN offer_name LIKE '%Loan%' THEN 'Loan Users'
                ELSE 'Mixed Users'
            END as user_segment
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
            SUM(ce.deduct_amount) as total_revenue,
            ROW_NUMBER() OVER (PARTITION BY cs.user_segment ORDER BY SUM(ce.deduct_amount) DESC) as recommendation_rank_calc
        FROM customer_segments cs
        JOIN cdr_events ce ON ce.cust_num = cs.cust_num
        JOIN offers o ON ce.offer_id = o.offer_id
        WHERE CASE
            WHEN cs.user_segment = 'Data Users' THEN o.offer_name LIKE '%Data%' OR o.offer_name LIKE '%Internet%'
            WHEN cs.user_segment = 'Voice-centric users' THEN o.offer_name LIKE '%Voice%' OR o.offer_name LIKE '%min%' OR o.offer_name LIKE '%SMS%'
            WHEN cs.user_segment = 'Loan Users' THEN o.offer_name LIKE '%Loan%'
            ELSE TRUE
        END
        AND o.offer_name != cs.primary_offer -- Exclude the offer that defined their primary segment
        GROUP BY cs.user_segment, o.offer_name
    )
    SELECT
        user_segment,
        recommended_offer,
        users_count,
        avg_revenue_per_use,
        total_revenue,
        recommendation_rank_calc as recommendation_rank
    FROM recommendations
    WHERE recommendation_rank_calc <= 3
    ORDER BY user_segment, recommendation_rank
    """)
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


# --- API Group: Detailed Entity Lookups ---
@app.route('/api/customer_details/<int:cust_num>')
@login_required
@cache.cached(timeout=cache_timeout)
def customer_details(cust_num):
    """
    API endpoint for detailed information about a specific customer.
    Fetches customer's basic info, associated objects/balances, and recent CDR events.
    This is a lookup endpoint, returning raw data, not a chart.
    """
    customer_query = text("""
    SELECT cust_num_bigint, cust_type FROM customers WHERE cust_num_bigint = :cust_num
    """)
    customer_df = query_to_dataframe(customer_query, params={'cust_num': cust_num})
    if customer_df.empty:
        return jsonify({'error': 'Customer not found'}), 404

    objects_query = text("""
    SELECT 
        o.object_id_bigint, 
        ot.description as object_type_description, 
        o.expiry_date
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    WHERE o.cust_num_bigint = :cust_num
    ORDER BY o.expiry_date DESC
    """)
    objects_df = query_to_dataframe(objects_query, params={'cust_num': cust_num})

    cdr_events_query = text("""
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
    WHERE ce.cust_num = :cust_num
    ORDER BY ce.creation_time DESC
    LIMIT 100
    """)
    cdr_events_df = query_to_dataframe(cdr_events_query, params={'cust_num': cust_num})

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
    object_query = text("""
    SELECT 
        o.object_id_bigint, 
        o.cust_num_bigint, 
        ot.description as object_type_description, 
        o.expiry_date
    FROM objects o
    JOIN object_types ot ON o.object_type_id = ot.object_type_id
    WHERE o.object_id_bigint = :object_id
    """)
    object_df = query_to_dataframe(object_query, params={'object_id': object_id})
    if object_df.empty:
        return jsonify({'error': 'Object not found'}), 404

    cdr_events_query = text("""
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
    WHERE ce.object_id = :object_id
    ORDER BY ce.creation_time DESC
    LIMIT 100
    """)
    cdr_events_df = query_to_dataframe(cdr_events_query, params={'object_id': object_id})

    return jsonify({
        'object_info': object_df.iloc[0].to_dict(),
        'cdr_events': cdr_events_df.to_dict('records')
    })


# --- API Group: Real-time Monitoring & System Health ---
@app.route('/api/realtime_dashboard')
@login_required
@cache.cached(timeout=60)  # Shorter cache for real-time data
def realtime_dashboard():
    """
    Real-time dashboard showing current system activity and key metrics.
    """
    query = text("""
    SELECT 
        'current_hour' as metric_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT cust_num) as active_users,
        SUM(IFNULL(deduct_amount, 0)) as total_revenue,
        AVG(IFNULL(deduct_amount, 0)) as avg_transaction_value
    FROM cdr_events
    WHERE creation_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

    UNION ALL

    SELECT 
        'today' as metric_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT cust_num) as active_users,
        SUM(IFNULL(deduct_amount, 0)) as total_revenue,
        AVG(IFNULL(deduct_amount, 0)) as avg_transaction_value
    FROM cdr_events
    WHERE DATE(creation_time) = CURDATE()

    UNION ALL

    SELECT 
        'yesterday' as metric_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT cust_num) as active_users,
        SUM(IFNULL(deduct_amount, 0)) as total_revenue,
        AVG(IFNULL(deduct_amount, 0)) as avg_transaction_value
    FROM cdr_events
    WHERE DATE(creation_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
    """)
    df = query_to_dataframe(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No real-time data available.")
    else:
        # Re-index df to ensure order for plotting consistency
        df_metrics = df.set_index('metric_type')
        # Ensure all expected rows exist, filling with zeros if no data for that period
        expected_metrics = ['current_hour', 'today', 'yesterday']
        for metric in expected_metrics:
            if metric not in df_metrics.index:
                df_metrics.loc[metric] = [0, 0, 0, 0]  # event_count, active_users, total_revenue, avg_transaction_value

        df_metrics = df_metrics.reindex(expected_metrics)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Events', 'Users', 'Revenue', 'Avg Transaction'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=df_metrics.index, y=df_metrics['event_count'], name='Events'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df_metrics.index, y=df_metrics['active_users'], name='Users'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=df_metrics.index, y=df_metrics['total_revenue'], name='Revenue'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=df_metrics.index, y=df_metrics['avg_transaction_value'], name='Avg Transaction'),
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
    query = text("""
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
    """)
    df = query_to_dataframe(query)

    health_status = "Healthy"
    if not df.empty:
        freshness_row = df[df['health_metric'] == 'data_freshness']
        if not freshness_row.empty and pd.notna(freshness_row.iloc[0]['minutes_since_last_record']) and \
                freshness_row.iloc[0]['minutes_since_last_record'] > 60:
            health_status = "Warning - Data not fresh (last record over 60 mins ago)"

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
    """
    query = text("""
    WITH offer_performance AS (
        SELECT 
            o.offer_name,
            COUNT(DISTINCT ce.cust_num) as unique_users,
            COUNT(*) as total_usage_events, -- Renamed to avoid confusion with total_usage_amount
            SUM(IFNULL(ce.deduct_amount, 0)) as total_revenue,
            AVG(IFNULL(ce.deduct_amount, 0)) as avg_revenue_per_use,
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
    """)
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


# ==============================================================================
# 8. APPLICATION RUNNER
# ==============================================================================
if __name__ == '__main__':
    # Initialize the database engine once at startup
    get_db_engine()

    # For production, it's recommended to use a WSGI server like Gunicorn or uWSGI
    # instead of Flask's built-in development server.
    app.run(debug=True, host='0.0.0.0', port=5000)
