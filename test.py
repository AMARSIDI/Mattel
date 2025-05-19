from flask import Flask, render_template, jsonify
import pandas as pd
from sqlalchemy import create_engine, text
import json
from datetime import datetime, timedelta
import calendar

# === Configuration ===
DB_USER = 'root'
DB_PASS = ''
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'mattel'

# Create SQLAlchemy engine
engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
    echo=False
)

app = Flask(__name__)


# Helper function to execute SQL queries and return results as JSON
def execute_query(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params if params else {})
        return [dict(row) for row in result.mappings().all()]


# Handle datetime serialization for JSON responses
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S') # Or '%Y-%m-%d' for dates
        return super().default(obj)


app.json_encoder = CustomJSONEncoder


# --- Dashboard Route ---
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# --- API Routes for Dashboard Data ---

@app.route("/api/customer_counts")
def customer_counts():
    query = """
    SELECT
        cust_type,
        COUNT(*) as count
    FROM customers
    GROUP BY cust_type
    """
    return jsonify(execute_query(query))


@app.route("/api/objects_by_type")
def objects_by_type():
    query = """
    SELECT
        ot.object_type_code,
        ot.description,
        COUNT(o.object_id_bigint) as count
    FROM
        objects o
    JOIN
        object_types ot ON o.object_type_id = ot.object_type_id
    GROUP BY
        ot.object_type_code, ot.description
    """
    return jsonify(execute_query(query))


@app.route("/api/operations_summary")
def operations_summary():
    query = """
    SELECT
        ot.operation_type_id,
        ot.description,
        COUNT(ce.cdr_event_id) as count,
        SUM(COALESCE(ce.deduct_amount, 0)) as total_deduct,
        SUM(COALESCE(ce.charge_amount, 0)) as total_charge
    FROM
        cdr_events ce
    JOIN
        operation_types ot ON ce.operation_type_id = ot.operation_type_id
    GROUP BY
        ot.operation_type_id, ot.description
    ORDER BY
        count DESC
    LIMIT 10
    """
    return jsonify(execute_query(query))


@app.route("/api/offers_distribution")
def offers_distribution():
    query = """
    SELECT
        o.offer_id,
        o.offer_name,
        COUNT(ce.cdr_event_id) as usage_count
    FROM
        cdr_events ce
    JOIN
        offers o ON ce.offer_id = o.offer_id
    GROUP BY
        o.offer_id, o.offer_name
    ORDER BY
        usage_count DESC
    LIMIT 10
    """
    return jsonify(execute_query(query))


@app.route("/api/daily_activity")
def daily_activity():
    # MODIFIED: Removed the 30-day filter
    query = """
    SELECT
        DATE(ce.creation_time) as date,
        COUNT(*) as transaction_count,
        COUNT(DISTINCT ce.cust_num) as active_customers,
        SUM(CASE WHEN ce.deduct_amount > 0 THEN 1 ELSE 0 END) as consumption_events,
        SUM(CASE WHEN ce.charge_amount > 0 THEN 1 ELSE 0 END) as charging_events
    FROM
        cdr_events ce
    GROUP BY
        DATE(ce.creation_time)
    ORDER BY
        date
    """
    return jsonify(execute_query(query))


@app.route("/api/customer_balance_stats")
def customer_balance_stats():
    query = """
    SELECT
        AVG(o.current_amount) as avg_balance,
        MIN(o.current_amount) as min_balance,
        MAX(o.current_amount) as max_balance,
        STDDEV(o.current_amount) as stddev_balance,
        COUNT(DISTINCT o.cust_num_bigint) as customer_count
    FROM
        objects o
    JOIN
        object_types ot ON o.object_type_id = ot.object_type_id
    WHERE
        ot.object_type_code = 'B'  -- Main Balance
    """
    return jsonify(execute_query(query))


@app.route("/api/top_customers")
def top_customers():
    query = """
    SELECT
        ce.cust_num,
        COUNT(ce.cdr_event_id) as transaction_count,
        SUM(COALESCE(ce.deduct_amount, 0)) as total_consumption,
        SUM(COALESCE(ce.charge_amount, 0)) as total_recharge
    FROM
        cdr_events ce
    GROUP BY
        ce.cust_num
    ORDER BY
        transaction_count DESC
    LIMIT 10
    """
    return jsonify(execute_query(query))


@app.route("/api/resource_expiry_forecast")
def resource_expiry_forecast():
    # MODIFIED: Removed the 30-day filter for expiry forecast,
    # now shows all future expiries (change if you only want upcoming)
    query = """
    SELECT
        DATE(o.expiry_date) as date,
        COUNT(*) as expiring_objects,
        COUNT(DISTINCT o.cust_num_bigint) as affected_customers
    FROM
        objects o
    WHERE
        o.expiry_date > NOW()
    GROUP BY
        DATE(o.expiry_date)
    ORDER BY
        date
    """
    return jsonify(execute_query(query))


@app.route("/api/hourly_usage_pattern")
def hourly_usage_pattern():
    # MODIFIED: Removed the 30-day filter
    query = """
    SELECT
        HOUR(ce.creation_time) as hour_of_day,
        COUNT(*) as event_count,
        SUM(COALESCE(ce.deduct_amount, 0)) as total_consumption
    FROM
        cdr_events ce
    GROUP BY
        HOUR(ce.creation_time)
    ORDER BY
        hour_of_day
    """
    return jsonify(execute_query(query))


@app.route("/api/kpis")
def kpis():
    # MODIFIED: Removed 30-day filter for Active Customers for KPI
    # If "Active Customers" specifically means last 30 days, you might want to revert this one KPI's filter.
    # Otherwise, it now shows all active customers over all time.
    active_customers_query = """
    SELECT COUNT(DISTINCT cust_num) as active_customers
    FROM cdr_events
    """

    # Total Objects (already all time)
    total_objects_query = """
    SELECT COUNT(*) as total_objects
    FROM objects
    """

    # Total Transactions (already all time)
    total_transactions_query = """
    SELECT COUNT(*) as total_transactions
    FROM cdr_events
    """

    # Average Transaction Value (already all time for charges > 0)
    avg_transaction_query = """
    SELECT AVG(COALESCE(charge_amount, 0)) as avg_charge
    FROM cdr_events
    WHERE charge_amount > 0
    """

    # Combine all KPIs
    active_customers = execute_query(active_customers_query)[0]['active_customers']
    total_objects = execute_query(total_objects_query)[0]['total_objects']
    total_transactions = execute_query(total_transactions_query)[0]['total_transactions']
    avg_transaction = execute_query(avg_transaction_query)[0]['avg_charge']

    return jsonify({
        'active_customers': active_customers,
        'total_objects': total_objects,
        'total_transactions': total_transactions,
        'avg_transaction': avg_transaction
    })


if __name__ == "__main__":
    app.run(debug=True)