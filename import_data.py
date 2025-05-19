import pandas as pd
from sqlalchemy import create_engine, text

# === Configuration ===
DB_USER = 'root'
DB_PASS = ''
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'mattel'
EXCEL_FILE = 'DONNEES_MATTEL.xlsx'

# Create SQLAlchemy engine
engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
    echo=False
)

# === 1) Load Excel sheets ===
df = pd.read_excel(EXCEL_FILE, sheet_name='DONNEES ')
df_offers = pd.read_excel(EXCEL_FILE, sheet_name='NOM DES OFFRES ')
print(f"Loaded {len(df)} records from main sheet and {len(df_offers)} records from offers sheet.")

with engine.begin() as conn:
    # -------------------------------------------------------
    # 2) Prepare object_types mapping (code → id)
    # -------------------------------------------------------
    type_rows = conn.execute(text("SELECT object_type_id, object_type_code FROM object_types"))
    type_map = {row['object_type_code']: row['object_type_id'] for row in type_rows.mappings().all()}
    print(f"Loaded {len(type_map)} object type mappings.")

    # -------------------------------------------------------
    # 3) Insert customers (INSERT IGNORE)
    # -------------------------------------------------------
    cust = (
        df[['CUST_NUM', 'CUST_NUM.1', 'CUST_TYPE']]
        .drop_duplicates()
        .rename(columns={
            'CUST_NUM': 'cust_num_bigint',
            'CUST_NUM.1': 'cust_short_int',
            'CUST_TYPE': 'cust_type'
        })
    )
    cust[['registration_date', 'status', 'last_activity_date']] = pd.NaT
    if not cust.empty:
        recs = cust.to_dict('records')
        result = conn.execute(text("""
            INSERT IGNORE INTO customers
              (cust_num_bigint, cust_short_int, cust_type, registration_date, status, last_activity_date)
            VALUES
              (:cust_num_bigint, :cust_short_int, :cust_type, :registration_date, :status, :last_activity_date)
        """), recs)
        print(f"Inserted {result.rowcount} new customers.")

    # -------------------------------------------------------
    # 4) Insert object_types if new codes appear in Excel
    # -------------------------------------------------------
    excel_codes = df['OBJECT_type'].dropna().unique()
    new_codes = [c for c in excel_codes if c not in type_map]
    if new_codes:
        # Insert with NULL code placeholders
        result = conn.execute(text("""
            INSERT INTO object_types (object_type_code, description)
            VALUES (:code, NULL)
        """), [{"code": c} for c in new_codes])
        print(f"Inserted {result.rowcount} new object types.")

        # Refresh map
        rows = conn.execute(text("SELECT object_type_id, object_type_code FROM object_types"))
        type_map = {r['object_type_code']: r['object_type_id'] for r in rows.mappings().all()}

    # -------------------------------------------------------
    # 5) Insert offers
    # -------------------------------------------------------
    offer_rows = conn.execute(text("SELECT offer_id FROM offers"))
    existing_offers = {r[0] for r in offer_rows}
    offers = df_offers.rename(columns={'OFfERID': 'offer_id', "Nom de l'offre": 'offer_name'})
    new_offers = offers[~offers['offer_id'].isin(existing_offers)]
    if not new_offers.empty:
        new_offers = new_offers.assign(
            description=None, price=None, validity_days=None, service_type=None,
            allowance_data_mb=None, allowance_voice_min=None, allowance_sms_count=None,
            allowance_duration_hours=None, is_unlimited_data=False, is_unlimited_voice=False,
            loan_amount=None, offer_category=None, start_date=pd.NaT, end_date=pd.NaT,
            is_recurring=False
        )
        new_offers.to_sql('offers', conn, if_exists='append', index=False)
        print(f"Inserted {len(new_offers)} new offers.")

    # -------------------------------------------------------
    # 6) Insert operation_types
    # -------------------------------------------------------
    op_rows = conn.execute(text("SELECT operation_type_id FROM operation_types"))
    existing_ops = {r[0] for r in op_rows}
    ops = (
        df[['OPERATION_TYPE']]
        .drop_duplicates()
        .rename(columns={'OPERATION_TYPE': 'operation_type_id'})
    )
    new_ops = ops[~ops['operation_type_id'].isin(existing_ops)]
    if not new_ops.empty:
        new_ops['description'] = None
        new_ops['action_category'] = None
        new_ops.to_sql('operation_types', conn, if_exists='append', index=False)
        print(f"Inserted {len(new_ops)} new operation types.")

    # -------------------------------------------------------
    # 7) Insert objects, mapping code → id
    # -------------------------------------------------------
    # First, get all existing object_id_bigint values
    obj_rows = conn.execute(text("SELECT object_id_bigint FROM objects"))
    existing_objs = {r[0] for r in obj_rows}
    print(f"Found {len(existing_objs)} existing objects in the database.")

    # Prepare objects dataframe
    objs = (
        df[['OBJECT_ID', 'CUST_NUM', 'OBJECT_type', 'CURENT_AMOUNT_EXPIRE_TIME']]
        .drop_duplicates()
        .rename(columns={
            'OBJECT_ID': 'object_id_bigint',
            'CUST_NUM': 'cust_num_bigint',
            'OBJECT_type': 'object_type_code',
            'CURENT_AMOUNT_EXPIRE_TIME': 'expiry_date'
        })
    )

    # Handle expiry_date format - convert from numeric to datetime if needed
    if objs['expiry_date'].dtype == 'int64' or objs['expiry_date'].dtype == 'float64':
        objs['expiry_date'] = pd.to_datetime(
            objs['expiry_date'].astype(str), format='%Y%m%d%H%M%S', errors='coerce'
        )

    # Map to numeric IDs
    objs['object_type_id'] = objs['object_type_code'].map(type_map)
    # Drop rows with no valid FK
    objs = objs.dropna(subset=['object_type_id'])
    objs['object_type_id'] = objs['object_type_id'].astype(int)

    # Filter out already-existing objects
    new_objs = objs[~objs['object_id_bigint'].isin(existing_objs)]
    print(f"Found {len(new_objs)} new objects to insert.")

    # Process in batches to avoid memory/performance issues
    if not new_objs.empty:
        new_objs = new_objs.assign(
            creation_date=pd.NaT, status=None, initial_amount=None, current_amount=None
        )

        batch_size = 1000
        total_inserted = 0

        for start_idx in range(0, len(new_objs), batch_size):
            end_idx = min(start_idx + batch_size, len(new_objs))
            batch = new_objs.iloc[start_idx:end_idx]

            # Use parameterized query instead of to_sql for more control
            values = []
            for _, row in batch.iterrows():
                values.append({
                    'object_id_bigint': row['object_id_bigint'],
                    'cust_num_bigint': row['cust_num_bigint'],
                    'object_type_id': row['object_type_id'],
                    'expiry_date': row['expiry_date'],
                    'creation_date': row['creation_date'],
                    'status': row['status'],
                    'initial_amount': row['initial_amount'],
                    'current_amount': row['current_amount']
                })

            try:
                result = conn.execute(text("""
                    INSERT IGNORE INTO objects 
                    (object_id_bigint, cust_num_bigint, object_type_id, 
                     expiry_date, creation_date, status, initial_amount, current_amount)
                    VALUES 
                    (:object_id_bigint, :cust_num_bigint, :object_type_id, 
                     :expiry_date, :creation_date, :status, :initial_amount, :current_amount)
                """), values)
                total_inserted += result.rowcount
                print(f"Inserted batch {start_idx}-{end_idx}: {result.rowcount} rows")
            except Exception as e:
                print(f"Error inserting batch {start_idx}-{end_idx}: {e}")

        print(f"Total objects inserted: {total_inserted} out of {len(new_objs)} attempted")

    # -------------------------------------------------------
    # 8) Insert cdr_events
    # -------------------------------------------------------
    # First, get existing cdr_ids
    cdr_rows = conn.execute(text("SELECT cdr_id FROM cdr_events"))
    existing_cdrs = {r[0] for r in cdr_rows}
    print(f"Found {len(existing_cdrs)} existing CDR events in the database.")

    # Prepare events dataframe
    ev = df.rename(columns={
        'CDR_ID': 'cdr_id', 'CREATION_TIME': 'creation_time',
        'CUST_NUM': 'cust_num', 'OBJECT_ID': 'object_id',
        'OPERATION_TYPE': 'operation_type_id', 'OFFER_ID': 'offer_id',
        'DEDUCT_AMOUNT': 'deduct_amount', 'CHARGE_AMOUNT': 'charge_amount',
        'CURENT_AMOUNT': 'current_amount_pre',
        'CURENT_AMOUNT.2': 'current_amount_post',
        'TOTAL_AMOUNT': 'main_balance_post',
        'DEDUCT_CHARGE_AMOUNT': 'deduct_charge_amount'
    })

    # Handle datetime format conversion
    if ev['creation_time'].dtype == 'int64' or ev['creation_time'].dtype == 'float64':
        ev['creation_time'] = pd.to_datetime(
            ev['creation_time'].astype(str), format='%Y%m%d%H%M%S', errors='coerce'
        )

    # Handle charge_expire_post
    if 'CHARGE_AMOUNT_EXPIRE_TIME' in ev.columns:
        if ev['CHARGE_AMOUNT_EXPIRE_TIME'].dtype == 'int64' or ev['CHARGE_AMOUNT_EXPIRE_TIME'].dtype == 'float64':
            ev['charge_expire_post'] = pd.to_datetime(
                ev['CHARGE_AMOUNT_EXPIRE_TIME'].astype(str), format='%Y%m%d%H%M%S', errors='coerce'
            )
        else:
            ev['charge_expire_post'] = pd.to_datetime(
                ev['CHARGE_AMOUNT_EXPIRE_TIME'], format='%Y%m%d%H%M%S', errors='coerce'
            )
    else:
        ev['charge_expire_post'] = pd.NaT

    # Handle balance_expire_post
    if 'CURENT_AMOUNT_EXPIRE_TIME' in ev.columns:
        if ev['CURENT_AMOUNT_EXPIRE_TIME'].dtype == 'int64' or ev['CURENT_AMOUNT_EXPIRE_TIME'].dtype == 'float64':
            ev['balance_expire_post'] = pd.to_datetime(
                ev['CURENT_AMOUNT_EXPIRE_TIME'].astype(str), format='%Y%m%d%H%M%S', errors='coerce'
            )
        else:
            ev['balance_expire_post'] = pd.to_datetime(
                ev['CURENT_AMOUNT_EXPIRE_TIME'], format='%Y%m%d%H%M%S', errors='coerce'
            )
    else:
        ev['balance_expire_post'] = pd.NaT

    # Select only needed columns
    cols = [
        'cdr_id', 'creation_time', 'cust_num', 'object_id',
        'offer_id', 'operation_type_id', 'deduct_amount',
        'charge_amount', 'current_amount_pre', 'current_amount_post',
        'charge_expire_post', 'balance_expire_post',
        'deduct_charge_amount', 'main_balance_post'
    ]

    # Keep only columns that exist in our dataframe
    valid_cols = [col for col in cols if col in ev.columns]
    events = ev[valid_cols]

    # Filter out existing CDRs
    new_ev = events[~events['cdr_id'].isin(existing_cdrs)]
    print(f"Found {len(new_ev)} new CDR events to insert.")

    # Process in batches
    if not new_ev.empty:
        batch_size = 1000
        total_inserted = 0

        for start_idx in range(0, len(new_ev), batch_size):
            end_idx = min(start_idx + batch_size, len(new_ev))
            batch = new_ev.iloc[start_idx:end_idx]

            try:
                batch.to_sql('cdr_events', conn, if_exists='append', index=False)
                rows_inserted = len(batch)
                total_inserted += rows_inserted
                print(f"Inserted CDR batch {start_idx}-{end_idx}: {rows_inserted} rows")
            except Exception as e:
                print(f"Error inserting CDR batch {start_idx}-{end_idx}: {e}")

        print(f"Total CDR events inserted: {total_inserted} out of {len(new_ev)} attempted")

print("Data load complete.")