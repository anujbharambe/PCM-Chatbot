import json
from math import ceil
from chatbot_src.services.sql_query_generator.generate_query import generate_sql_query
from chatbot_src.services.forcasting.generate_forecasting_query import (
    generate_forecasting_sql_query,
)
from chatbot_src.services.sql_query_executer.sql_query_executer import sql_query_executer
from chatbot_src.services.contextual_query.params_extraction.get_extracted_params import get_extracted_params
from chatbot_src.services.contextual_query.comparative_sql_query_generator.generate_comparative_sql_query import generate_comparative_sql_query
from chatbot_src.services.response_formatter.generate_response import generate_response
from chatbot_src.services.normal_response_formatter.generate_normal_response import (
    generate_normal_response,
)
from chatbot_src.services.other_questions.get_other_questions_answer import get_other_questions_answer
from chatbot_src.services.query_classifier.classify import classify_query
from fastapi import APIRouter, Depends, HTTPException, Query, Body, FastAPI
from uvicorn import run
from chatbot_src.db import get_db, get_raw_db
from fastapi.middleware.cors import CORSMiddleware
from chatbot_src.db.alchemy_models import o_iot_pcm_raw_data, device_forecast, o_site
from sqlalchemy.orm import Session
from loguru import logger
import psycopg2.extras
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from chatbot_src.services.vector_store.pinecone_client import search_similar_context, upsert_texts
from chatbot_src.services.followup_handler.get_followup_question import get_followup_question
import os
from typing import List, Optional
from enum import Enum
import re
import jwt
# from routes.Scope.scope import get_a_users_scope_json
# from utils.jwt_utils import verify_token
from dotenv import load_dotenv
load_dotenv()
from os import getenv;

SECRET = getenv("SECRET_KEY")
ALGORITHM = "HS256"


LOG_FILE = 'logging.jsonl'

def log_conversation(user_input, assistant_output):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "assistant": assistant_output
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_recent_log_context(n=10):
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
        context = ""
        for entry in lines:
            try:
                j = json.loads(entry)
                context += f"User: {j['user']}\nAssistant: {j['assistant']}\n"
            except:
                continue
        return context.strip()
    
class ScopeType(Enum):
    SITE = "site"
    AREA = "area"
    REGION = "region"
    ZONE = "zone"
    ORG = "organisation"  # Add this line

# Add user scope configuration
class UserScope:
    def __init__(self, scope_type: ScopeType, scope_values: List[str]):
        self.scope_type = scope_type
        self.scope_values = scope_values



def apply_scope_to_query(sql_query: str, scope_json: dict, token_payload: dict = None) -> str:
    """Apply scope restrictions dynamically based on scope_json and token_payload."""

    # 🛡️ Super admin → no restriction
    if token_payload and token_payload.get("is_super_admin", 0) == 1:
        return sql_query.strip().rstrip(";") + ";"

    # 🛡️ No scope → deny
    if not scope_json:
        raise HTTPException(status_code=403, detail="No scope permissions found")

    # 🧹 Clean input SQL
    sql_query = sql_query.strip()
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()
    if sql_query.endswith(";"):
        sql_query = sql_query[:-1].strip()

    # 🔍 Parse scope IDs
    site_ids = []
    area_ids = []
    region_ids = []
    org_ids = []

    for key in scope_json:
        try:
            scope_type_id, filter_id = key.split("_")
            if scope_json[key]:
                if scope_type_id == "5":
                    site_ids.append(filter_id)
                elif scope_type_id == "4":
                    area_ids.append(filter_id)
                elif scope_type_id == "3":
                    region_ids.append(filter_id)
                elif scope_type_id == "2":
                    org_ids.append(filter_id)
        except Exception as e:
            continue  # Ignore invalid keys

    # 🧠 Build WHERE conditions
    conditions = []
    sql_flat = sql_query.lower().replace(" ", "")

    if site_ids:
        site_str = ", ".join(site_ids)
        conditions.append(f"o_site.site_id IN ({site_str})")

    if area_ids:
        area_str = ", ".join(area_ids)
        conditions.append(f"o_site.area_id IN ({area_str})")

    if region_ids:
        region_str = ", ".join(region_ids)
        conditions.append(f"o_site.region_id IN ({region_str})")

    if org_ids:
        org_str = ", ".join(org_ids)
    
        # Check if the existing query includes the correct org scope
        existing_match = re.search(r"organisation\.o_id\s+IN\s*\(([^)]+)\)", sql_query, re.IGNORECASE)
    
        if existing_match:
            existing_ids = [x.strip() for x in existing_match.group(1).split(",")]
            expected_ids = [str(x) for x in org_ids]
            if set(existing_ids).issubset(set(expected_ids)):
                print("✅ organisation.o_id filter already matches scope – no injection needed.")
            else:
                print(f"⚠️ organisation.o_id filter does not match scope — overriding.")
                # remove the existing wrong filter
                sql_query = re.sub(r"\s+AND\s+organisation\.o_id\s+IN\s*\([^)]+\)", "", sql_query, flags=re.IGNORECASE)
                conditions.append(f"organisation.o_id IN ({org_str})")
        else:
            conditions.append(f"organisation.o_id IN ({org_str})")


    #  Deny if no valid scope condition was applied and LLM didn’t do it
    if not conditions:
        if not any(x in sql_flat for x in ["organisation.o_id", "o_site.site_id", "o_site.region_id", "o_site.area_id"]):
            raise HTTPException(status_code=403, detail="No valid scope restrictions could be applied")

    #  Inject WHERE clause
    scope_clause = " AND (" + " OR ".join(conditions) + ")" if conditions else ""
    
    group_by_match = re.search(r"\bGROUP BY\b", sql_query, re.IGNORECASE)
    order_by_match = re.search(r"\bORDER BY\b", sql_query, re.IGNORECASE)
    
    if scope_clause:
        if "WHERE" in sql_query.upper():
            if group_by_match:
                sql_query = re.sub(r"\bGROUP BY\b", f"{scope_clause} GROUP BY", sql_query, flags=re.IGNORECASE)
            elif order_by_match:
                sql_query = re.sub(r"\bORDER BY\b", f"{scope_clause} ORDER BY", sql_query, flags=re.IGNORECASE)
            else:
                sql_query += scope_clause
        else:
            if group_by_match:
                sql_query = re.sub(r"\bGROUP BY\b", f"WHERE 1=1 {scope_clause} GROUP BY", sql_query, flags=re.IGNORECASE)
            elif order_by_match:
                sql_query = re.sub(r"\bORDER BY\b", f"WHERE 1=1 {scope_clause} ORDER BY", sql_query, flags=re.IGNORECASE)
            else:
                sql_query += f" WHERE 1=1 {scope_clause}"


    # Inject JOINs
    sql_query = ensure_join_for_scope(sql_query, scope_json)

    return sql_query.strip() + ";"


def ensure_join_for_scope(sql_query: str, scope_json: dict) -> str:
    """
    Ensure necessary JOINs (e.g., o_site, organisation) are injected for queries that need them,
    such as those referencing device_activity_forecast or o_iot_pcm_raw_data.
    """

    # 1. Inject JOIN o_site for o_iot_pcm_raw_data if not already joined
    if (
        re.search(r"\bFROM\s+o_iot_pcm_raw_data\b", sql_query, re.IGNORECASE)
        and not re.search(r"\bJOIN\s+o_site\b", sql_query, re.IGNORECASE)
    ):
        sql_query = re.sub( 
            r"\bFROM\s+o_iot_pcm_raw_data\b",
            "FROM o_iot_pcm_raw_data JOIN o_site ON o_iot_pcm_raw_data.o_site_id = o_site.site_id",
            sql_query,
            flags=re.IGNORECASE
        )

    # 2. Inject JOIN o_site for device_activity_forecast if not already joined
    if (
        re.search(r"\bFROM\s+device_activity_forecast\b", sql_query, re.IGNORECASE)
        and not re.search(r"\bJOIN\s+o_site\b", sql_query, re.IGNORECASE)
    ):
        sql_query = re.sub(
            r"\bFROM\s+device_activity_forecast\b",
            "FROM device_activity_forecast JOIN o_site ON device_activity_forecast.site_id = o_site.site_id",
            sql_query,
            flags=re.IGNORECASE
        )

    # 3. Inject JOIN organisation if organisation.o_id is referenced and not already joined
    if (
        "organisation.o_id" in sql_query
        and not re.search(r"\bJOIN\s+organisation\b", sql_query, re.IGNORECASE)
    ):
        # Determine where to inject the organisation join — after o_site
        if re.search(r"\bJOIN\s+o_site\s+ON\b.*?site_id", sql_query, re.IGNORECASE):
            sql_query = re.sub(
                r"(JOIN\s+o_site\s+ON\s+[\w\.]+\s*=\s*[\w\.]+)",
                r"\1 JOIN organisation ON o_site.o_id = organisation.o_id",
                sql_query,
                flags=re.IGNORECASE
            )
        # If o_site isn't joined yet (edge case), fallback inject after FROM clause
        elif re.search(r"\bFROM\s+\w+", sql_query, re.IGNORECASE):
            sql_query = re.sub(
                r"(\bFROM\s+\w+)",
                r"\1 JOIN o_site ON o_iot_pcm_raw_data.o_site_id = o_site.site_id JOIN organisation ON o_site.o_id = organisation.o_id",
                sql_query,
                flags=re.IGNORECASE
            )

    return sql_query


def ensure_where_clause(sql_query: str) -> str:
    # Only add WHERE if not already present
    if re.search(r"\bWHERE\b", sql_query, re.IGNORECASE):
        return sql_query
    # Insert WHERE before the first condition after the last JOIN
    pattern = r"(JOIN\s+organisation\s+ON\s+o_site\.o_id\s*=\s*organisation\.o_id)(\s*)([a-zA-Z_]+\s+BETWEEN)"
    replacement = r"\1 WHERE \3"
    sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
    return sql_query

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )
    
def extract_scope_from_token_payload(payload: dict) -> Optional[UserScope]:
    """
    Extracts UserScope from a JWT token payload.
    If 'scope_type' and 'scope_values' are present, use them.
    Otherwise, if 'o_id' is present, use organisation-level scope.
    """
    try:
        scope_type_str = payload.get("scope_type")
        scope_values = payload.get("scope_values")

        if scope_type_str and scope_values:
            scope_type = ScopeType(scope_type_str.lower())
            return UserScope(scope_type=scope_type, scope_values=scope_values)

        # Fallback: if o_id is present, treat as organisation scope
        o_id = payload.get("o_id")
        if o_id:
            return UserScope(scope_type=ScopeType.ORG, scope_values=[str(o_id)])

        return None

    except Exception as e:
        logger.error(f"Error extracting user scope from token: {e}")
        return None
    
def extract_sql(label: str, text: str) -> str:
    """
    Extracts the SQL statement following a label (e.g., CURRENT_PERIOD_SQL:) from the LLM output.
    Returns the SQL string, or raises ValueError if not found.
    """
    pattern = rf"{label}:\s*([\s\S]+?);(?:\s*BASELINE_PERIOD_SQL:|\s*$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip() + ";"
    raise ValueError(f"Could not find SQL for {label}")

def fix_extract_functions(sql_query: str) -> str:
    """
    Replace any EXTRACT(YEAR/MONTH FROM ... JOIN ...) or malformed EXTRACT usage
    with EXTRACT(YEAR/MONTH FROM o_iot_pcm_raw_data.from_time).
    """
    # Replace any EXTRACT(YEAR FROM ...JOIN...) with correct column
    sql_query = re.sub(
        r"EXTRACT\(YEAR FROM [^)]+?JOIN[^\)]*?\)",
        "EXTRACT(YEAR FROM o_iot_pcm_raw_data.from_time)",
        sql_query,
        flags=re.IGNORECASE
    )
    # Replace any EXTRACT(MONTH FROM ...JOIN...) with correct column
    sql_query = re.sub(
        r"EXTRACT\(MONTH FROM [^)]+?JOIN[^\)]*?\)",
        "EXTRACT(MONTH FROM o_iot_pcm_raw_data.from_time)",
        sql_query,
        flags=re.IGNORECASE
    )
    # Replace any EXTRACT(YEAR FROM ...) not referencing a column
    sql_query = re.sub(
        r"EXTRACT\(YEAR FROM [^)]+?\)",
        "EXTRACT(YEAR FROM o_iot_pcm_raw_data.from_time)",
        sql_query,
        flags=re.IGNORECASE
    )
    # Replace any EXTRACT(MONTH FROM ...) not referencing a column
    sql_query = re.sub(
        r"EXTRACT\(MONTH FROM [^)]+?\)",
        "EXTRACT(MONTH FROM o_iot_pcm_raw_data.from_time)",
        sql_query,
        flags=re.IGNORECASE
    )
    return sql_query

def get_scope_json_from_db(scope_id: int, rdb) -> dict:
    """Query o_scope_template_mapping and return scope JSON."""
    cursor = rdb.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = """
        SELECT scope_type_id, filter_id
        FROM o_scope_template_mapping
        WHERE scope_id = %s
    """
    cursor.execute(query, (scope_id,))
    results = cursor.fetchall()
    
    scope_json = {}
    for row in results:
        key = f"{row['scope_type_id']}_{row['filter_id']}"
        scope_json[key] = True
    return scope_json


def resolve_date_placeholders(sql_query: str) -> str:
    today = datetime.today()
    # Example for last month
    first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day_last_month = today.replace(day=1) - timedelta(days=1)
    sql_query = sql_query.replace("{today | start_of_last_month}", first_day_last_month.strftime("%Y-%m-%d"))
    sql_query = sql_query.replace("{today | end_of_last_month}", last_day_last_month.strftime("%Y-%m-%d"))
    return sql_query

def insert_forecasting_data_for_site(site_id: int):
    """
    Generates and inserts forecasting data for a given site_id using Prophet.
    """
    rdb = next(get_raw_db())
    cursor = rdb.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Fetch daily activity data for the site
        query = """
            SELECT
                from_time::date as ds,
                COUNT(*) as y
            FROM o_iot_pcm_raw_data
            WHERE o_site_id = %(o_site_id)s
            GROUP BY from_time::date, o_site_id
            ORDER BY from_time::date;
        """
        cursor.execute(query, {"o_site_id": site_id})
        data = cursor.fetchall()
        if not data:
            print(f"⚠️ No data found for site_id {site_id}. Skipping.")
            return

        # Prepare DataFrame
        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
        df["floor"] = 0
        df["cap"] = df["y"].max() * 1.2  

        # Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            growth="logistic"
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=365)
        future["floor"] = 0
        future["cap"] = df["cap"].iloc[0] 

        forecast = model.predict(future)
        forecast[["yhat", "yhat_lower", "yhat_upper"]] = (
            forecast[["yhat", "yhat_lower", "yhat_upper"]]
            .clip(lower=0)
            .round(0)
            .astype(int)
        )

        # Insert forecasted data into device_activity_forecast
        for _, row in forecast.tail(365).iterrows():
            cursor.execute(
                """
                INSERT INTO device_activity_forecast 
                (site_id, date, predicted, predicted_lower, predicted_upper, is_deleted, created_time, updated_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    site_id,
                    str(row["ds"]),
                    row["yhat"],
                    row["yhat_lower"],
                    row["yhat_upper"],
                    0,
                    str(datetime.now()),
                    str(datetime.now()),
                ),
            )
        rdb.commit()
        print(f"✅ Forecasting data inserted for site_id {site_id}")

    except Exception as e:
        raise ValueError(f"Error fetching data for site_id {site_id}: {e}")
    

def insert_forecasting_data(scope_json: Optional[dict] = None):
    """
    Loops over all active sites in the scope_json and inserts forecasting data for each.
    """
    rdb = next(get_raw_db())
    cursor = rdb.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        site_ids = []

        if scope_json:
            # Parse site and org IDs from scope_json
            scoped_site_ids = tuple(
                int(key.split('_')[1]) for key in scope_json if key.startswith("5_") and scope_json[key]
            )
            scoped_org_ids = tuple(
                int(key.split('_')[1]) for key in scope_json if key.startswith("2_") and scope_json[key]
            )

            if scoped_site_ids:
                query = "SELECT site_id FROM o_site WHERE is_active = 1 AND site_id IN %s"
                cursor.execute(query, (scoped_site_ids,))
                site_ids = [row["site_id"] for row in cursor.fetchall()]
            elif scoped_org_ids:
                query = "SELECT site_id FROM o_site WHERE is_active = 1 AND o_id IN %s"
                cursor.execute(query, (scoped_org_ids,))
                site_ids = [row["site_id"] for row in cursor.fetchall()]
            else:
                print("⚠️ No valid site/org scope values provided. Fetching all active sites.")
                query = "SELECT site_id FROM o_site WHERE is_active = 1"
                cursor.execute(query)
                site_ids = [row["site_id"] for row in cursor.fetchall()]
        else:
            # No scope provided, fetch all active sites
            query = "SELECT site_id FROM o_site WHERE is_active = 1"
            cursor.execute(query)
            site_ids = [row["site_id"] for row in cursor.fetchall()]

        if not site_ids:
            print("⚠️ No active scoped sites found for forecasting.")
            return

        for site_id in site_ids:
            insert_forecasting_data_for_site(site_id=site_id)

    except Exception as e:
        raise ValueError(f"Error inserting forecasting data: {e}")



def main():
    
    # Example: get token from user and extract scope
    print("🔍 AI Assistant")
    print("Available scopes: site, area, region")

    token = input("Enter your JWT token: ").strip()
    try:
        token_payload = verify_token(token)
        rdb = next(get_raw_db())
        db = next(get_db())
    
        user_scope = extract_scope_from_token_payload(token_payload)
    
        if token_payload.get("is_super_admin", 0) == 1:
            print("✅ Super admin — no scope restrictions will apply.")
            scope_json = {}
        else:
            scope_id = token_payload.get("scope_id")
            if scope_id is not None:
                scope_json = get_scope_json_from_db(scope_id, rdb)
                print(f"✅ Extracted scope JSON: {scope_json}")
            else:
                print("⚠️ No scope_id in token payload; no scope restrictions will apply.")
                scope_json = {}

    except Exception as e:
        print(f"❌ Error decoding token: {e}")
        exit()

    insert_forecasting_data(scope_json)

    print("Type your query below (type 'exit' to quit):\n")

    while True:
        user_input = input("📝 Query: ").strip()
        output = ""
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Conversation Saved to logging.json. Goodbye!")
            break

        log_context = get_recent_log_context() #I have set this to 10 recent messages, you can change it as needed
        vector_context_results = search_similar_context(user_input, session_id='default')
        vector_context = "\n\n".join([r.page_content for r in vector_context_results]) if vector_context_results else ""
        
        combined_context = f"{log_context}\n\n{vector_context}".strip()
        # user_input = get_followup_question(combined_context, user_input)

        classification = classify_query(user_input)
        print(f"📊 Classification: {classification}\n")
        if classification == "sql_query":
            sql_query = generate_sql_query(user_input)
            
            if user_scope:
                sql_query = apply_scope_to_query(sql_query, scope_json, token_payload)
                print(f"🔒 Applied scope filter: {user_scope.scope_type.value} = {user_scope.scope_values}")
            
            sql_query = resolve_date_placeholders(sql_query)
            sql_query = fix_extract_functions(sql_query)
            output = sql_query_executer(sql_query, user_input)


        elif classification == "forecasting_query":
            extracted_params = get_extracted_params(user_input)
            extracted_params = json.loads(extracted_params.replace('```json', '').replace('```', ''))
        
            metric = extracted_params.get("metric", "footfall")
            time_period = extracted_params.get("time_period", "this month")
            trend = extracted_params.get("trend", "daily")
        
            sql_query = generate_forecasting_sql_query(metric, user_input, time_period, trend)
            
            # Patch ambiguous is_deleted reference
            sql_query = re.sub(r"\bis_deleted\b", "device_activity_forecast.is_deleted", sql_query)
            
            sql_query = apply_scope_to_query(sql_query, scope_json, token_payload)
            sql_query = ensure_where_clause(sql_query)
            sql_query = resolve_date_placeholders(sql_query)
            output = sql_query_executer(sql_query, user_input)

        elif classification == "contextual_query":
            extracted_params = get_extracted_params(user_input)
            extracted_params = json.loads(extracted_params.replace('```json', '').replace('```', ''))
            sql_block = generate_comparative_sql_query(
                extracted_params["metric"],
                extracted_params["trend"],
                extracted_params["time_period"],
                extracted_params["previous_time_period"]
            )
            # Extract both SQLs
            current_sql = extract_sql("CURRENT_PERIOD_SQL", sql_block)
            baseline_sql = extract_sql("BASELINE_PERIOD_SQL", sql_block)
            # Apply scope if needed
            if user_scope:
                current_sql = apply_scope_to_query(sql_query, scope_json, token_payload)
                baseline_sql = apply_scope_to_query(sql_query, scope_json, token_payload)
                print(f"🔒 Applied scope filter: {user_scope.scope_type.value} = {user_scope.scope_values}")
            print(f"🔍 Final SQL:\nCURRENT_PERIOD_SQL:\n{current_sql}\n\nBASELINE_PERIOD_SQL:\n{baseline_sql}")
            # You may want to execute both or just one depending on your logic
            output = sql_query_executer(current_sql, user_input)
        else:
            output = get_other_questions_answer(user_input)


        qa_pair = f"User:{user_input}\nAssistant:{output}\n"
        log_conversation(user_input, output)
        metadata = {"session_id" : "default"}
        upsert_texts([qa_pair], [metadata], session_id="default")
        print(f"Output : {output}")


if __name__ == "__main__":
    main()

logger.info("Started main")