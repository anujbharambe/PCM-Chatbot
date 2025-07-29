# Functioning Overview
This is a chatbot specifically designed for PCM Portal. It can answer user queries(primarily made by clients) through converting them into suitable SQL queries and running them on the database. The process starts at the main function where the user enters his/her JWT token required for authorization. The scope is then extracted and stored in a json for future use. This scope will be automatically applied to all queries made by the user without fail. The user query is then sent into a query classifier that will classify the query as one of the following- sql query, forecasting query, contextual query and others. 
SQL Query- Any query that requires historical DB data.
Forecasting Query- Any query that requires forecasting/predicting of footfall for the future. This is done using Prophet model by facebook.
Contextual Query- Any query that requires a comparative footfall analysis between any 2 sites, areas, regions, orgs, etc.
Other- Any query that does not fall into the above categories will automatically be classified as an other query.

All of the above classifications have a separate folder within the services. Each of these folders contains a chain.py, examples.py and prompt_prefix.py.
chain.py-> Uses an LLM and passes the examples.py along with the prompt_prefix.py.
examples.py-> This file mainly contains respective examples of each type to guide the LLM.
prompt_prefix.py-> This file is has an in-detailed explaination of the Database Schema along with the tables used within. 

-->The chain.py is called within a flagship file named respectively in each of the above folders.

-->The scope restrictions are once again applied to the SQL Query for safety within the main function.

-->The SQL Query is then executed by the sql_query_executer.py within the sql_query_executer folder.

-->The response from the SQL Query execution is then passed to the normal response formatter folder which follows the same work-flow as the above 4 folders.

-->This cycle continues until the user enters exit/quit. The code exits at main.py.

**The JWT Token must be re-generated every 8 hours for authorization.

## Architecture
Mermaid Flowchart Code-> Paste the following into mermaid to view the flowchart-:

**---------------------------------------------------------------------------------**
flowchart TD
    A[Start: main.py] --> B[User enters JWT Token]
    B --> C[Validate & Store JWT Token]
    C --> D[Extract Scope and store in JSON]
    D --> E[User enters a query]
    E --> F[Query Classifier]
    F -->|SQL Query| G1[services/sql_query/chain.py]
    F -->|Forecasting Query| G2[services/forecasting/chain.py]
    F -->|Contextual Query| G3[services/contextual/chain.py]
    F -->|Other Query| G4[services/others/chain.py]

    subgraph Service Folder
        G1 --> H1[Load examples.py]
        G1 --> I1[Load prompt_prefix.py]
        G2 --> H2[Load examples.py]
        G2 --> I2[Load prompt_prefix.py]
        G3 --> H3[Load examples.py]
        G3 --> I3[Load prompt_prefix.py]
        G4 --> H4[Load examples.py]
        G4 --> I4[Load prompt_prefix.py]
    end

    G1 --> J[Re-apply Scope Restrictions]
    J --> K[Execute SQL in sql_query_executer.py]
    K --> L[Format response using response_formatter/chain.py]
    L --> M[Return response to user]
    G2 --> N[Forecast using Prophet Model]
    N --> O[Format response using response_formatter/chain.py]
    O --> M
    G3 --> P[Compare footfall across entities]
    P --> Q[Format response using response_formatter/chain.py]
    Q --> M
    G4 --> R[Handle Other Queries]
    R --> S[Format response using response_formatter/chain.py]
    S --> M

    M --> T{Exit?}
    T -->|Yes| U[End: Exit main.py]
    T -->|No| E

    B -.-> V[JWT expires in 8 hours]
    V -.-> B
**--------------------------------------------------------------------------------------**

# PCM Chatbot

A comprehensive AI-powered analytics chatbot that provides intelligent querying capabilities for IoT and people counting data with advanced scope-based security, forecasting, and contextual analysis features.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Database Schema](#database-schema)


## Overview

The Analytics Chatbot is designed to provide natural language querying capabilities for analytics data, specifically focusing on IoT device data and people counting metrics. It leverages advanced AI models to classify queries, generate SQL statements, perform forecasting, and provide contextual analysis while maintaining strict security through JWT-based authentication and scope-based access control.

## Features

### Core Capabilities

- **Natural Language Query Processing**: Convert natural language questions into SQL queries
- **Query Classification**: Automatically classify queries into different types (SQL, forecasting, contextual, general)
- **Forecasting**: Advanced time-series forecasting using Facebook Prophet
- **Contextual Analysis**: Comparative analysis between time periods
- **Vector Search**: Semantic search using Pinecone for context retrieval
- **Conversation Logging**: Persistent conversation history with JSONL format *(Note: The support for this feature has been added. To enable this feature pass 'recent_log_context' into all query generator functions and use them as a BasePrompt in the prompt prefix)*

### Security Features

- **JWT Authentication**: Secure token-based authentication
- **Scope-Based Access Control**: Multi-level access control (Organization, Region, Area, Site)
- **Super Admin Support**: Override capabilities for administrative users
- **Dynamic Query Injection**: Automatic scope filtering in SQL queries

### Data Processing

- **Real-time SQL Execution**: Execute generated queries against PostgreSQL database
- **Date Placeholder Resolution**: Smart date handling with relative date expressions
- **Query Optimization**: Automatic JOIN injection and query fixing
- **Response Formatting**: Intelligent formatting of query results


## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Pinecone account
- Required environment variables

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Analytics-Chatbot/pcm_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Database setup**
   ```bash
   # Ensure PostgreSQL is running and accessible
   # Import your database schema and data
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# JWT Configuration
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# OpenAI Configuration (if using OpenAI for LLM)
GOOGLE_API_KEY=your_google_api_key
```

### Database Schema

The application expects the following key tables:

- `o_iot_pcm_raw_data`: IoT device raw data
- `device_activity_forecast`: Forecasting data
- `o_site`: Site information
- `organisation`: Organization data
- `o_scope_template_mapping`: Scope permissions

## Usage

### Command Line Interface

```bash
python main.py
```

The application will prompt for:
1. JWT token for authentication
2. Natural language queries

### Query Types

#### 1. SQL Queries
Natural language questions that translate to SQL:
```
"Show me the footfall data for last month"
"What are the top 5 sites by activity?"
```

#### 2. Forecasting Queries
Requests for predictive analysis:
```
"Predict footfall for next month"
"What's the forecasted activity for site 123?"
```

#### 3. Contextual Queries
Comparative analysis between periods:
```
"Compare this month's footfall with last month"
"How does this quarter compare to last quarter?"
```

#### 4. General Questions
Non-data specific questions:
```
"How does the forecasting work?"
"What data sources are available?"
```

### Example Session

```
🔍 AI Assistant
Available scopes: site, area, region

Enter your JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
✅ Extracted scope JSON: {'2_1': True, '5_123': True}

Type your query below (type 'exit' to quit):

📝 Query: Show me footfall data for last week
📊 Classification: sql_query
🔒 Applied scope filter: organisation = ['1']
Output: [Formatted results showing footfall data with scope restrictions applied]

📝 Query: exit
👋 Conversation Saved to logging.json. Goodbye!
```


### Core Functions

#### Authentication & Authorization

```python
def verify_token(token: str) -> dict
```
Verifies JWT token and returns payload.

```python
def extract_scope_from_token_payload(payload: dict) -> Optional[UserScope]
```
Extracts user scope from JWT payload.

#### Query Processing

```python
def classify_query(user_input: str) -> str
```
Classifies query type: `sql_query`, `forecasting_query`, `contextual_query`, or `other`.

```python
def apply_scope_to_query(sql_query: str, scope_json: dict, token_payload: dict = None) -> str
```
Applies scope-based security filters to SQL queries.

#### Forecasting

```python
def insert_forecasting_data_for_site(site_id: int)
```
Generates Prophet-based forecasts for a specific site.

```python
def insert_forecasting_data(scope_json: Optional[dict] = None)
```
Bulk forecast generation for all sites within scope.

### Scope Types

```python
class ScopeType(Enum):
    SITE = "site"
    AREA = "area" 
    REGION = "region"
    ZONE = "zone"
    ORG = "organisation"
```

### User Scope Configuration

```python
class UserScope:
    def __init__(self, scope_type: ScopeType, scope_values: List[str]):
        self.scope_type = scope_type
        self.scope_values = scope_values
```

## Database Schema

### Key Tables

#### o_iot_pcm_raw_data
- `o_site_id`: Foreign key to site
- `from_time`: Timestamp of data point
- Device activity and counting data

#### device_activity_forecast  
- `site_id`: Foreign key to site
- `date`: Forecast date
- `predicted`: Predicted value
- `predicted_lower`: Lower bound
- `predicted_upper`: Upper bound

#### o_site
- `site_id`: Primary key
- `o_id`: Organization ID
- `area_id`: Area identifier
- `region_id`: Region identifier
- `is_active`: Active status

#### o_scope_template_mapping
- `scope_id`: Scope identifier
- `scope_type_id`: Type of scope (2=org, 3=region, 4=area, 5=site)
- `filter_id`: ID of the filtered entity

## Security

### JWT Token Structure

Expected JWT payload:
```json
{
  "user_id": "123",
  "scope_id": 456,
  "is_super_admin": 0,
  "o_id": 789,
  "scope_type": "organisation",
  "scope_values": ["1", "2"],
  "exp": 1234567890
}
```

### Scope-Based Access Control

The system implements hierarchical access control:

1. **Super Admin**: Full access, no restrictions
2. **Organization Level**: Access to all sites within organization(s)
3. **Region Level**: Access to sites within specific region(s)
4. **Area Level**: Access to sites within specific area(s)
5. **Site Level**: Access to specific site(s) only

