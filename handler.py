import json
import os
import boto3
import base64
import logging
from botocore.exceptions import ClientError

from openai import OpenAI
from sqlalchemy import create_engine, text
from collections import defaultdict
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)



# Load environment variables
profilesActive = os.getenv("PROFILES_ACTIVE")
smDBAppPostgresName = os.getenv("SM_DB_APP_POSTGRES_NAME")
smOpenAiName = os.getenv("SM_OPEN_AI_NAME")
smAppConfigName = os.getenv("SM_APP_CONFIG_NAME")

## Load the variables from ssm
botoSession = boto3.session.Session()
smClient = botoSession.client(service_name='secretsmanager')

smDBAppPostgresResponse = smClient.get_secret_value(SecretId=smDBAppPostgresName)
smDBAppPostgresSecretDict = json.loads(smDBAppPostgresResponse['SecretString'])

smOpenAiNameResponse = smClient.get_secret_value(SecretId=smOpenAiName)
smOpenAiNameDict = json.loads(smOpenAiNameResponse['SecretString'])

smAppConfigNameResponse = smClient.get_secret_value(SecretId=smAppConfigName)
smAppConfigNameDict = json.loads(smAppConfigNameResponse['SecretString'])

api_key = smOpenAiNameDict['OPENAI_API_KEY'] 

# Load the configuration file
# with open('config.json', 'r') as file:
#    config = json.load(file)
config = json.loads(base64.b64decode(smAppConfigNameDict["appConfig"]))

# Extract schema from the config
schema = config['schema']

# Combine the lines to form the full prompt with f-string
prompt_template = "\n".join(config["prompt"])

# Format the prompt with the schema variable
base_prompt = prompt_template.format(schema=schema)

# Extract results_instructions from config
results_instructions = config['results_instructions']

# Extract main_bot_instructions from config
main_bot_instructions = config['main_bot_instructions']

q3 = config['query_template']

sql_sys_content = config['sql_sys_content']

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Default assignment if 'than' is not in filters
sym_str = '>'

def get_db_params(database_type='main'):
    # Define mapping between database type and its corresponding key suffix in the secrets dictionary
    db_key_mapping = {
        'main': '',
        'AI': '_AI'
    }
    
    # Get the appropriate suffix for the database type
    suffix = db_key_mapping.get(database_type)
    
    if suffix is None:
        raise ValueError(f"Unsupported database type: {database_type}")
    
    # Retrieve the connection parameters from the secrets dictionary
    db_params = {
        'dbname': smDBAppPostgresSecretDict[f'dbname{suffix}'],
        'user': smDBAppPostgresSecretDict[f'username{suffix}'],
        'password': smDBAppPostgresSecretDict[f'password{suffix}'],
        'host': smDBAppPostgresSecretDict[f'host{suffix}'],
        'port': smDBAppPostgresSecretDict[f'port{suffix}'],
        'engine': smDBAppPostgresSecretDict[f'engine{suffix}']
    }
    
    # Create the connection string using db_params
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    
    return connection_string

def clean_filters(filters):
    # Define price-related keys
    price_keys = {'max_price_per_day', 'start_price', 'price', 'price_per_night'}
    
    # Clean the filters
    cleaned_filters = {k: v for k, v in filters.items()
                       if v is not None and not (k in price_keys and v == 0)}
    
    unwanted_substrings = ['themed', 'office', 'cinema', 'disney', 'theme']
    
    # Find keys containing unwanted substrings
    keys_to_pop = [key for key in cleaned_filters if any(sub in key.lower() for sub in unwanted_substrings)]


    # Handle special categories
    if any('office' in key.lower() for key in cleaned_filters):
        if 'house_category' in cleaned_filters:
            if 'INTEGRATED_OFFICE' not in cleaned_filters['house_category']:
                cleaned_filters['house_category'].append('INTEGRATED_OFFICE')
        else:
            cleaned_filters['house_category'] = ['INTEGRATED_OFFICE']
                
    if any('cinema' in key.lower() for key in cleaned_filters):
        if 'house_category' in cleaned_filters:
            if 'CINEMA' not in cleaned_filters['house_category']:
                cleaned_filters['house_category'].append('CINEMA')
        else:
            cleaned_filters['house_category'] = ['CINEMA']
    
    if any(substring in key.lower() for key in cleaned_filters for substring in ['themed', 'theme']):
        if 'house_category' in cleaned_filters:
            if 'THEMATIC' not in cleaned_filters['house_category']:
                cleaned_filters['house_category'].append('THEMATIC')
        else:
            cleaned_filters['house_category'] = ['THEMATIC']


    if 'Disney' in cleaned_filters.get('nearby_attractions', '') or cleaned_filters.get('location', '') or cleaned_filters.get('proximity_to_disney', ''):
        if 'house_category' in cleaned_filters:
            if 'NEAR_DISNEY' not in cleaned_filters['house_category']:
                cleaned_filters['house_category'].append('NEAR_DISNEY')
        else:
            cleaned_filters['house_category'] = ['NEAR_DISNEY']
    
    # Remove each key found
    for key in keys_to_pop:
        cleaned_filters.pop(key, None)
            
    # Remove specific keys
    cleaned_filters.pop('location', None)
    cleaned_filters.pop('proximity_to_disney', None)
    cleaned_filters.pop('has_office', None)
    cleaned_filters.pop('city', None)
    cleaned_filters.pop('available', None)
    cleaned_filters.pop('tables', None)
   
    # Handle 'have_pool' value
    if 'have_pool' in cleaned_filters:
        cleaned_filters['have_pool'] = cleaned_filters['have_pool'] == 1

    # Handle max occupancy
    keys_to_check = {'accommodation', 'accommodation_capacity', 'accommodates', 'capacity', 'max_occupancy', 'max_capacity', 'max_guests'}
    
    # Set default max_occupancy value
    max_occupancy_value = 1
    
    # Update max_occupancy if any of the keys are present
    for key in list(cleaned_filters.keys()):
        if key in keys_to_check:
            max_occupancy_value = cleaned_filters.pop(key)
            break  # Exit loop once the first matching key is found

    cleaned_filters['max_occupancy'] = max_occupancy_value


    return cleaned_filters

def handle_condition(key, value, filters):
    
    


    
    # Handle house categories
    if key == 'house_category':
    # Use a single 'IN' condition
        categories = ', '.join([f"'{category}'" for category in value])
        return f"hc.name IN ({categories})"
    
    # Handle max occupancy
    if key == 'accommodation' or key == 'accommodation_capacity' or key == 'accommodates' or key == 'capacity' or key == 'max_occupancy' or key == 'max_capacity':                          
        return f"h.max_occupancy >= {value}"
    
    # Handle check-in and check-out dates
    if key in ['check_in', 'check_out', 'check_in_date', 'check_out_date']:
        if 'check_in' in filters and 'check_out' in filters:
            check_in_date = filters.get('check_in') or filters.get('check_in_date')
            check_out_date = filters.get('check_out') or filters.get('check_out_date')
            
            # Get the current year
            current_year = datetime.now().year

            # Convert string to a datetime object
            check_in_date = datetime.strptime(check_in_date, "%Y-%m-%d")
            check_out_date = datetime.strptime(check_out_date, "%Y-%m-%d")

            # Check if the year is last year and replace it if necessary
            if check_in_date.year == (current_year - 1):
                check_in_date = check_in_date.replace(year=current_year)
            # Check if the year is last year and replace it if necessary
            if check_out_date.year == (current_year - 1):
                check_out_date = check_out_date.replace(year=current_year)

            # Convert back to string if needed
            check_in_date = check_in_date.strftime("%Y-%m-%d")
            check_out_date = check_out_date.strftime("%Y-%m-%d")

            
            return f"(hs.date BETWEEN '{check_in_date}' AND '{check_out_date}')"
    
    # Handle max price per day
    if key == 'max_price_per_day' or key == 'start_price' or key == 'price' or key == 'price_per_night':
        return f"h.start_price * c.value {sym_str} {value}"
    
    # Handle boolean or other simple key-value conditions
    return f"h.{key} = {value}"

def generate_where_query_pandas(filters):
    # Base query
    query = "HAVING\n    COUNT(CASE WHEN hs.value = 0 THEN 1 END) = 0"

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(filters.items()), columns=["key", "value"])
    

    # Generate the SQL conditions vectorized by key
    df['condition'] = df.apply(lambda row: handle_condition(row['key'], row['value'], filters), axis=1)
    
    if "check_in" in df['key'].values:
        where_query = f"WHERE\n    {df.loc[df['key'] == 'check_in', 'condition'].values[0]}"
        df = df[~df['key'].isin(['check_in', 'check_out'])]
    
    # Combine all the conditions into a single SQL query
    additional_conditions = '\n    AND '.join(df['condition'].dropna().unique())

    # Combine the base query with the additional conditions
    if additional_conditions:
        query += f"\n    AND {additional_conditions}"
    
    query = query.replace('True', 'TRUE').replace('False', 'FALSE')

    return query, where_query
    
def aggregate_categories(data):
    # Create a dictionary to store the aggregated results
    aggregated_data = defaultdict(lambda: {
        "houseId": None,
        "houseName": None,
        "bathroomAmount": None,
        "maxOccupancy": None,
        "latitude": None,
        "longitude": None,
        "roomAmount": None,
        "city": None,
        "condominium_id": None,
        "mainPicture": None,
        "startPriceDollar": None,
        "startPriceReal": None,
        "amenities": [],
        "categoriesName": []
    })

    # Iterate over the data and aggregate the category names
    for item in data:
        house_id = item["id"]
        
        # Initialize the dictionary for the house if it hasn't been already
        if aggregated_data[house_id]["houseId"] is None:
            aggregated_data[house_id]["houseId"] = item["id"]
            aggregated_data[house_id]["houseName"] = item["name"]
            aggregated_data[house_id]["startPriceDollar"] = item["average_price"]
            aggregated_data[house_id]["mainPicture"] = item["url"]
            aggregated_data[house_id]["roomAmount"] = item["room_amount"]
            aggregated_data[house_id]["bathroomAmount"] = item["bathroom_amount"]
            aggregated_data[house_id]["maxOccupancy"] = item["max_occupancy"]
            aggregated_data[house_id]["latitude"] = item["latitude"]
            aggregated_data[house_id]["longitude"] = item["longitude"]
            aggregated_data[house_id]["startPriceReal"] = item["price"]
            aggregated_data[house_id]["city"] = item["city"]
            aggregated_data[house_id]["condominium_id"] = item["condominium_id"]

        # Check if 'category_name' exists before appending
        if "category_name" in item:
            aggregated_data[house_id]["categoriesName"].append(item["category_name"])


    # Convert the aggregated data to a list
    aggregated_list = list(aggregated_data.values())

    return aggregated_list


def get_answer_from_prompt(prompt):
   
    # Get the current year
    current_year = datetime.now().year
    
    # Generate the SQL query
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sql_sys_content+f' If date year isnt mentioned, use the current year: {current_year}. house_category should not have more than 3 values in the list.'},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(schema)}
        ],
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
   
    # Simulate the response content
    res_mes = response.choices[0].message.content
    
    # Replace 'true' with 'True', 'false' with 'False', and 'null' with 'None'
    data_str = res_mes.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    
    # Safely convert string to dictionary
    filters = eval(data_str)
    
    
    
    
    filters = clean_filters(filters)

    logger.info(filters)


    
   
    
    # Handle the 'than' condition
    # 1 if its less than "<" and 0 if its more than ">"
    if 'than' in filters:
        #print(filters.get('than'))
        if filters.get('than') == 1:
            sym_str = '<'
            
        else:
            sym_str = '>'
        # Remove 'than' from filters
        filters.pop('than')
   
        
    

    
    
    having_query, where_query = generate_where_query_pandas(filters)
    
    sql_query = q3.replace("app.currency c ON 1 = 1", f"app.currency c ON 1 = 1\n{where_query}")
    sql_query = sql_query[:-1] + f"\n{having_query}"

    logger.info(sql_query)
   
    # Create a connection string
    connection_string = get_db_params()
    
    # Connect to the PostgreSQL database using SQLAlchemy
    engine = create_engine(connection_string)

    with engine.connect() as conn:
        # Execute the query and fetch the results
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        
        # Convert the result to a list of dictionaries
        data = [dict(row._asdict()) for row in rows]
        
        # Aggregate the categories
        aggregated_data = aggregate_categories(data)

        # Convert the aggregated data to a JSON response
        json_response = json.dumps({"results": aggregated_data})
        logger.info(json_response)
    return json_response, filters
        

# Function to interpret the JSON response and generate a human-readable response
def interpret_results(prompt, json_response, filters):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a bot assistant that will answer to the client, so you can never mention anything related to SQL or queries or similar. Keep the answer brief and relevant to the user's request. Never give details of any house result. You will just give a brief introduction if results are found without showing the results; and if not, then express that you are sorry."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Here are the results from the database query: {json_response}"}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    json_string = json_response
    
    # Step 1: Strip the outer quotes manually
    if json_string.startswith('"') and json_string.endswith('"'):
        json_string = json_string[1:-1]
    
    # Input data
    data = json.loads(json_response)
    
    # Access the list of houses inside 'results'
    houses = data['results']
    
    # Create a defaultdict to group the houses by categories
    grouped_data = defaultdict(list)
    
    # Iterate over each house and group by the categoriesName
    for house in houses:
        for category in house['categoriesName']:
            grouped_data[category].append(house)
    
    # Convert defaultdict back to a normal dict if needed
    grouped_data = dict(grouped_data)
    
    # Initialize variables to store the dates
    check_in_date = "Error"
    check_out_date = "Error"
    guests = "Error"

    for key, value in filters.items():
        if 'check_in' in key.lower():  # Checking for 'check_in' in the key name
            # Convert to dd-mm-yyyy format
            check_in_date = datetime.strptime(value, '%Y-%m-%d').strftime('%d-%m-%Y')
        elif 'check_out' in key.lower():  # Checking for 'check_out' in the key name
            # Convert to dd-mm-yyyy format
            check_out_date = datetime.strptime(value, '%Y-%m-%d').strftime('%d-%m-%Y')
        elif 'max_occupancy' in key.lower():
            guests = str(value)

    return json.dumps({"check_in": check_in_date, "check_out": check_out_date, "guests": guests, "results": response.choices[0].message.content, "options": grouped_data})

# Function to call the OpenAI API with the user's prompt and generate a human-readable response
def get_chatbot_response(prompt, history): 

    # Insert system message at the start of the history
    system_message = {"role": "system", "content": 'Always answer in the same language as the user. Check the past messages (history) for dates mentioned before; if no dates are found in either the history (past messages) or the current prompt, return an error message stating that dates must be provided.' + main_bot_instructions}

    # Ensure system message is always the first in the list
    history.insert(0, system_message)

    # Append the user message
    history.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        functions=[
            {
                "name": "get_answer_from_prompt",
                "description": "The get_answer_from_prompt function uses OpenAI's GPT-4 model to generate a SQL query based on a given prompt, executes the query on a PostgreSQL database, processes the results to include a URL for each entry, and returns the results in a JSON format. Never recommend any other website.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The user prompt"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        ]
    )

    # Check if the response includes a function call
    function_call = response.choices[0].message.function_call if hasattr(response.choices[0].message, 'function_call') else None

    # Call the function with the extracted arguments if present
    if function_call and function_call.name == "get_answer_from_prompt":
        function_args = json.loads(function_call.arguments)
        json_response, filters = get_answer_from_prompt(function_args["prompt"])
        return interpret_results(prompt, json_response, filters), filters
    else:
        filters = None
        return json.dumps({"results": response.choices[0].message.content}), filters

def load_database(engine, userid):   
    """ Load the user history from the PostgreSQL database into a DataFrame. """
    
    #query = f"SELECT * FROM alfred_dev_user.temp_db WHERE userid = '{userid}'"
    #df = pd.read_sql(query, engine)
    
    # Calculate the time one hour ago
    one_hour_ago = datetime.now() - timedelta(minutes=10)

    # Format the timestamp for the SQL query
    formatted_time = one_hour_ago.strftime('%Y-%m-%d %H:%M:%S')

    # Update the SQL query to filter results based on userid and timestamp
    query = f"""
    SELECT * FROM {smDBAppPostgresSecretDict['username_AI']}.temp_db 
    WHERE userid = '{userid}' 
    AND timestamp >= '{formatted_time}'
    """

    df = pd.read_sql(query, engine)
    
    
    return df

def create_history(userid, df):
    
    # Initialize an empty list to hold the formatted data
    formatted_history = []
    userid = str(userid)  # Ensure userid is a string

    # Check if the user ID exists in the DataFrame
    if userid in df['userid'].values:
        # Filter the DataFrame for the specific user ID
        #user_history = df[df['userid'] == userid]
        # Convert the Timestamp column to datetime
        #user_history['timestamp'] = pd.to_datetime(user_history['timestamp'])
        
        # Calculate the time limit (1 hour ago)
        #time_limit = datetime.now() - timedelta(hours=1)

        # Filter user history for entries from the past hour
        #recent_history = user_history[user_history['timestamp'] >= time_limit]
        
        # Select only the Prompt and Response columns
        filtered_history = df[['prompt', 'response']]
        
        # Iterate through the DataFrame rows
        for index, row in filtered_history.iterrows():
            # Append user prompt
            formatted_history.append({"role": "user", "content": row['prompt']})
            # Append assistant response
            formatted_history.append({"role": "assistant", "content": row['response']})

    return formatted_history

def update_history(userid, prompt, response, engine, filters):
    
    # New data to add
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    filters = str(filters)

    # Insert the new record into the PostgreSQL table
    new_data = pd.DataFrame({'userid': [userid], 'prompt': [prompt], 'response': [response], 'timestamp': [timestamp], 'filters': [filters]})
    new_data.to_sql('temp_db', engine, if_exists='append', index=False, schema= f"{smDBAppPostgresSecretDict['username_AI']}")
 
def hello(event, context):
    try:
        
        decoded_body = json.loads(event['body'])
               
        prompt = decoded_body.get('text', 'Key not found')
        userid = int(decoded_body.get('userId', 'Key not found'))

        # Create a connection string
        connection_string = get_db_params('AI')
    
        # Connect to the PostgreSQL database using SQLAlchemy
        engine = create_engine(connection_string)

        df = load_database(engine, userid)      

        history = create_history(userid, df)  
        
        raw_response, filters = get_chatbot_response(prompt, history)
        
        message_content = json.loads(raw_response)

        response_structure = {
            "results": message_content.get("results", "")  
        }

        for key, value in message_content.items():
            if key != "results":
                if isinstance(value, dict):
                    response_structure[key] = value
                else:
                    try:
                        response_structure[key] = json.loads(value)
                    except json.JSONDecodeError:
                        response_structure[key] = value

        results = response_structure.get('results', '') # Extract the results key
        
        update_history(userid, prompt, results, engine, filters)

        response = {
            "statusCode": 200,
            "body": json.dumps(response_structure, ensure_ascii=False, indent=4)
        }

        return response

    except Exception as e:
        logger.error(f"Erro genérico: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Erro inesperado",
                "error": str(e)
            }, indent=4)
        }
