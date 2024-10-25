# ChatGPT Database Navigator

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [AWS Lambda Deployment](#aws-lambda-deployment)
4. [Prerequisites](#prerequisites)

---

## Project Overview

### Description
This project integrates OpenAI's ChatGPT with a PostgreSQL database, allowing users to interact with the database through natural language queries. Users can ask questions about properties in a specific location, and the system translates these queries into SQL to fetch the relevant information.

### Purpose
The primary goal is to provide a seamless interface for users to query a database without needing to know SQL, leveraging the natural language understanding capabilities of ChatGPT.

---

## Features

- **Natural Language Processing**: Convert user prompts into SQL queries.
- **Dynamic Schema Mapping**: Automatically map database columns to user queries.
- **Data Aggregation**: Organize retrieved data by property category for easy survey.
- **Sample Data Retrieval**: Fetch and display sample data for better context.
- **Error Handling**: Robust error handling and logging for better reliability.
- **Temporal-Based Queries**: Interpret and handle date-based enquiries effectively.
- **AWS Lambda Deployment**: Designed to be deployed as a serverless function on AWS Lambda.

---

## AWS Lambda Deployment

This codebase is designed to be deployed as a serverless function on AWS Lambda. It uses AWS services, such as AWS Lambda for running the application, AWS Secrets Manager to securely store and retrieve application secrets, and AWS RDS for providing the PostgreSQL database.

The main function, `hello(event, context)`, is the AWS Lambda handler, which gets invoked with each request. The function accepts a JSON payload from the `event` argument, containing two parameters: `text` (user's query) and `userId` (user's unique identifier).

---

## Prerequisites

- Python 3.6+
- PostgreSQL
- OpenAI API key
- `psycopg2` library
- AWS credentials (AWS Lambda, AWS Secrets Manager, AWS RDS)
