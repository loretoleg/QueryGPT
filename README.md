# ChatGPT Database Integration

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)

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
- **Sample Data Retrieval**: Fetch and display sample data for better context.
- **Error Handling**: Robust error handling and logging for better reliability.

---

## Prerequisites

- Python 3.6+
- PostgreSQL
- OpenAI API key
- `psycopg2` library