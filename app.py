#!/usr/bin/env python3
"""
Demo Flask application with intentional security vulnerabilities for testing
"""

from flask import Flask, request, render_template_string, session, redirect, url_for
import sqlite3
import subprocess
import os
import pickle

app = Flask(__name__)
app.secret_key = "hardcoded_secret_key_123"  # Security Issue: Hardcoded secret

# Security Issue: SQL Injection vulnerability
@app.route('/user/<username>')
def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # VULNERABLE: Direct string interpolation allows SQL injection
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return str(result)

# Security Issue: Cross-Site Scripting (XSS) vulnerability
@app.route('/search')
def search():
    search_term = request.args.get('q', '')
    # VULNERABLE: Direct template rendering without escaping
    template = f"""
    <html>
    <body>
        <h1>Search Results for: {search_term}</h1>
        <p>You searched for: {search_term}</p>
    </body>
    </html>
    """
    return render_template_string(template)

# Security Issue: Command injection vulnerability
@app.route('/ping')
def ping():
    host = request.args.get('host', 'localhost')
    # VULNERABLE: Direct command execution with user input
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True, text=True)
    return f"<pre>{result.stdout}</pre>"

# Security Issue: Insecure deserialization
@app.route('/deserialize', methods=['POST'])
def deserialize():
    data = request.get_data()
    try:
        # VULNERABLE: Pickle deserialization of untrusted data
        obj = pickle.loads(data)
        return f"Deserialized: {obj}"
    except Exception as e:
        return f"Error: {e}"

# Security Issue: Path traversal vulnerability
@app.route('/file/<path:filename>')
def get_file(filename):
    # VULNERABLE: No path validation allows directory traversal
    file_path = f"./files/{filename}"
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    except Exception as e:
        return f"Error reading file: {e}"

# Security Issue: Weak session management
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # VULNERABLE: Weak password validation
    if username == "admin" and password == "password":
        session['user'] = username
        session['is_admin'] = True
        return "Login successful"
    return "Login failed"

# Security Issue: Lack of CSRF protection
@app.route('/delete_user/<user_id>', methods=['GET'])
def delete_user(user_id):
    # VULNERABLE: State-changing operation via GET without CSRF token
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM users WHERE id = {user_id}")
    conn.commit()
    conn.close()
    return "User deleted"

# Security Issue: Information disclosure
@app.route('/debug')
def debug():
    # VULNERABLE: Exposing sensitive debugging information
    debug_info = {
        'env_vars': dict(os.environ),
        'app_config': app.config,
        'session': dict(session)
    }
    return str(debug_info)

# Security Issue: Improper error handling
@app.route('/divide/<int:a>/<int:b>')
def divide(a, b):
    try:
        result = a / b
        return f"Result: {result}"
    except Exception as e:
        # VULNERABLE: Exposing internal error details
        return f"Internal Error: {e.__class__.__name__}: {str(e)}"

if __name__ == '__main__':
    # Security Issue: Running with debug=True in production
    app.run(debug=True, host='0.0.0.0', port=5000)