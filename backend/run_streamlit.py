"""
Server wrapper to run Streamlit on port 8001
"""
import subprocess
import os

if __name__ == "__main__":
    # Change to backend directory
    os.chdir('/app/backend')
    
    # Run Streamlit on port 8001
    subprocess.run([
        'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '8001',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ])
