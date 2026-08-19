"""
Entrypoint redirect for web_app.
Allows running `python web_app/web_app.py` or `python web_app/app.py`.
"""
from app import app, init_app

if __name__ == '__main__':
    import os
    os.makedirs(".cache", exist_ok=True)
    init_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
