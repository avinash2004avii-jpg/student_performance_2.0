import sys
import os
sys.path.append(os.getcwd())
try:
    from app import create_app
    app = create_app()
    print("✅ App initialized successfully. Routes are clean.")
except Exception as e:
    print(f"❌ Error initializing app: {e}")
    import traceback
    traceback.print_exc()
