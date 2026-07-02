"""conftest.py — делает пакет retrieval_fairness импортируемым из tests/."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
