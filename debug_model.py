from predict import CreditRiskModel
import traceback

try:
    model = CreditRiskModel()
    print("Success")
except Exception:
    traceback.print_exc()
